"""
inference.py
─────────────────────────────────────────────────────────────────────────────
Load the QLoRA routing adapter and route a customer complaint.

Returns, for any input text:
    * top_label        – the predicted routing queue
    * confidence       – softmax probability of the top route
    * top_3            – the 3 most likely queues with their probabilities

Also provides `expected_calibration_error` so you can sanity-check whether the
reported confidence is trustworthy (does "80% confident" mean right ~80% of the
time?).

CLI:
    uv run python inference.py --text "I was charged twice for one purchase"
    uv run python inference.py            # runs a few built-in demo messages

Programmatic:
    from inference import RoutingModel
    rm = RoutingModel()                   # loads base + adapter (4-bit)
    print(rm.route("My card was swallowed by the ATM"))
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from routing_pipeline import CONFIG, RouteScorer
from routing_taxonomy import ROUTES, ID2ROUTE


@dataclass
class RouteResult:
    top_label: str
    confidence: float
    top_3: list[tuple[str, float]]

    def __repr__(self) -> str:
        top3 = ", ".join(f"{r} ({p:.2f})" for r, p in self.top_3)
        return (f"RouteResult(top={self.top_label!r}, conf={self.confidence:.3f}, "
                f"top3=[{top3}])")


class RoutingModel:
    """Base Qwen (4-bit) + optional LoRA adapter, wrapped around RouteScorer."""

    def __init__(self, adapter_dir=None, use_base_only: bool = False,
                 cfg=CONFIG, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.cfg = cfg
        adapter_dir = adapter_dir or cfg.adapter_dir

        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, quantization_config=bnb, device_map=device,
        )

        self.adapter_loaded = False
        if not use_base_only:
            from pathlib import Path
            if Path(adapter_dir).exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_dir))
                self.adapter_loaded = True
            else:
                print(f"[inference] No adapter at {adapter_dir} — using BASE model "
                      "(zero-shot). Train first to load the fine-tuned router.")
        model.eval()
        self.model = model
        self.scorer = RouteScorer(model, self.tok, cfg)

    def route(self, text: str) -> RouteResult:
        return self.route_batch([text])[0]

    def route_batch(self, texts: list[str]) -> list[RouteResult]:
        probs = self.scorer.score_texts(list(texts))
        results = []
        for p in probs:
            order = np.argsort(-p)
            top3 = [(ID2ROUTE[int(i)], float(p[i])) for i in order[:3]]
            results.append(RouteResult(ID2ROUTE[int(order[0])], float(p[order[0]]), top3))
        return results


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    ECE: weighted gap between confidence and accuracy across confidence bins.
    0 = perfectly calibrated. Uses the top-1 confidence of each prediction.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


_DEMOS = [
    "I was charged twice for the same coffee purchase this morning.",
    "My new card still hasn't arrived after three weeks.",
    "The ATM took my card and didn't give it back.",
    "How do I verify my identity to lift the limit?",
    "Why is the exchange rate different from what Google shows?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None, help="complaint to route")
    ap.add_argument("--base-only", action="store_true", help="ignore adapter (zero-shot)")
    args = ap.parse_args()

    rm = RoutingModel(use_base_only=args.base_only)
    mode = "FINE-TUNED adapter" if rm.adapter_loaded else "BASE (zero-shot)"
    print(f"\nRouting with: {mode}\n" + "─" * 60)

    texts = [args.text] if args.text else _DEMOS
    for t, r in zip(texts, rm.route_batch(texts)):
        print(f"\n> {t}")
        print(f"  → {r.top_label}  (confidence {r.confidence:.1%})")
        print("    top-3: " + " | ".join(f"{lbl} {p:.1%}" for lbl, p in r.top_3))


if __name__ == "__main__":
    main()

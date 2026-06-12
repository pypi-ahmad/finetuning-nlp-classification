"""
inference.py
─────────────────────────────────────────────────────────────────────────────
Two-stage hierarchical triage at inference time.

    stage 1 → predict L1 (product) over all products
    stage 2 → predict L2 (issue) over ONLY the issues under the chosen L1

Returns, for any complaint:
    l1, l1_confidence, l1_top3
    l2, l2_confidence, l2_top3      (None if the chosen L1 has no issue set)

The label hierarchy is loaded from <adapter_dir>/hierarchy.json (saved at train
time). If absent, it is rebuilt from the curation pipeline.

CLI:
    uv run python inference.py --text "A debt collector keeps calling about a paid loan"
    uv run python inference.py                 # built-in demos
    uv run python inference.py --base-only     # zero-shot (no adapter)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from triage_config import CONFIG
from triage_model import HierScorer, l1_messages, l2_messages, load_base_model


def load_hierarchy(adapter_dir) -> dict[str, list[str]]:
    p = Path(adapter_dir) / "hierarchy.json"
    if p.exists():
        return json.loads(p.read_text())
    # fallback: rebuild from data pipeline
    from triage_data import build_dataframe
    _, hier = build_dataframe()
    return hier


@dataclass
class TriageResult:
    l1: str
    l1_confidence: float
    l1_top3: list[tuple[str, float]]
    l2: str | None
    l2_confidence: float | None
    l2_top3: list[tuple[str, float]] | None

    def pretty(self) -> str:
        s = [f"L1: {self.l1}  ({self.l1_confidence:.0%})",
             "    top-3: " + " | ".join(f"{l} {p:.0%}" for l, p in self.l1_top3)]
        if self.l2 is not None:
            s += [f"L2: {self.l2}  ({self.l2_confidence:.0%})",
                  "    top-3: " + " | ".join(f"{l} {p:.0%}" for l, p in self.l2_top3)]
        return "\n".join(s)


class TriageRouter:
    def __init__(self, adapter_dir=None, base_only: bool = False, temperature_l1: float = 1.0):
        adapter_dir = adapter_dir or CONFIG.adapter_dir
        self.hierarchy = load_hierarchy(adapter_dir)
        self.l1_labels = sorted(self.hierarchy.keys())
        self.temperature_l1 = temperature_l1
        self.model, self.tok = load_base_model(
            CONFIG, adapter_dir=None if base_only else adapter_dir)
        self.adapter_loaded = (not base_only) and Path(adapter_dir).exists()
        self.scorer = HierScorer(self.model, self.tok, CONFIG)

    @staticmethod
    def _top3(prob: dict):
        return sorted(prob.items(), key=lambda kv: -kv[1])[:3]

    def route_batch(self, texts: list[str]) -> list[TriageResult]:
        # Stage 1
        l1_probs = self.scorer.score_l1(texts, self.l1_labels,
                                        temperature=self.temperature_l1)
        results = []
        # Stage 2 — only for texts whose predicted L1 has an issue set
        pred_l1 = [max(p, key=p.get) for p in l1_probs]
        l2_inputs = [(i, t, l1) for i, (t, l1) in enumerate(zip(texts, pred_l1))
                     if self.hierarchy.get(l1)]
        l2_probs_by_i: dict[int, dict] = {}
        if l2_inputs:
            idxs = [i for i, _, _ in l2_inputs]
            l2_texts = [t for _, t, _ in l2_inputs]
            l2_l1s = [l1 for _, _, l1 in l2_inputs]
            l2_scored = self.scorer.score_l2(l2_texts, l2_l1s, self.hierarchy)
            l2_probs_by_i = dict(zip(idxs, l2_scored))

        for i, (t, p1) in enumerate(zip(texts, l1_probs)):
            l1 = pred_l1[i]
            top1_l1 = self._top3(p1)
            if i in l2_probs_by_i:
                p2 = l2_probs_by_i[i]
                top1_l2 = self._top3(p2)
                results.append(TriageResult(l1, p1[l1], top1_l1,
                                            top1_l2[0][0], p2[top1_l2[0][0]], top1_l2))
            else:
                results.append(TriageResult(l1, p1[l1], top1_l1, None, None, None))
        return results

    def route(self, text: str) -> TriageResult:
        return self.route_batch([text])[0]


_DEMOS = [
    "A debt collector keeps calling me at work about a loan I already paid off.",
    "There is a hard inquiry on my credit report that I never authorized.",
    "My mortgage servicer applied my payment to the wrong month and charged a late fee.",
    "Someone made a fraudulent charge on my credit card and the bank won't refund it.",
    "My wire transfer to another bank never arrived after five business days.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--base-only", action="store_true")
    args = ap.parse_args()

    router = TriageRouter(base_only=args.base_only)
    mode = "FINE-TUNED adapter" if router.adapter_loaded else "BASE model (zero-shot)"
    print(f"Triage routing with: {mode}\n" + "=" * 64)
    texts = [args.text] if args.text else _DEMOS
    for t, r in zip(texts, router.route_batch(texts)):
        print(f"\n> {t}\n" + r.pretty())


if __name__ == "__main__":
    main()

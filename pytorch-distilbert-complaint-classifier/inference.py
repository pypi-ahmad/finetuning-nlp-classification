#!/usr/bin/env python
"""
inference.py — Predict banking intents with a fine-tuned DistilBERT.

The notebook saves the best model in HuggingFace format to checkpoints/best_model/
(model + tokenizer). This script reloads it and predicts intents with per-class
confidence, for either a single query or a batch CSV/TXT file.

Usage
-----
Single text:
    uv run python inference.py --text "My card still hasn't arrived"

Batch (one query per line, .txt):
    uv run python inference.py --input queries.txt --top-k 3

Batch (CSV with a 'text' column):
    uv run python inference.py --input queries.csv --output preds.csv

Train the model first by running pytorch_distilbert_complaint_classifier.ipynb.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        sys.exit(
            f"Model directory not found: {model_dir}\n"
            f"Train the model first by running the notebook "
            f"(it saves to {model_dir})."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    texts,
    model,
    tokenizer,
    device: torch.device,
    max_len: int = 64,
    top_k: int = 3,
    batch_size: int = 64,
):
    """Return, per input, a list of (intent, probability) sorted high→low, length top_k."""
    id2label = model.config.id2label
    results = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk, truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs = F.softmax(logits.float(), dim=1).cpu()
        k = min(top_k, probs.shape[1])
        top_p, top_i = probs.topk(k, dim=1)
        for p_row, i_row in zip(top_p, top_i):
            results.append([(id2label[int(i)], float(p)) for p, i in zip(p_row, i_row)])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_inputs(path: Path):
    """Read queries from a .txt (one per line) or .csv (a 'text' column)."""
    if path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if "text" not in df.columns:
            sys.exit(f"CSV {path} must contain a 'text' column. Found: {list(df.columns)}")
        return df["text"].astype(str).tolist()
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict banking intents with fine-tuned DistilBERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="A single query to classify")
    src.add_argument("--input", help="Path to a .txt (one query/line) or .csv (with 'text' column)")

    parser.add_argument("--model-dir", default="checkpoints/best_model",
                        help="Directory with the saved HF model + tokenizer")
    parser.add_argument("--output", default=None,
                        help="Optional CSV path to write batch predictions")
    parser.add_argument("--top-k", type=int, default=3, help="How many intents to show per query")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    model, tokenizer = load_model(Path(args.model_dir), device)
    print(f"Device : {device}   |   {len(model.config.id2label)} intents loaded\n")

    texts = [args.text] if args.text else read_inputs(Path(args.input))
    preds = predict(texts, model, tokenizer, device, max_len=args.max_len, top_k=args.top_k)

    for text, p in zip(texts, preds):
        print(f'Q: "{text}"')
        for name, prob in p:
            print(f"    {prob*100:5.1f}%  {name}")
        print()

    if args.output:
        import pandas as pd
        rows = []
        for text, p in zip(texts, preds):
            row = {"text": text, "predicted_intent": p[0][0], "confidence": round(p[0][1], 4)}
            for rank, (name, prob) in enumerate(p, 1):
                row[f"top{rank}_intent"] = name
                row[f"top{rank}_prob"] = round(prob, 4)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Wrote {len(rows)} predictions → {args.output}")


if __name__ == "__main__":
    main()

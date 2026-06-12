# DistilBERT Complaint / Intent Classifier (Banking77)

Fine-tuning **DistilBERT** to classify short banking customer messages into **77 fine-grained
intents** — a practical, learning-focused PyTorch + Transformers NLP project with a clean,
from-scratch training loop (no `Trainer` black box) and an honest classical baseline.

## Task

Multi-class text classification. Given a free-text customer query
(*"My card still hasn't arrived"*, *"Why was I charged twice?"*), predict one of **77 banking
intents** (e.g. `card_arrival`, `declined_transaction`, `extra_charge_on_statement`). This is the
routing step a support system uses to auto-answer or escalate a complaint.

## Dataset

**[Banking77](https://huggingface.co/datasets/PolyAI/banking77)** — 13,083 real online-banking
queries labelled with 77 intents (10,003 train / 3,080 test). Loaded via HuggingFace `datasets`
from the auto-converted **parquet branch** (`revision="refs/convert/parquet"`), because
`datasets >= 5` no longer runs the script-based loader the original repo ships. A stratified 10% of
train is held out as a validation set for model selection; the test split is touched **only** for
the final report.

## Model

- **`distilbert-base-uncased`** (~66M params) via `AutoModelForSequenceClassification` with a fresh
  77-way classification head. Human-readable intent names are baked into the model config
  (`id2label` / `label2id`).
- Loss: cross-entropy (computed inside the model).
- **Mixed precision: bf16** autocast on Ada+ GPUs (no `GradScaler` needed; automatic fp16 + scaler
  fallback on older GPUs, fp32 on CPU).

## Training procedure

| Setting | Value |
|---|---|
| Optimizer | AdamW (no weight decay on bias / LayerNorm) |
| Learning rate | 5e-5, **linear warmup (10%) → linear decay** |
| Epochs | 8 (best validation macro-F1 around epoch 5; later epochs overfit) |
| Batch size | 32 train / 64 eval, **dynamic padding** via `DataCollatorWithPadding` |
| Max length | 64 tokens (covers >99% of queries) |
| Grad clipping | 1.0 |
| Model selection | checkpoint the **best validation macro-F1** |
| Seed | 42 (Python / NumPy / Torch CPU+CUDA) |

Custom PyTorch loop: `Dataset` → `DataLoader` → autocast forward → backward → clip → step →
scheduler, with per-epoch validation. Trains in **~2–3 minutes** on an 8 GB RTX 4060.

## Key metrics (held-out test set)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF (1–2 gram) + Logistic Regression | ~0.890 | ~0.890 | ~0.890 |
| **DistilBERT (fine-tuned)** | **~0.923** | **~0.923** | **~0.923** |

DistilBERT beats a strong bag-of-words baseline by **~3 accuracy points**. The gain concentrates on
near-synonym intent pairs that TF-IDF cannot disambiguate. **Confidence is usable for selective
prediction**: thresholding the max-softmax probability at 0.9 keeps ~88% coverage at ~97% accuracy,
so low-confidence queries can be routed to a human. (Exact numbers regenerate when you run the
notebook.)

The notebook also produces: training curves, a 77×77 confusion matrix, top confusion pairs,
confidence/coverage analysis, and error analysis on real misclassified queries —
see `figures/`.

## Setup

```bash
git clone https://github.com/pypi-ahmad/pytorch-distilbert-complaint-classifier.git
cd pytorch-distilbert-complaint-classifier
```


Requires [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU with recent drivers (CPU works too,
just slower). Python 3.12.10 and a CUDA 12.8 PyTorch build are pinned.

```bash
cd pytorch-distilbert-complaint-classifier
uv sync                       # creates .venv, installs torch (cu128), transformers, datasets, ...
```

## How to run

**1 · Regenerate the notebook (optional — it's already committed with outputs):**
```bash
uv run python generate_notebook.py
```

**2 · Run the notebook** (trains, evaluates, saves checkpoints + figures):
```bash
uv run jupyter lab pytorch_distilbert_complaint_classifier.ipynb
# or headless, end-to-end:
uv run jupyter nbconvert --to notebook --execute --inplace \
    pytorch_distilbert_complaint_classifier.ipynb
```
Running the notebook saves the best model to `checkpoints/best_model/` (HF format) and
`checkpoints/best_model.pt` (raw state-dict).

**3 · Predict with the trained model** via `inference.py`:
```bash
# single query, top-3 intents with confidence
uv run python inference.py --text "My card still hasn't arrived"

# batch: one query per line (.txt) or a CSV with a 'text' column
uv run python inference.py --input queries.txt --top-k 3
uv run python inference.py --input queries.csv --output predictions.csv
```

## Project layout

```
pytorch-distilbert-complaint-classifier/
├── pytorch_distilbert_complaint_classifier.ipynb   # main deliverable (executed, with outputs)
├── generate_notebook.py    # builds the notebook (source of truth, diff-friendly)
├── inference.py            # CLI: single / batch prediction with per-class confidence
├── pyproject.toml          # uv project + pinned cu128 torch index
├── checkpoints/            # best_model.pt + best_model/ (HF format)  [gitignored]
├── figures/                # generated plots
└── README.md
```

## Limitations

Single-seed run; max-softmax confidence is uncalibrated (ranks well but over-confident — temperature
scaling would fix it); Banking77 is clean, English-only, single-intent, and closed-set (no
out-of-scope class). See the notebook's final section for details and next steps.

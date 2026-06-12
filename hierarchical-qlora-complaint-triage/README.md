# Hierarchical Complaint Triage with QLoRA (L1 product → L2 issue)

A two-level financial-complaint triage system fine-tuned with **QLoRA** on an
**8 GB** GPU. Goes beyond single-label intent classification: it predicts a
**label hierarchy** and routes in **two conditioned stages**, with **calibrated
confidence**, an **active-learning** loop, and full error analysis.

```
Stage 1:  complaint ─────────────────► L1 product   (1 of 9)
Stage 2:  complaint + chosen L1 ─────► L2 issue      (1 of the issues under that L1)
```

---

## Datasets

### 1. Banking77 — clean intent benchmark (L1 augmentation)
- Link: https://huggingface.co/datasets/PolyAI/banking77
- Loaded via 🤗 `datasets`:
  ```python
  from datasets import load_dataset
  ds = load_dataset("PolyAI/banking77")
  ```
- Its 77 intents are keyword-mapped to the three consumer-banking L1 products
  (`triage_data.banking77_to_l1`). No CFPB-style issue labels → **does not feed
  stage 2**.

### 2. CFPB Consumer Complaint Database — the real hierarchy
- Portal: https://www.consumerfinance.gov/data-research/consumer-complaints/
- API: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/
- Bulk CSV: https://files.consumerfinance.gov/ccdb/complaints.csv.zip
- We use **`complaint_what_happened`** (narrative), **`product`** → **L1**, and
  **`issue`** → **L2**.

**Download (recommended — balanced API subset, no multi-GB file):**
```bash
uv run python download_cfpb.py --per-product 2000
# → data/raw/cfpb/cfpb_subset.parquet  (~8k narrative complaints, 9 products)
```
`download_cfpb.py` consolidates CFPB's historical product renames into 9 canonical
L1 products and pulls a class-balanced, narrative-only sample per product.

<details><summary>Alternative: bulk CSV / raw API page (per the task spec)</summary>

```bash
mkdir -p data/raw/cfpb
# bulk (large — hundreds of MB zipped, ~3.8M rows, ~90% credit-reporting)
wget -O data/raw/cfpb/complaints.csv.zip https://files.consumerfinance.gov/ccdb/complaints.csv.zip
unzip data/raw/cfpb/complaints.csv.zip -d data/raw/cfpb
# single API page
curl "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=1000&from=0" \
     -o data/raw/cfpb/page_0.json
```
</details>

## Preprocessing choices (`triage_data.py`)
- **Product consolidation**: 19 historical CFPB product names → **9 canonical L1**.
- **Narrative cleaning**: collapse PII redactions (`XXXX → [redacted]`), drop
  near-empty narratives, truncate to 1200 chars.
- **Anti-sparsity**: keep only `issue` (L2) classes with **≥ 80** samples; **cap**
  each issue at 400 for balance.
- **Banking77** capped at 2000 rows so CFPB stays the backbone.
- **Stratified** train/val/test/**pool** split (pool = unlabeled reserve for
  active learning).

## Model & method
- **`Qwen/Qwen2.5-1.5B-Instruct`**, 4-bit **NF4** QLoRA — fits *training* on 8 GB.
- **One adapter, multi-task**: stage-1 (every row → L1) + stage-2 (CFPB rows → L2
  under gold L1) instruction examples.
- **Constrained label scoring** (no free generation): each stage scores its
  candidate labels and softmaxes length-normalised log-likelihoods → valid label
  + calibrated probability (top-3, confidence). See `HierScorer`.
- **Part B — second backbone (head-to-head):** the notebook re-runs the full
  two-stage pipeline on **`ibm-granite/granite-4.1-3b`** (dense, ungated, ~3B;
  same LoRA targets) and compares Qwen-1.5B vs Granite-3B on L1/L2 macro-F1 and
  hierarchical exact-match — the harder L2 step is where the bigger model is
  expected to pay off. Runs at micro-batch 1 (704 tokens) to stay within 8 GB.

## Training config (8 GB VRAM)
| Knob | Value |
|---|---|
| Quantization | 4-bit NF4 + double-quant, bf16 compute |
| LoRA | r=16, α=32, dropout 0.05, all attn+MLP proj (~18M params) |
| Effective batch | 2 × 8 grad-accum = 16 (micro-batch 2 for 640-token seqs) |
| Optimizer | paged_adamw_8bit, lr 2e-4, cosine, 3% warmup |
| Memory | gradient checkpointing, `max_length=640` |
| Epochs | 2 (1 in QUICK_MODE) |
| Peak VRAM | ~5–6 GB |

## Metrics reported
**L1 macro-F1**, **L2 macro-F1**, **hierarchical exact-match** (both levels right),
**top-3 L1 accuracy**, **ECE** (calibration). Compared across **TF-IDF+LogReg**,
**zero-shot LLM**, and **QLoRA fine-tuned**.

Sanity baseline already measured: **TF-IDF+LogReg → L1 macro-F1 0.82, L2 0.54** —
L2 is the hard part the LLM should improve.

---

## How to run locally

```bash
cd hierarchical-qlora-complaint-triage

# 1. environment (Python 3.12.10, CUDA wheels)
uv sync

# 2. get the CFPB subset (Banking77 + Qwen download on first notebook run)
uv run python download_cfpb.py --per-product 2000

# 3. (re)generate the notebook  — optional, it's committed
uv run python make_notebook.py

# 4. launch and Run All
uv run jupyter lab hierarchical_qlora_complaint_triage.ipynb
```

> The notebook has a **`QUICK_MODE`** flag (top cell, default `True`) that
> validates the entire pipeline — including a real QLoRA train + the
> active-learning loop — on subsampled data in a few minutes. Set it to `False`
> for the full run (the active-learning section retrains the adapter several
> times and is the slowest part).

### Command-line inference (after training)
```bash
uv run python inference.py --text "A debt collector keeps calling about a paid loan"
uv run python inference.py                 # built-in demos
uv run python inference.py --base-only     # zero-shot, no adapter
```

### Inspect data / baselines without a GPU
```bash
uv run python triage_data.py        # curation summary + hierarchy + splits
uv run python triage_baselines.py   # TF-IDF + LogReg L1/L2 macro-F1
uv run python calibration.py        # temperature-scaling self-test
```

## Project layout
```
hierarchical-qlora-complaint-triage/
├── hierarchical_qlora_complaint_triage.ipynb   # ← main deliverable
├── make_notebook.py            # regenerates the notebook (nbformat)
├── download_cfpb.py            # balanced CFPB subset via the public API
├── triage_config.py            # all knobs (model, data, LoRA, training)
├── triage_data.py              # CFPB+B77 curation, hierarchy, splits
├── triage_model.py             # prompts, HierScorer, SFT records, hier_evaluate
├── triage_baselines.py         # TF-IDF + Logistic Regression
├── calibration.py              # temperature scaling / isotonic / ECE
├── active_learning.py          # uncertainty sampling strategies
├── inference.py                # two-stage TriageRouter (L1+L2, confidence, top-3)
├── pyproject.toml / uv.lock    # uv-managed env
├── data/raw/cfpb/              # downloaded subset (gitignored)
├── figures/                    # plots saved by the notebook
└── outputs/qlora-triage-adapter/  # saved adapter + hierarchy.json
```

## Limitations
Banking77↔CFPB domain gap (B77 augments L1 only, 3 of 9 products); CFPB labels are
self-assigned and noisy; 4-bit slightly caps the ceiling vs fp16; `issue` used as
L2 (`sub_issue` as L3 is too sparse without more data); candidate scoring cost
grows with the label set (cache the prompt prefix or add a classification head at
scale). See the notebook's final section for the full deployment recommendation.

## Setup

```bash
git clone https://github.com/pypi-ahmad/hierarchical-qlora-complaint-triage.git
cd hierarchical-qlora-complaint-triage
```

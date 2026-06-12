# LoRA / QLoRA — Dispute Reason Classification & Support-Ticket Routing

A compact, learning-focused **QLoRA fine-tuning** project that routes customer
banking complaints into the correct **support queue**. Built to train *and* run
on an **8 GB** laptop GPU.

> Sibling to `qlora-complaint-intent-classifier`, but **routing-focused**: coarse
> operational queues, top-3 routing usefulness, misrouting analysis, and a
> confidence/calibration check for confidence-gated routing.

---

## Task
8-way single-label classification framed as **support-ticket routing**: read a
free-text complaint → assign it to one of 8 operational queues. Evaluated on what
a real triage system cares about — accuracy, **macro-F1**, **top-3 accuracy**
(is the right queue in the agent's short list?), and **calibrated confidence**.

## Dataset
[**Banking77**](https://huggingface.co/datasets/PolyAI/banking77) (13k queries,
77 fine intents) **remapped into 8 routing queues** via an explicit, audited map
in `routing_taxonomy.py`:

`Card Disputes & Fraud` · `Payments & Transfers` · `Card Operations` ·
`ATM & Cash` · `Account Access & Security` · `Top-Up & Funding` ·
`Identity & Verification` · `Fees, Rates & Product Info`

Splits are **stratified by queue**. `assert_full_coverage()` fails loudly if the
dataset and the taxonomy ever drift apart.

## Model choice
**`Qwen/Qwen2.5-1.5B-Instruct`** — a strong, small instruct model that:
- runs **and trains** on 8 GB in 4-bit,
- already follows the routing rubric zero-shot (a real baseline to beat),
- is big enough that a low-rank adapter has something to specialise.

We read a label out of this generative model by **constrained label scoring**:
score all 8 queue names as candidate completions, softmax their length-normalised
log-likelihoods → a valid label + per-queue probability (→ top-3 + calibration),
identical for the base and fine-tuned models. See `RouteScorer` in
`routing_pipeline.py`.

**Part B — second backbone (head-to-head).** The notebook also runs the *same*
pipeline on **`ibm-granite/granite-4.1-3b`** (dense, ungated, ~3B; identical LoRA
target modules) and compares Qwen-1.5B vs Granite-3B on accuracy/macro-F1/top-3
+ calibration, with analysis and a `figures/granite_vs_qwen.png` chart.
Verified to fit 8 GB (train ~3.5 GB at micro-batch 2, eval ~3.3 GB).

## Training setup (tuned for 8 GB VRAM)

```bash
git clone https://github.com/pypi-ahmad/lora-dispute-ticket-routing.git
cd lora-dispute-ticket-routing
```

| Knob | Value |
|---|---|
| Base quantization | 4-bit **NF4** + double-quant, bf16 compute |
| LoRA | `r=16`, `α=32`, dropout `0.05`, targets = all attn + MLP proj (~18M params, ~1.2%) |
| Effective batch | **4 × 4 grad-accum = 16** (micro-batch 4 keeps the 152k-vocab loss tensor in budget) |
| Optimizer | `paged_adamw_8bit`, lr `2e-4`, cosine, 3% warmup |
| Memory | gradient checkpointing on, `max_length=384` |
| Epochs | 2 |
| **Peak VRAM** | **~4.7 GB train · ~3–5 GB eval** (fits 8 GB with headroom) |

Loss is computed **only on the queue name** (`completion_only_loss=True`), not on
the rubric.

## Key metrics
The notebook produces a before/after table + `figures/base_vs_finetuned.png`:

| Metric | Zero-shot base | QLoRA fine-tuned |
|---|---|---|
| Accuracy | _run notebook_ | _run notebook_ |
| Macro-F1 | _run notebook_ | _run notebook_ |
| Top-3 accuracy | _run notebook_ | _run notebook_ |
| ECE (calibration error) | _run notebook_ | _run notebook_ |

(Quick check on a 160-ticket sample: zero-shot ≈ **0.72 acc / 0.72 macro-F1 /
0.91 top-3** — fine-tuning is expected to lift accuracy/F1 meaningfully.)

Other artifacts: `figures/label_distribution.png`, `confusion_matrix.png`,
`calibration.png`, and the saved adapter in `outputs/qlora-routing-adapter/`.

---

## Setup and Run

Prereqs: an NVIDIA GPU (~8 GB), [`uv`](https://docs.astral.sh/uv/), and a CUDA
driver. The model (~2.9 GB) and dataset download on first run from the HF Hub.

```bash
cd lora-dispute-ticket-routing

# 1. Create the environment (Python 3.12.10) and install everything
uv sync

# 2. (Re)generate the notebook from source — optional, it's already committed
uv run python make_notebook.py

# 3. Launch Jupyter and run top-to-bottom (Run All)
uv run jupyter lab lora_dispute_ticket_routing.ipynb
#   or:  uv run jupyter notebook lora_dispute_ticket_routing.ipynb
```

Run the notebook **top to bottom**: it loads the data, evaluates the zero-shot
baseline, fine-tunes the QLoRA adapter, then evaluates + plots before/after,
confusion matrix, qualitative examples, and calibration.

### Command-line inference (after training)
```bash
# route one complaint with the fine-tuned adapter
uv run python inference.py --text "I was charged twice for one purchase"

# built-in demo messages
uv run python inference.py

# compare against the zero-shot base model
uv run python inference.py --base-only
```

### Sanity-check the taxonomy without a GPU
```bash
uv run python routing_taxonomy.py     # prints the 77 → 8 intent mapping summary
```

---

## Project layout
```
lora-dispute-ticket-routing/
├── lora_dispute_ticket_routing.ipynb   # ← main deliverable
├── make_notebook.py                    # regenerates the notebook (nbformat)
├── routing_taxonomy.py                 # Banking77 (77) → 8 routing queues, audited
├── routing_pipeline.py                 # dataset, prompt, RouteScorer, metrics (shared)
├── inference.py                        # route() → top label + confidence + top-3; ECE
├── pyproject.toml / uv.lock            # uv-managed env (Python 3.12.10, CUDA wheels)
├── figures/                            # plots saved by the notebook
└── outputs/qlora-routing-adapter/      # saved LoRA adapter (created on train)
```

## Why LoRA / QLoRA (the short version)
Full fine-tuning a 1.5B model needs ~18 GB just for optimizer state. **LoRA**
trains only small low-rank matrices (~1% of params); **QLoRA** additionally
stores the frozen base in 4-bit, so the whole thing *trains* in ~5 GB. The
trade-off — 4-bit is slightly lossy — is measured in the notebook, not assumed.

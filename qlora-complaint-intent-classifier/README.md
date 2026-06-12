# QLoRA Fine-Tuning — Banking Intent Classifier

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` on the Banking77 dataset using QLoRA,
targeting an NVIDIA GPU with 8 GB VRAM.

---

## Task

**Multi-class intent classification** — given a short banking customer query,
predict one of 77 fine-grained intent labels (e.g. `declined_card_payment`,
`change_pin`, `lost_or_stolen_card`).

This is a realistic applied NLP task that tests whether fine-tuning on a narrow
domain outperforms zero-shot prompting of the same base model.

---

## Dataset

**Banking77** ([PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77))

| Split | Examples | Classes |
|---|---|---|
| Train | 13,083 | 77 |
| Test | 3,080 | 77 |

Labels are near-uniformly distributed (~170 train examples per intent).
The intents are fine-grained and semantically close — making zero-shot prompting
hard and fine-tuning valuable.

---

## Model Choice

**`Qwen/Qwen2.5-1.5B-Instruct`**

| | |
|---|---|
| Parameters | 1.5 B |
| Architecture | Decoder-only Transformer (GQA, RoPE) |
| Context window | 32,768 tokens |
| BF16 weight footprint | ~3.0 GB |
| **4-bit NF4 footprint** | **~0.9 GB** |
| Fits 8 GB VRAM for QLoRA fine-tuning? | **Yes** |

Why this model:
- Small enough that QLoRA fits in 8 GB VRAM with ~3 GB headroom for activations.
- Instruction-tuned — understands the system-prompt + classification format.
- No gating on Hugging Face.
- Competitive benchmark scores for its size class.

---

## Setup

```bash
git clone https://github.com/pypi-ahmad/qlora-complaint-intent-classifier.git
cd qlora-complaint-intent-classifier
```


### Requirements
- Ubuntu 22.04 / 24.04 / 26.04
- NVIDIA GPU with ≥ 8 GB VRAM + CUDA 12.x / 13.x
- [`uv`](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.12.10 (managed automatically by uv)

### Install dependencies

```bash
cd qlora-complaint-intent-classifier
uv sync
```

This installs PyTorch 2.12+ (CUDA 12.8 build), Transformers 5.x, PEFT 0.19,
TRL 1.5, BitsAndBytes 0.49, and all other dependencies into an isolated venv at `.venv/`.

### Register the Jupyter kernel

```bash
uv run python -m ipykernel install --user \
    --name qlora-complaint-intent-classifier \
    --display-name "Python 3.12 (qlora-env)"
```

---

## Running the Notebook

```bash
# Activate the venv (optional — uv run handles it automatically)
source .venv/bin/activate

# Launch Jupyter
uv run jupyter notebook qlora_complaint_intent_classifier.ipynb
```

Or open `qlora_complaint_intent_classifier.ipynb` directly in VS Code / JupyterLab.

**Expected runtime on RTX 4060 (8 GB):**

| Step | Time |
|---|---|
| Dataset download & format | ~1–2 min |
| Baseline evaluation (200 examples) | ~3–5 min |
| QLoRA fine-tuning (3 epochs) | ~25–40 min |
| Full test evaluation (3,080 examples) | ~10–15 min |

Total: approximately **45–60 minutes** end-to-end.

---

## Training Summary

| Param | Value |
|---|---|
| Quantisation | 4-bit NF4 + double quant |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o + gate/up/down proj |
| Trainable params | ~4.7 M (0.31% of 1.5 B) |
| Effective batch size | 4 × 4 = 16 |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine decay) |
| Optimiser | paged AdamW 8-bit |
| Gradient checkpointing | yes |
| Max sequence length | 256 tokens |

---

## Evaluation

Metrics on the full Banking77 test set (3,080 examples):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Base model (zero-shot, 200 examples) | see notebook | see notebook |
| QLoRA fine-tuned (3 epochs, 3,080 examples) | see notebook | see notebook |

Run the notebook to get the exact numbers for your hardware and run.

---

## Inference Script

After training, use `inference.py` to run predictions:

```bash
# Single query
uv run python inference.py "My card was declined at the supermarket"

# Interactive mode
uv run python inference.py

# Batch mode (one query per line)
uv run python inference.py --batch queries.txt
```

---

## Project Structure

```
qlora-complaint-intent-classifier/
├── pyproject.toml                         # uv project config + deps
├── .python-version                        # pins Python 3.12.10
├── .venv/                                 # isolated venv (created by uv sync)
├── qlora_complaint_intent_classifier.ipynb  # main notebook
├── inference.py                           # standalone inference helper
├── make_notebook.py                       # generates the .ipynb from source
├── README.md
└── qlora-banking77-adapter/               # saved after training
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

---

## Notes

- The LoRA adapter (`qlora-banking77-adapter/`) is only ~20–30 MB on disk.
  The base model stays frozen and can be shared across multiple adapters.
- Increasing rank to `r=32` or training for 5+ epochs can push accuracy higher,
  at the cost of a slightly larger adapter and potential overfitting.
- For production inference, merge the adapter into the base model with
  `model.merge_and_unload()` for faster decoding.

# Project 2: LoRA vs QLoRA Model Tuning Lab

Notebook-based, end-to-end fine-tuning project comparing:

- LoRA on `distilgpt2`
- QLoRA (4-bit) on `facebook/opt-350m`

Task: emotion classification via generative label prediction on `dair-ai/emotion`.

## Highlights

- Real adapter tuning runs with two PEFT methods and two different base models.
- Baseline vs tuned evaluation under the same protocol.
- Production-style structure: `src/`, CLI, tests, notebooks, artifacts, app.

## Stack

- Python `3.12.10`
- Env/package manager: `uv`
- Core: `transformers`, `peft`, `bitsandbytes`, `datasets`, `accelerate`, `torch`
- Reporting: `pandas`, `polars`, `plotly`, `jinja2`
- App: `streamlit`

## Setup

```bash
git clone https://github.com/pypi-ahmad/lora-qlora-finetuning-lab.git
cd lora-qlora-finetuning-lab
uv python pin 3.12.10
uv sync --dev
cp .env.example .env
```

## Run end-to-end

```bash
uv run lora-qlora-lab run-all
```

## Run notebooks (tutorial flow)

```bash
uv run python scripts/execute_notebooks.py
```

Notebook order:

1. `notebooks/01_data_and_baselines.ipynb`
2. `notebooks/02_lora_finetuning_tutorial.ipynb`
3. `notebooks/03_qlora_finetuning_tutorial.ipynb`
4. `notebooks/04_comparison_and_report.ipynb`

## Real results (executed on June 12, 2026)

From `artifacts/metrics/summary.json`:

- `tuned_lora_accuracy`: `0.3125`
- `tuned_qlora_accuracy`: `0.3625`
- `lora_accuracy_gain`: `+0.3125`
- `qlora_accuracy_gain`: `+0.2625`
- `lora_f1_gain`: `+0.0984`
- `qlora_f1_gain`: `+0.0040`
- `lora_train_runtime_seconds`: `6.9273`
- `qlora_train_runtime_seconds`: `51.3507`

## App

```bash
uv run lora-qlora-lab serve-app --port 8502
```

## Tests

```bash
uv run pytest
```

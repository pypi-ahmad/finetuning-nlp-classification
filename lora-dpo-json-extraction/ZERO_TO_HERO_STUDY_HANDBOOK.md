# Zero to Hero Study Handbook: lora-dpo-json-extraction

## Module 1: Foundations & Architecture

### 1) What this project does

This repository builds an end-to-end fine-tuning pipeline for **structured JSON extraction** from support tickets using:

- Baseline generation (no adapter)
- Supervised fine-tuning (SFT) with LoRA/QLoRA adapters
- Preference optimization with DPO
- Stage-by-stage evaluation with persisted prediction artifacts and summary metrics

Core runtime entrypoint is [`src/lora_dpo_json_extraction/run_pipeline.py`](src/lora_dpo_json_extraction/run_pipeline.py), usually called with:

```bash
uv run python -m lora_dpo_json_extraction.run_pipeline --config configs/default.yaml
```

### 2) Main use cases

- Learn a full adapter-tuning workflow (data -> SFT -> DPO -> evaluation).
- Compare baseline vs SFT vs DPO on JSON extraction metrics.
- Study deterministic synthetic data generation and optional internet dataset ingestion (`PolyAI/banking77`).
- Reuse the pipeline skeleton for other extraction schemas/tasks.

### 3) Core paradigms and patterns used in this repo

- **Pipeline orchestration pattern**: `run()` in `run_pipeline.py` coordinates fixed stages and writes all artifacts in one run directory.
- **Config-first design**: `ProjectConfig` (Pydantic) in `settings.py` validates YAML config into typed objects.
- **Hybrid style (functional + lightweight OOP)**:
  - Functional helpers in `data.py`, `evaluate.py`, `utils.py`.
  - OOP-like dataset/collator classes in `datasets.py`.
- **Adapter-based fine-tuning**: `models.py` builds PEFT LoRA adapters on top of a base causal LM.
- **Deterministic data pipeline**: `set_seed()` plus deterministic `random.Random(seed)` generation/sampling.
- **Evaluation with explicit audit artifacts**: `evaluate_stage()` computes rates and writes `predictions_{stage}.jsonl`.

### 4) Architecture: components and interactions

Key modules:

- `settings.py`: typed config models and YAML loading.
- `run_pipeline.py`: orchestration, run directory creation, final reporting.
- `data.py`: synthetic/Banking77 split generation + DPO preference pair construction.
- `datasets.py`: PyTorch dataset/collator wrappers for SFT and DPO.
- `models.py`: tokenizer/base model loading, LoRA/QLoRA setup, adapter loading for inference.
- `train.py`: SFT and DPO training loops.
- `evaluate.py`: generation-based evaluation and metric computation.
- `utils.py`: seed, file I/O, JSON extraction helper.

Main flow diagram:

```text
CLI (--config)
   |
   v
load_config() -> ProjectConfig
   |
   v
run(cfg)
   |
   +--> set_seed(cfg.seed)
   +--> create outputs/run_<timestamp>/
   +--> build_splits(...) --------------+
   |                                    |
   +--> build_preference_split(train)   |
   |                                    |
   +--> evaluate_stage(stage="base")    |
   +--> train_sft(...) -> sft_adapter/  |
   +--> evaluate_stage(stage="sft", adapter_dir=sft_adapter)
   +--> train_dpo(..., sft_adapter_dir=sft_adapter) -> dpo_adapter/
   +--> evaluate_stage(stage="dpo", adapter_dir=dpo_adapter)
   |
   +--> write metrics_summary.json
   +--> write RUN_REPORT.md
   +--> write predictions_*.jsonl and data/*.jsonl
```

---

## Module 2: Repository Map

| File/Directory Path | Primary Responsibility | Key Classes/Functions | Important Configs/Variables |
|---|---|---|---|
| `pyproject.toml` | Packaging/dependencies/tool config | N/A | `project.dependencies`, `optional-dependencies.cpu/gpu`, `requires-python` |
| `README.md` | User-facing tutorial and run commands | N/A | Quickstart commands, config file paths |
| `configs/default.yaml` | Default run configuration (synthetic by default) | N/A | `seed`, `output_root`, `model.*`, `data.train_size/val_size/test_size`, `train.*`, `eval.max_new_tokens` |
| `configs/internet_banking77.yaml` | Internet dataset configuration | N/A | `data.source=banking77`, `data.hf_dataset`, split names, training hyperparameters |
| `src/lora_dpo_json_extraction/run_pipeline.py` | End-to-end stage orchestration and reporting | `run`, `parse_args`, `main`, `_save_data_artifacts`, `_save_markdown_report` | `--config`, summary `deltas` keys |
| `src/lora_dpo_json_extraction/settings.py` | Typed config schema + YAML loader | `ModelConfig`, `DataConfig`, `TrainConfig`, `EvalConfig`, `ProjectConfig`, `load_config` | Defaults like `DataConfig.source="synthetic"` |
| `src/lora_dpo_json_extraction/data.py` | Synthetic/Banking77 dataset building and DPO pair generation | `build_splits`, `build_banking77_splits`, `build_preference_split`, `generate_examples`, `build_prompt` | `PROMPT_TEMPLATE`, intent/product/priority phrase maps, Banking77 mapping rules |
| `src/lora_dpo_json_extraction/datasets.py` | Torch datasets and collators for training | `SFTDataset`, `DPODataset`, `SFTCollator`, `DPOCollator` | Response-token loss masking, pad token handling |
| `src/lora_dpo_json_extraction/models.py` | Tokenizer/model loading + LoRA/QLoRA adapter setup | `load_tokenizer`, `load_base_model`, `build_lora_model`, `load_adapter_for_inference`, `device_of` | `HF_HUB_DISABLE_XET`, QLoRA branch via `BitsAndBytesConfig` |
| `src/lora_dpo_json_extraction/train.py` | SFT and DPO training loops | `train_sft`, `train_dpo`, `_sequence_log_probs`, `_evaluate_sft_loss` | `cfg.train.*` hyperparameters, gradient accumulation, scheduler warmup |
| `src/lora_dpo_json_extraction/evaluate.py` | Inference-time generation parsing and metrics | `evaluate_stage`, `_safe_parse_json`, `_generate_prediction`, `_coerce_prediction` | `EXPECTED_KEYS`, `cfg.eval.max_new_tokens` |
| `src/lora_dpo_json_extraction/schemas.py` | Target extraction schema (Pydantic) | `TicketExtraction` | Required keys: `intent`, `priority`, `product`, `needs_human` |
| `src/lora_dpo_json_extraction/utils.py` | Utility helpers for seed and file I/O | `set_seed`, `write_json`, `write_jsonl`, `read_jsonl`, `extract_first_json_object` | JSON object extraction by brace balancing |
| `tests/test_data_generation.py` | Static checks for split/prefs behavior | `test_splits_have_expected_sizes`, `test_preference_pairs_contain_chosen_and_rejected`, `test_unknown_data_source_raises` | Verifies expected split sizes and error handling |
| `tests/test_json_parser.py` | Static checks for JSON substring parser | `test_extract_first_json_object`, `test_extract_none_on_no_object` | Confirms parser behavior for valid/invalid text |

Files a new contributor should read first:

1. `README.md`
2. `configs/default.yaml`
3. `src/lora_dpo_json_extraction/run_pipeline.py`
4. `src/lora_dpo_json_extraction/data.py`
5. `src/lora_dpo_json_extraction/train.py`
6. `src/lora_dpo_json_extraction/evaluate.py`

---

## Module 3: Core Execution Flows

### Flow A: Entry point and orchestration (`run_pipeline.py`)

1. `main()` calls `parse_args()`, which defines `--config` defaulting to `configs/default.yaml`.
2. `main()` loads config via `load_config(args.config)` from `settings.py`.
3. `run(cfg)` executes the full pipeline:
   - `set_seed(cfg.seed)`
   - `_make_run_dir(cfg)` creates `outputs/run_<YYYYmmdd_HHMMSS>/`
   - `build_splits(...)` creates `train`, `val`, `test`
   - `build_preference_split(splits["train"], seed=cfg.seed)` creates DPO pairs
   - `_save_data_artifacts(...)` writes data JSONL files
   - `evaluate_stage(stage="base", ...)`
   - `train_sft(...)`
   - `evaluate_stage(stage="sft", adapter_dir=...)`
   - `train_dpo(..., sft_adapter_dir=...)`
   - `evaluate_stage(stage="dpo", adapter_dir=...)`
   - writes `metrics_summary.json` and `RUN_REPORT.md`

Short code fragment (real logic shape):

```python
base_metrics = evaluate_stage(cfg=cfg, stage="base", rows=splits["test"], output_dir=run_dir)
sft_summary = train_sft(cfg, splits["train"], splits["val"], run_dir)
sft_metrics = evaluate_stage(cfg=cfg, stage="sft", rows=splits["test"], output_dir=run_dir, adapter_dir=sft_summary["adapter_dir"])
```

### Flow B: Data construction (`data.py`)

#### B1) SFT/Eval split generation

`build_splits(...)` dispatches by `source`:

- `"synthetic"`: calls `generate_examples(total_size, seed)` and slices into train/val/test.
- `"banking77"` or aliases: calls `build_banking77_splits(...)` which uses `datasets.load_dataset(...)`.

Synthetic row shape:

```python
{
  "prompt": str,
  "response": str,   # JSON string
  "target": {
    "intent": str,
    "priority": str,      # "low" | "medium" | "high"
    "product": str,
    "needs_human": bool
  }
}
```

`build_prompt(ticket)` injects ticket text into `PROMPT_TEMPLATE` and ends with `JSON:` so the model is guided to produce JSON only.

#### B2) Banking77 mapping

In `_banking77_row_to_payload(text, label_name)`:

1. Normalize label with `_normalize_intent`.
2. Map to grouped `intent` via `_derive_intent_group_from_banking77_label`.
3. Derive `priority` via `_derive_priority_from_banking77`.
4. Derive `product` via `_derive_product_from_banking77_intent`.
5. Derive `needs_human` via `_derive_needs_human_from_banking77`.
6. Validate via `TicketExtraction.model_validate(...)`.

#### B3) DPO preference pair generation

`build_preference_split(train_rows, seed)` creates rows with:

```python
{
  "prompt": str,
  "chosen": str,      # row["response"] (correct JSON string)
  "rejected": str,    # intentionally weaker variant
  "target": {...}     # same target dict
}
```

`_build_rejected(...)` uses one of:

- `wrong_field`
- `missing_field`
- `extra_text` (prepends non-JSON text before JSON)

### Flow C: SFT training (`train.py` + `datasets.py` + `models.py`)

1. `train_sft(...)` loads tokenizer and LoRA model:
   - `tokenizer = load_tokenizer(cfg.model.name)`
   - `model = build_lora_model(cfg.model, for_training=True)`
2. Creates `SFTDataset` for train/val and `DataLoader` with `SFTCollator`.
3. `SFTDataset.__getitem__` encodes `prompt` + `response + eos`, and applies **response-only loss**:
   - prompt labels are `-100` (ignored by CE loss)
   - response token labels are actual token ids
4. Training loop does:
   - forward pass
   - loss scaling by `gradient_accumulation_steps`
   - `loss.backward()`
   - grad clipping, optimizer/scheduler steps
5. Validation loss via `_evaluate_sft_loss`.
6. Saves adapter to `run_dir / "sft_adapter"` and tokenizer to `run_dir / "tokenizer"`.

SFT summary output shape:

```python
{
  "adapter_dir": str,
  "tokenizer_dir": str,
  "history": [{"epoch": float, "train_loss": float, "val_loss": float}, ...],
  "global_steps": int
}
```

### Flow D: DPO training (`train.py`)

1. `train_dpo(...)` loads:
   - policy model from base + SFT adapter (`is_trainable=True`)
   - reference model from base + SFT adapter (`is_trainable=False`, frozen params)
2. Builds `DPODataset` + `DPOCollator`.
3. Per batch:
   - compute chosen/rejected sequence log-probs for policy and reference via `_sequence_log_probs(...)`
   - compute advantage:
     - `(chosen_pi - rejected_pi) - (chosen_ref - rejected_ref)`
   - DPO loss:
     - `-logsigmoid(cfg.train.dpo_beta * advantage).mean()`
4. Runs optimizer/scheduler steps and records epoch loss.
5. Saves adapter to `run_dir / "dpo_adapter"`.

DPO summary output shape:

```python
{
  "adapter_dir": str,
  "history": [{"epoch": float, "dpo_loss": float}, ...],
  "global_steps": int
}
```

### Flow E: Evaluation and metrics (`evaluate.py`)

`evaluate_stage(cfg, stage, rows, output_dir, adapter_dir=None)`:

1. Loads tokenizer (`load_tokenizer`).
2. Builds model:
   - `stage="base"` -> `load_base_model(..., for_training=False)`
   - `stage in {"sft","dpo"}` -> `load_adapter_for_inference(...)`
3. For each row:
   - generate output with `_generate_prediction(...)` (`do_sample=False`, `temperature=0.0`)
   - parse first balanced JSON object using `extract_first_json_object(...)` then `json.loads(...)`
   - normalize with `_coerce_prediction(...)`
4. Computes metrics and writes `predictions_{stage}.jsonl`.

Per-prediction JSONL row shape:

```python
{
  "prompt": str,
  "target": dict,
  "prediction_text": str,
  "prediction_json": dict | None,
  "valid_json": bool,
  "exact_match": bool,
  "latency_ms": float
}
```

Stage metrics dict shape:

```python
{
  "stage": str,
  "num_examples": int,
  "valid_json_rate": float,
  "schema_match_rate": float,
  "exact_match_rate": float,
  "field_accuracy": float,
  "avg_latency_ms": float
}
```

---

## Module 4: Setup & Run Guide

### 1) Tech stack inferred from repo files

- Language: Python 3.12 (`pyproject.toml` says `>=3.12,<3.13`)
- ML stack: PyTorch + Transformers + PEFT + Datasets
- Config: YAML + Pydantic models
- Package/tooling: `uv` + `setuptools`
- Logging: `loguru`

### 2) Clean-machine setup (static command sequence)

```bash
git clone https://github.com/pypi-ahmad/lora-dpo-json-extraction.git
cd lora-dpo-json-extraction
uv sync --extra gpu
```

CPU-only installation can use:

```bash
uv sync --extra cpu
```

### 3) Environment configuration

No `.env` file or required secret keys are defined in the source code.

Environment variables used or documented:

- `HF_HUB_DISABLE_XET`
  - Used in `models.py` via `os.environ.setdefault("HF_HUB_DISABLE_XET", "1")`.
  - README also shows setting it explicitly in run commands.
- `CUDA_VISIBLE_DEVICES`
  - Used in README command example to force CPU path (`CUDA_VISIBLE_DEVICES=''`).

### 4) Configuration files and key controls

- `configs/default.yaml`
  - Uses local Qwen snapshot path in `model.name`
  - `model.use_qlora: true`
  - No explicit `data.source`, so `DataConfig` default applies: `"synthetic"`
- `configs/internet_banking77.yaml`
  - `data.source: banking77`
  - `data.hf_dataset: PolyAI/banking77`
  - Uses `distilgpt2` and LoRA-only (`use_qlora: false`)

### 5) Typical run commands

Synthetic/default path:

```bash
HF_HUB_DISABLE_XET=1 uv run python -m lora_dpo_json_extraction.run_pipeline --config configs/default.yaml
```

Banking77 path:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_DISABLE_XET=1 uv run python -m lora_dpo_json_extraction.run_pipeline --config configs/internet_banking77.yaml
```

### 6) Runtime artifacts written by pipeline

Under `outputs/run_<timestamp>/`:

- `config_snapshot.json`
- `run.log`
- `metrics_summary.json`
- `RUN_REPORT.md`
- `predictions_base.jsonl`
- `predictions_sft.jsonl`
- `predictions_dpo.jsonl`
- `sft_training_summary.json`
- `dpo_training_summary.json`
- `sft_adapter/`
- `dpo_adapter/`
- `tokenizer/`
- `data/train_sft.jsonl`
- `data/val_sft.jsonl`
- `data/test_eval.jsonl`
- `data/train_dpo_prefs.jsonl`

### 7) Database migrations or seeding

- No database layer exists in this repo.
- No migration or DB seeding scripts exist.
- Data “seeding” is handled by deterministic split generation via `seed` in config and `set_seed()`.

### 8) Export this handbook to PDF (optional)

```bash
pandoc ZERO_TO_HERO_STUDY_HANDBOOK.md -o ZERO_TO_HERO_STUDY_HANDBOOK.pdf
```

---

## Module 5: Study Plan & Practice Exercises

### 1) Ordered study plan for a new learner

1. Read `README.md` for goals, stage order, and expected artifacts.
2. Read `pyproject.toml` for dependencies and Python/tooling constraints.
3. Read `settings.py` and both YAML files to understand config contract and defaults.
4. Read `schemas.py` and `data.py` to learn target schema and how datasets are built.
5. Read `datasets.py` to understand tokenization, truncation, masking, and padding.
6. Read `models.py` and `train.py` to understand LoRA/QLoRA and DPO loops.
7. Read `evaluate.py` to learn parsing and metric calculations.
8. Read `run_pipeline.py` last to connect all pieces end-to-end.
9. Read `tests/` to confirm intended behavior for split generation and JSON extraction utility.

### 2) Practice exercises (with pointers)

1. **Trace the default data source**
   - Question: If `configs/default.yaml` omits `data.source`, which source is used and where is that default defined?
   - Files: `configs/default.yaml`, `src/lora_dpo_json_extraction/settings.py`, `src/lora_dpo_json_extraction/data.py`

2. **Draw the exact stage order**
   - Question: In what exact order are baseline eval, SFT, DPO, and evaluations executed in `run()`?
   - Files: `src/lora_dpo_json_extraction/run_pipeline.py`

3. **Explain response-only SFT loss**
   - Question: How does `SFTDataset.__getitem__` ensure prompt tokens are ignored in loss?
   - Files: `src/lora_dpo_json_extraction/datasets.py`

4. **Explain how rejected DPO responses are built**
   - Question: What three corruption modes can `_build_rejected()` create?
   - Files: `src/lora_dpo_json_extraction/data.py`

5. **Banking77 mapping reasoning**
   - Question: How are `intent`, `priority`, `product`, and `needs_human` derived from raw Banking77 labels and text?
   - Files: `src/lora_dpo_json_extraction/data.py`

6. **Metric math check**
   - Question: How are `valid_json_rate`, `schema_match_rate`, and `field_accuracy` computed?
   - Files: `src/lora_dpo_json_extraction/evaluate.py`

7. **Adapter loading path**
   - Question: How does evaluation decide between base model and adapter-based model loading?
   - Files: `src/lora_dpo_json_extraction/evaluate.py`, `src/lora_dpo_json_extraction/models.py`

8. **Artifact contract**
   - Question: List all files/subfolders that `run_pipeline.py` guarantees to write when stages complete.
   - Files: `src/lora_dpo_json_extraction/run_pipeline.py`, `src/lora_dpo_json_extraction/train.py`, `src/lora_dpo_json_extraction/evaluate.py`

### 3) Model answer outlines

1. **Default data source answer outline**
   - `DataConfig.source` default is `"synthetic"` in `settings.py`.
   - `configs/default.yaml` does not override it.
   - `build_splits()` receives this value and follows synthetic path.

2. **Stage order answer outline**
   - `evaluate_stage("base")` -> `train_sft()` -> `evaluate_stage("sft")` -> `train_dpo()` -> `evaluate_stage("dpo")`.

3. **Response-only loss answer outline**
   - `labels = [-100] * prompt_len + input_ids[prompt_len:]`.
   - `-100` tells CE loss to ignore prompt tokens.

4. **Rejected response modes answer outline**
   - Wrong field value mutation.
   - Missing a required field.
   - Prefixing extra non-JSON text before JSON payload.

5. **Banking77 mapping answer outline**
   - Normalize label text.
   - Map to grouped intent using keyword rules.
   - Derive priority from intent group and urgency keywords.
   - Derive product from keyword mapping.
   - Derive needs_human from group/keywords/priority.
   - Validate final dict against `TicketExtraction`.

6. **Metric math answer outline**
   - `valid_json_rate = valid_json / len(rows)`
   - `schema_match_rate = schema_match / len(rows)` where schema match means exact expected key set.
   - `field_accuracy = field_hits / (len(rows) * len(EXPECTED_KEYS))`

7. **Adapter loading answer outline**
   - In `_build_model`, stage `"base"` calls `load_base_model(...)`.
   - Other stages require `adapter_dir` and call `load_adapter_for_inference(...)`.

8. **Artifact contract answer outline**
   - Core outputs from orchestrator: config snapshot, logs, summary/report, predictions.
   - Training adds `sft_adapter/`, `dpo_adapter/`, tokenizer and training summary JSON files.
   - Data artifacts saved under `data/` JSONL files.

---

## Understanding Checklist

Use this checklist before claiming mastery:

- Can you explain what `run()` in `run_pipeline.py` does without looking at the file?
- Can you describe how `build_splits()` changes behavior for `synthetic` vs `banking77`?
- Can you explain why `TicketExtraction` validation is used in data building?
- Can you describe exactly how SFT loss masking is implemented?
- Can you explain the difference between policy and reference models in `train_dpo()`?
- Can you derive DPO loss terms from `_sequence_log_probs()` outputs?
- Can you explain how prediction text is converted into `prediction_json` safely?
- Can you compute each evaluation metric manually for a tiny 2-row example?
- Can you list which output files are used for debugging bad model behavior?
- Can you state which environment variables are optional and where they are used?


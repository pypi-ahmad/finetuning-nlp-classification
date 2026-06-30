# 📘 MASTER LEARNING HANDBOOK: QLoRA Complaint Intent Classifier (`qlora-complaint-intent-classifier`)

## Handbook Integrity Contract

- Analysis mode: **static only** (no execute/compile/test performed for this handbook).
- Evidence rule: every concrete claim below is grounded in files inside this repository.
- Repository root used: `/home/ahmad/AI/Github/finetuning-nlp-classification/qlora-complaint-intent-classifier`.
- Visible scope covered:
  - **13 directories**
  - **67 files**

---

## 🌐 Module 1: Theoretical Foundations & Architecture

### 1.1 CS Theory & Definitions (and how this repo uses each)

#### 1) Decoder-only Transformer (Causal LM)

**Definition:**
A decoder-only Transformer predicts the next token autoregressively. Given tokens `x_1..x_t`, it models `P(x_{t+1} | x_1..x_t)`.

**Where implemented in this repo:**
- Base model ID is hard-coded as `Qwen/Qwen2.5-1.5B-Instruct` in:
  - `inference.py` (`BASE_MODEL_ID`)
  - `make_notebook.py` (`MODEL_ID`)
- Task type is explicitly configured as `TaskType.CAUSAL_LM` in `make_notebook.py` LoRA config.

#### 2) Supervised Fine-Tuning (SFT)

**Definition:**
SFT trains a model on labeled `(input -> desired output)` examples using supervised loss on target tokens.

**Where implemented:**
- `make_notebook.py` builds supervised chat text with `build_chat(...)` and `format_for_sft(...)`, then uses:
  - `SFTTrainer`
  - `SFTConfig(dataset_text_field="text")`
- Dataset record becomes one text prompt that includes system, user, and assistant label.

#### 3) LoRA (Low-Rank Adaptation)

**Definition:**
Instead of updating full weight matrix `W`, LoRA learns low-rank matrices `A` and `B` such that:
`ΔW = (α / r) * B * A`
where `r` is rank and `α` is scaling.

**Where implemented:**
- `make_notebook.py` sets:
  - `r=16`
  - `lora_alpha=32`
  - `lora_dropout=0.05`
  - `bias="none"`
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
- Persisted in `qlora-banking77-adapter/adapter_config.json` with same values.

#### 4) QLoRA (Quantized LoRA)

**Definition:**
QLoRA keeps base model weights quantized (commonly 4-bit) and frozen, while training only LoRA adapters.

**Where implemented:**
- Both `inference.py` and notebook logic use `BitsAndBytesConfig` with:
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type="nf4"`
  - `bnb_4bit_compute_dtype=torch.bfloat16`
  - `bnb_4bit_use_double_quant=True`

#### 5) Greedy Decoding

**Definition:**
At each generation step, pick the highest-probability token (`argmax`) rather than sampling.

**Where implemented:**
- `model.generate(..., do_sample=False, max_new_tokens=16)` in both notebook `predict_intent(...)` and `inference.py` `predict(...)`.

#### 6) Determinism / Reproducibility Basics

**Definition:**
Set random seeds so pseudo-random procedures are repeatable.

**Where implemented:**
- `make_notebook.py` defines `SEED = 42`, sets:
  - `random.seed(SEED)`
  - `np.random.seed(SEED)`
  - `torch.manual_seed(SEED)`
  - `torch.cuda.manual_seed_all(SEED)` when CUDA is available.

#### 7) Multi-class Classification Metrics

**Accuracy Definition:** fraction of correct predictions among all examples.

**Macro F1 Definition:** average F1 across classes with equal class weight.

**Where implemented:**
- `accuracy_score(...)` and `f1_score(..., average="macro", labels=LABEL_NAMES, zero_division=0)` in notebook evaluation cells.

#### 8) Confusion Matrix (Row-normalized)

**Definition:**
Matrix where row = true class, column = predicted class. Row normalization gives per-class error distribution.

**Where implemented:**
- `confusion_matrix(..., normalize="true")` in notebook confusion analysis.

---

### 1.2 System Topology (ASCII Flow)

```text
+------------------------------+
| pyproject.toml + uv.lock     |
| (env + dependency contract)  |
+--------------+---------------+
               |
               v
+------------------------------+
| make_notebook.py             |
| - builds 37-cell .ipynb JSON |
| - embeds full training logic |
+--------------+---------------+
               |
               v
+-----------------------------------------------+
| qlora_complaint_intent_classifier.ipynb       |
| Core runtime pipeline encoded in cells:       |
| 1) Load Banking77                             |
| 2) Build chat-formatted SFT text              |
| 3) Baseline zero-shot eval (200 subset)       |
| 4) QLoRA SFT training (3 epochs)              |
| 5) Save adapter + tokenizer                   |
| 6) Reload adapter and eval full test (3080)   |
| 7) Export plots and diagnostics               |
+--------------+--------------------------------+
               |
               v
+-----------------------------------------------+
| qlora-banking77-adapter/                      |
| - adapter_config.json                          |
| - adapter_model.safetensors                    |
| - tokenizer(.json/.config)                     |
| - checkpoint-* training states                 |
+--------------+--------------------------------+
               |
               v
+-----------------------------------------------+
| inference.py                                   |
| - load_model(adapter_dir)                      |
| - predict(text, model, tokenizer)              |
| - CLI modes: single / batch / interactive      |
+-----------------------------------------------+
```

---

### 1.3 Technology Stack (Exact Role + Real Config)

| Component | Version / Value Found | Role in System | Exact Repo Grounding |
|---|---|---|---|
| Python | `3.12.10` | Runtime baseline | `.python-version`; notebook metadata `language_info.version` |
| UV | `0.11.19` (venv metadata) | Env + lock resolver | `.venv/pyvenv.cfg` |
| Torch backend | `cu128` | CUDA backend preference | `pyproject.toml [tool.uv] torch-backend="cu128"` |
| PyTorch (locked) | `2.12.0` | Tensor compute / model runtime | `uv.lock`; notebook stdout prints `2.12.0+cu130` |
| Transformers | `5.10.2` | Model/tokenizer APIs | `uv.lock`; notebook stdout |
| PEFT | `0.19.1` | LoRA adapter composition | `uv.lock`; adapter config `peft_version` |
| TRL | `1.5.1` | SFTTrainer orchestration | `uv.lock`; adapter README |
| BitsAndBytes | `0.49.2` | 4-bit quantization for QLoRA | `uv.lock`; notebook stdout |
| Datasets | `5.0.0` | Banking77 loading + mapping | `uv.lock`; notebook stdout |
| Accelerate | `1.13.0` | Trainer runtime support | `uv.lock`; notebook stdout |
| Evaluate | `0.4.6` | Evaluation toolkit dependency | `uv.lock` |
| scikit-learn | `1.9.0` | accuracy/F1/report/confusion matrix | `uv.lock`; `make_notebook.py` imports |
| numpy | `2.4.6` | arrays/stats support | `uv.lock` |
| pandas | `3.0.3` | tabular transforms/reporting | `uv.lock` |
| matplotlib | `3.10.9` | plot generation | `uv.lock`; output PNGs |
| seaborn | `0.13.2` | confusion heatmap + styling | `uv.lock` |
| sentencepiece | `0.2.1` | tokenizer dependency | `uv.lock` |
| Notebook format | `nbformat=4`, `minor=5` | serialized notebook schema | `qlora_complaint_intent_classifier.ipynb` |
| Base model ID | `Qwen/Qwen2.5-1.5B-Instruct` | Frozen foundation model | `inference.py`, `make_notebook.py`, adapter metadata |
| LoRA rank `r` | `16` | adapter dimensionality | `make_notebook.py`, `adapter_config.json` |
| LoRA alpha | `32` | scaling factor | same as above |
| LoRA dropout | `0.05` | regularization | same as above |
| LoRA task type | `CAUSAL_LM` | objective mode | same as above |
| Target modules | `q/k/v/o + gate/up/down proj` | adapted sublayers | same as above |
| Sequence length | `MAX_LENGTH=256` | truncation/training context cap | `make_notebook.py`; tokenizer truncation serialized |
| Batching | `per_device_train_batch_size=4`, `gradient_accumulation_steps=4` | effective batch `16` | `SFTConfig` in notebook logic |
| Epochs | `3` | training duration | `SFTConfig` |
| Optimizer | `paged_adamw_8bit` | memory-efficient optimizer | `SFTConfig` |
| LR scheduler | `cosine` | decay strategy | `SFTConfig` |
| Learning rate | `2e-4` | optimization step scale | `SFTConfig` |
| Warmup | `warmup_ratio=0.05` | scheduler ramp-up | `SFTConfig` |
| Precision | `bf16=True`, `fp16=False` | numeric mode | `SFTConfig` |
| Gradient checkpointing | `True` | memory/computation tradeoff | `SFTConfig`; model input grads enabled |
| Logging cadence | `logging_steps=50` | trainer telemetry granularity | `SFTConfig`; trainer state |
| Save strategy | `epoch` | checkpointing policy | `SFTConfig` |
| Report target | `none` | disables W&B/TensorBoard auto logging | `SFTConfig` |
| Inference decode cap | `MAX_NEW_TOKENS=16` | max emitted label tokens | `inference.py` |
| Inference decoding policy | `do_sample=False` | greedy deterministic decode | `inference.py` and notebook `predict_intent` |

---

### 1.4 Grounded Architecture Notes (Critical)

1. **Single-source training logic is notebook-driven.**
`make_notebook.py` is the code source that writes `qlora_complaint_intent_classifier.ipynb`. The notebook file contains stored outputs and metrics.

2. **Adapter-first deployment pattern.**
The system persists LoRA weights + tokenizer in `qlora-banking77-adapter/` and loads them in `inference.py` via `PeftModel.from_pretrained(base, adapter_dir)`.

3. **Structured fallback parsing for label robustness.**
`inference.py` normalizes generation text (`_normalise`) and applies:
- exact match
- prefix match
- character-overlap fallback over `LABEL_NAMES`.

4. **Potential consistency pitfall already visible in files.**
`README.md` dataset section claims train size `13,083`, while notebook output and code path report `10,003` train rows from the loaded dataset split.

5. **Another data-contract caveat visible in constants.**
`LABEL_NAMES` in `inference.py` includes `"reverted_card_payment?"` (question mark retained). This exact string also appears in notebook outputs, so training/eval are internally consistent, but external consumers should preserve this literal class label.

---

## 📂 Module 2: Strict Repository Tour & Mapping

### 2.1 Directory Map (All Visible Directories)

| File/Directory Path | Primary Responsibility | Key Classes/Functions Exported | Actual Configurations/Variables Defined |
|---|---|---|---|
| `.` | Repository root | N/A | Contains 67 files and 12 subdirectories (visible scope). |
| `.agents/` | Hidden tooling placeholder | N/A | Empty in visible scope (directory exists, no files listed). |
| `.codex/` | Hidden tooling placeholder | N/A | Empty in visible scope (directory exists, no files listed). |
| `.git/` | Hidden VCS placeholder in this environment | N/A | Directory exists; no internal files visible here. |
| `.venv/` | Local virtual environment root | shell activators + runtime patch stubs | `pyvenv.cfg` defines `implementation=CPython`, `version_info=3.12.10`, `uv=0.11.19`, prompt name. |
| `.venv/bin/` | Cross-shell activation/deactivation scripts | shell functions/aliases in each script | Sets `VIRTUAL_ENV` path and `VIRTUAL_ENV_PROMPT=qlora-complaint-intent-classifier`. |
| `.venv/lib/` | Virtualenv library root | N/A | Contains `python3.12` subtree. |
| `.venv/lib/python3.12/` | Python ABI subtree | N/A | Contains `site-packages` patch files. |
| `.venv/lib/python3.12/site-packages/` | Runtime patch injection for virtualenv | `_virtualenv.py` patch hooks | `_virtualenv.pth` imports `_virtualenv`; `_virtualenv.py` patches distutils/setuptools install path behavior. |
| `qlora-banking77-adapter/` | Final adapter artifact bundle | N/A (artifact directory) | adapter config, safetensors weights, tokenizer assets, training args, and checkpoints. |
| `qlora-banking77-adapter/checkpoint-626/` | Epoch-1-like checkpoint snapshot | N/A | `trainer_state.json` global_step `626`, epoch `1.0`. |
| `qlora-banking77-adapter/checkpoint-1252/` | Epoch-2-like checkpoint snapshot | N/A | `trainer_state.json` global_step `1252`, epoch `2.0`. |
| `qlora-banking77-adapter/checkpoint-1878/` | Final epoch-3 checkpoint snapshot | N/A | `trainer_state.json` global_step `1878`, epoch `3.0`. |

---

### 2.2 File Map (All Visible Files)

| File/Directory Path | Primary Responsibility | Key Classes/Functions Exported | Actual Configurations/Variables Defined |
|---|---|---|---|
| `.gitignore` | Ignore policy for generated/runtime files | N/A | Ignores `__pycache__/`, `*.py[oc]`, `build/`, `dist/`, `wheels/`, `*.egg-info`, `.venv`, `.ipynb_checkpoints/`. |
| `.python-version` | Python version pin | N/A | Literal value: `3.12.10`. |
| `.venv/.gitignore` | Venv internal ignore marker | N/A | Single character `*`. |
| `.venv/.lock` | Venv lock marker | N/A | Empty file (0 bytes). |
| `.venv/CACHEDIR.TAG` | Cache-dir signature marker | N/A | `Signature: 8a477f597d28d172789f06886806bc55`. |
| `.venv/pyvenv.cfg` | Virtualenv metadata | N/A | `home=...cpython-3.12.10...`, `implementation=CPython`, `uv=0.11.19`, `include-system-site-packages=false`, prompt name. |
| `.venv/bin/activate` | Bash/Zsh/Ksh activator | `deactivate()`, `pydoc()` shell funcs | Sets `VIRTUAL_ENV` absolute path, prepends `PATH`, unsets `PYTHONHOME`, sets `VIRTUAL_ENV_PROMPT`. |
| `.venv/bin/activate.bat` | Windows CMD activator | batch labels | Sets `%VIRTUAL_ENV%`, `%VIRTUAL_ENV_PROMPT%`, rewrites `%PATH%`, clears `%PYTHONHOME%`. |
| `.venv/bin/activate.csh` | C-shell activator | `deactivate` alias, `pydoc` alias | Sets env vars and prompt wrapping with venv prompt text. |
| `.venv/bin/activate.fish` | Fish activator | fish funcs `deactivate`, `pydoc`, prompt override | Sets `VIRTUAL_ENV`, modifies `PATH`, optionally patches `fish_prompt`. |
| `.venv/bin/activate.nu` | Nushell activator | `deactivate` alias, `pydoc` alias | Requires Nushell `>=0.106`; exports `VIRTUAL_ENV`, prepends path, sets prompt prefix. |
| `.venv/bin/activate.ps1` | PowerShell activator | `global:deactivate`, `global:pydoc`, prompt function | Sets `$env:VIRTUAL_ENV`, `$env:VIRTUAL_ENV_PROMPT`, rewrites `$env:PATH`. |
| `.venv/bin/activate_this.py` | Python-side venv activation helper | top-level script logic | Sets env `PATH`, `VIRTUAL_ENV`, `VIRTUAL_ENV_PROMPT`; injects `../lib/python3.12/site-packages` into `sys.path`. |
| `.venv/bin/deactivate.bat` | Windows CMD deactivator | batch labels | Clears `VIRTUAL_ENV`, `VIRTUAL_ENV_PROMPT`, restores old `PROMPT`, `PYTHONHOME`, and `PATH` from `_OLD_*`. |
| `.venv/bin/pydoc.bat` | Windows pydoc shim | N/A | Runs `python.exe -m pydoc %*`. |
| `.venv/lib/python3.12/site-packages/_virtualenv.pth` | Virtualenv startup hook | N/A | Line content: `import _virtualenv`. |
| `.venv/lib/python3.12/site-packages/_virtualenv.py` | Distutils/setuptools path safety patch | `patch_dist`, class `_Finder` | Patches install config parsing to keep installs inside venv and hooks `sys.meta_path`. |
| `README.md` | Human-facing project guide | N/A | Defines task/dataset/model/training summary; setup commands; notebook and inference usage; includes several hard-coded numerical claims. |
| `pyproject.toml` | Project metadata + dependency spec | N/A | `name=qlora-complaint-intent-classifier`, `requires-python>=3.12.10`, dependency list, `torch-backend=cu128`, `package=false`. |
| `uv.lock` | Fully resolved dependency lock | N/A | Locks concrete versions (e.g., torch 2.12.0, transformers 5.10.2, peft 0.19.1, trl 1.5.1, bitsandbytes 0.49.2, datasets 5.0.0). |
| `make_notebook.py` | Programmatically generates notebook JSON | `_uid`, `md`, `code` | Defines 37-cell notebook composition, training/eval code text, and writes `qlora_complaint_intent_classifier.ipynb`. |
| `qlora_complaint_intent_classifier.ipynb` | Executable training/eval notebook artifact | N/A (notebook cell functions inside) | `nbformat=4`, 37 cells (22 code, 15 markdown), stored outputs including metrics and logs. |
| `inference.py` | Runtime inference CLI using saved adapter | `load_model`, `_normalise`, `predict`, `main` | `BASE_MODEL_ID`, `ADAPTER_DIR`, `MAX_NEW_TOKENS`, `DEVICE`, `SYSTEM_MSG`, 77-item `LABEL_NAMES`, argparse interface. |
| `MASTER_LEARNING_HANDBOOK.md` | Comprehensive static learning handbook for this repository | N/A | 5-module architecture/tutorial document generated from repository evidence; includes full directory/file mapping. |
| `label_distribution.png` | Plot artifact | N/A | PNG, 2982x877. |
| `query_length.png` | Plot artifact | N/A | PNG, 1800x600. |
| `base_vs_finetuned.png` | Comparison chart artifact | N/A | PNG, 1484x594. |
| `confusion_matrix.png` | Confusion heatmap artifact | N/A | PNG, 1959x1784. |
| `qlora-banking77-adapter/README.md` | Adapter model card (concise generated card) | N/A | Front-matter includes base model and framework versions (`PEFT 0.19.1`, `TRL 1.5.1`, `Transformers 5.10.2`, `Pytorch 2.12.0`, `Datasets 5.0.0`, `Tokenizers 0.22.2`). |
| `qlora-banking77-adapter/adapter_config.json` | Canonical LoRA adapter config | N/A | `peft_type=LORA`, `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `task_type=CAUSAL_LM`, target modules list, `base_model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct`. |
| `qlora-banking77-adapter/adapter_model.safetensors` | Final LoRA weights tensor file | N/A | Size `36,981,856` bytes; metadata includes BF16 tensors; string scan shows 28 layers and LoRA A/B tensors for target modules. |
| `qlora-banking77-adapter/chat_template.jinja` | Tokenizer chat serialization template | Jinja template logic | Emits `<|im_start|>`/`<|im_end|>` framing; supports tool-call and tool-response branches; handles `add_generation_prompt`. |
| `qlora-banking77-adapter/tokenizer.json` | Tokenizer model data | N/A | Version `1.0`, BPE model, `151,643` vocab entries, `151,387` merges, truncation max length `256`, 22 added special tokens. |
| `qlora-banking77-adapter/tokenizer_config.json` | Tokenizer runtime config | N/A | `tokenizer_class=Qwen2Tokenizer`, `model_max_length=131072`, `eos_token=<|im_end|>`, `pad_token=<|endoftext|>`, special token list. |
| `qlora-banking77-adapter/training_args.bin` | Serialized trainer args bundle | N/A | Zip archive with `training_args/data.pkl` + metadata members; strings show SFTConfig fields (batch size, epochs, optimizer, scheduler, etc.). |
| `qlora-banking77-adapter/checkpoint-626/README.md` | Checkpoint model card template | N/A | Mostly generic “More Information Needed” template; framework versions appended. |
| `qlora-banking77-adapter/checkpoint-626/adapter_config.json` | Checkpoint adapter config | N/A | Same schema/values as final `adapter_config.json` (identical hash). |
| `qlora-banking77-adapter/checkpoint-626/adapter_model.safetensors` | Epoch checkpoint LoRA weights | N/A | Size `36,981,856` bytes; hash differs from final checkpoint and later snapshots. |
| `qlora-banking77-adapter/checkpoint-626/chat_template.jinja` | Checkpoint tokenizer template | Jinja template | Same content/hash as final template. |
| `qlora-banking77-adapter/checkpoint-626/optimizer.pt` | Optimizer state snapshot | N/A | Zip archive; 1,576 members, total listed length `37,671,578`. |
| `qlora-banking77-adapter/checkpoint-626/rng_state.pth` | RNG state snapshot | N/A | Zip archive; 8 members, total listed length `13,109`. |
| `qlora-banking77-adapter/checkpoint-626/scheduler.pt` | Scheduler state snapshot | N/A | Zip archive; 6 members, total listed length `257`. |
| `qlora-banking77-adapter/checkpoint-626/tokenizer.json` | Checkpoint tokenizer data | N/A | Same content/hash as final tokenizer JSON. |
| `qlora-banking77-adapter/checkpoint-626/tokenizer_config.json` | Checkpoint tokenizer config | N/A | Same content/hash as final tokenizer config. |
| `qlora-banking77-adapter/checkpoint-626/trainer_state.json` | Trainer telemetry at step 626 | N/A | `epoch=1.0`, `global_step=626`, `max_steps=1878`, first and last logged loss/metrics up to step 600. |
| `qlora-banking77-adapter/checkpoint-626/training_args.bin` | Serialized args at ckpt 626 | N/A | Same archive structure/hash as final `training_args.bin`. |
| `qlora-banking77-adapter/checkpoint-1252/README.md` | Checkpoint model card template | N/A | Same generic template as other checkpoint README files. |
| `qlora-banking77-adapter/checkpoint-1252/adapter_config.json` | Checkpoint adapter config | N/A | Same values/hash as final adapter config. |
| `qlora-banking77-adapter/checkpoint-1252/adapter_model.safetensors` | Mid-training LoRA weights | N/A | Size `36,981,856` bytes; distinct hash from 626 and final. |
| `qlora-banking77-adapter/checkpoint-1252/chat_template.jinja` | Checkpoint tokenizer template | Jinja template | Same content/hash as final template. |
| `qlora-banking77-adapter/checkpoint-1252/optimizer.pt` | Optimizer state snapshot | N/A | Zip archive; 1,576 members, total listed length `37,671,578`. |
| `qlora-banking77-adapter/checkpoint-1252/rng_state.pth` | RNG state snapshot | N/A | Zip archive; 8 members, total listed length `13,109`. |
| `qlora-banking77-adapter/checkpoint-1252/scheduler.pt` | Scheduler state snapshot | N/A | Zip archive; 6 members, total listed length `257`. |
| `qlora-banking77-adapter/checkpoint-1252/tokenizer.json` | Checkpoint tokenizer data | N/A | Same hash/content as final tokenizer JSON. |
| `qlora-banking77-adapter/checkpoint-1252/tokenizer_config.json` | Checkpoint tokenizer config | N/A | Same hash/content as final tokenizer config. |
| `qlora-banking77-adapter/checkpoint-1252/trainer_state.json` | Trainer telemetry at step 1252 | N/A | `epoch=2.0`, `global_step=1252`, `max_steps=1878`, logs through step 1250. |
| `qlora-banking77-adapter/checkpoint-1252/training_args.bin` | Serialized args at ckpt 1252 | N/A | Same archive structure/hash as final `training_args.bin`. |
| `qlora-banking77-adapter/checkpoint-1878/README.md` | Final checkpoint model card template | N/A | Same generic template structure as other checkpoint README files. |
| `qlora-banking77-adapter/checkpoint-1878/adapter_config.json` | Final checkpoint adapter config | N/A | Same values/hash as final adapter config. |
| `qlora-banking77-adapter/checkpoint-1878/adapter_model.safetensors` | Final checkpoint LoRA weights | N/A | Size `36,981,856` bytes; hash matches top-level final adapter model. |
| `qlora-banking77-adapter/checkpoint-1878/chat_template.jinja` | Final checkpoint tokenizer template | Jinja template | Same content/hash as final template. |
| `qlora-banking77-adapter/checkpoint-1878/optimizer.pt` | Final checkpoint optimizer state | N/A | Zip archive; 1,576 members, total listed length `37,671,578`. |
| `qlora-banking77-adapter/checkpoint-1878/rng_state.pth` | Final checkpoint RNG state | N/A | Zip archive; 8 members, total listed length `13,109`. |
| `qlora-banking77-adapter/checkpoint-1878/scheduler.pt` | Final checkpoint scheduler state | N/A | Zip archive; 6 members, total listed length `257`. |
| `qlora-banking77-adapter/checkpoint-1878/tokenizer.json` | Final checkpoint tokenizer data | N/A | Same hash/content as final tokenizer JSON. |
| `qlora-banking77-adapter/checkpoint-1878/tokenizer_config.json` | Final checkpoint tokenizer config | N/A | Same hash/content as final tokenizer config. |
| `qlora-banking77-adapter/checkpoint-1878/trainer_state.json` | Final training telemetry | N/A | `epoch=3.0`, `global_step=max_steps=1878`, `logging_steps=50`, `save_steps=500`, includes full log history. |
| `qlora-banking77-adapter/checkpoint-1878/training_args.bin` | Serialized args at final ckpt | N/A | Same archive structure/hash as top-level training args. |

---

## 🔍 Module 3: Line-by-Line Code & Output Breakdown

## 3.1 Flow A — Notebook Generator (`make_notebook.py`)

### A1) Notebook cell factory helpers

- `def _uid():` generates UUID4 IDs for each cell.
- `def md(src: str):` converts markdown string into notebook `source` list format.
- `def code(src: str):` wraps source into code cell schema with:
  - `execution_count=None`
  - `outputs=[]`

**Concrete output object shape produced by helpers:**

```python
{"cell_type": "markdown"|"code", "id": <uuid>, "metadata": {}, "source": [...]}
```

### A2) Cell assembly pattern

- `cells = []`
- multiple `cells.append(md(...))` and `cells.append(code(...))` calls encode the full training/eval pipeline as string literals.

### A3) Final notebook write

At bottom:

- builds `notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": ..., "cells": cells}`
- writes JSON to `out_path = "qlora_complaint_intent_classifier.ipynb"` using `json.dump(..., indent=1, ensure_ascii=False)`.

**Produced notebook metadata in this repo artifact:**
- `kernelspec.display_name = "Python 3 (ipykernel)"`
- `language_info.version = "3.12.10"`
- `total_cells = 37`
- `code_cells = 22`
- `markdown_cells = 15`

---

## 3.2 Flow B — Data Pipeline and Prompt Construction (Notebook Logic)

### B1) Dataset load and label extraction

Notebook logic text (embedded in `make_notebook.py`) loads:

```python
raw_ds = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")
train_ds = raw_ds["train"]
test_ds = raw_ds["test"]
LABEL_NAMES = [n.lower() for n in train_ds.features["label"].names]
```

**Actual stored notebook output:**
- `train num_rows: 10003`
- `test num_rows: 3080`
- `Intents: 77`

### B2) SFT chat format contract

System instruction constant:

```python
SYSTEM_MSG = (
  "You are a banking intent classifier. "
  "Given a customer query, respond with exactly one intent label using "
  "lowercase letters and underscores. Output only the label, nothing else."
)
```

`build_chat(text, intent=None)` constructs message list:

```python
[
  {"role": "system", "content": SYSTEM_MSG},
  {"role": "user", "content": text},
  # optionally assistant label during training
]
```

Then serializes using tokenizer chat template with:
- training: `add_generation_prompt=False`
- inference: `add_generation_prompt=True`

### B3) SFT dataset mapping

`format_for_sft(example)` returns:

```python
{"text": build_chat(example["text"], LABEL_NAMES[example["label"]])}
```

Mapped sets:
- `train_sft = raw_ds["train"].map(..., remove_columns=["text","label"])`
- `test_sft  = raw_ds["test"].map(..., remove_columns=["text","label"])`

**Actual sample serialized training record shown in notebook output:**
Includes `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`, and label `card_arrival`.

---

## 3.3 Flow C — Baseline Zero-Shot Evaluation

### C1) Base model quantized load

Baseline code builds `bnb_config` with 4-bit NF4 settings and loads:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

### C2) `predict_intent(...)` normalization and fallback

Exact logic sequence:
1. Build chat prompt (`build_chat(text)`)
2. Tokenize with truncation `max_length=MAX_LENGTH`
3. Generate with `max_new_tokens=16`, `do_sample=False`
4. Decode generated suffix only
5. Normalize output via regex:
   - replace non `[a-z0-9_]` with `_`
   - collapse repeated `_`
6. Fallback order:
   - exact in `LABEL_NAMES`
   - prefix/substring matching
   - char-overlap score against each label

### C3) Baseline sampling/eval

- `BASELINE_N = 200`
- `step = len(test_ds) // BASELINE_N`
- `eval_indices = list(range(0, len(test_ds), step))[:BASELINE_N]`

Metrics:

```python
base_acc = accuracy_score(base_true, base_preds)
base_f1 = f1_score(..., average="macro", zero_division=0, labels=LABEL_NAMES)
```

**Actual stored notebook output:**
- Baseline Accuracy: `4.50%`
- Baseline Macro F1: `4.18%`

---

## 3.4 Flow D — QLoRA Fine-Tuning

### D1) Model + LoRA injection

Training path sets:

```python
model.config.use_cache = False
model.enable_input_require_grads()
```

LoRA config:

```python
LoraConfig(
  r=16,
  lora_alpha=32,
  lora_dropout=0.05,
  bias="none",
  task_type=TaskType.CAUSAL_LM,
  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
```

### D2) SFTConfig exact training contract

```python
SFTConfig(
  output_dir="./qlora-banking77-adapter",
  dataset_text_field="text",
  max_length=256,
  packing=False,
  num_train_epochs=3,
  per_device_train_batch_size=4,
  gradient_accumulation_steps=4,
  gradient_checkpointing=True,
  optim="paged_adamw_8bit",
  learning_rate=2e-4,
  lr_scheduler_type="cosine",
  warmup_ratio=0.05,
  weight_decay=0.01,
  max_grad_norm=1.0,
  bf16=True,
  fp16=False,
  logging_steps=50,
  save_strategy="epoch",
  load_best_model_at_end=False,
  seed=SEED,
  report_to="none",
)
```

Derived values printed by notebook:
- effective batch size = `16`
- steps per epoch = `625`
- total steps (computed estimate) = `1875`

### D3) Training outputs persisted in notebook

Stored stdout block reports:
- `Global steps: 1878`
- `Training loss: 0.4705`
- `Runtime: 43.2 min`
- Adapter saved to `./qlora-banking77-adapter/`

### D4) Checkpoint telemetry (`trainer_state.json`)

- checkpoint-626: `epoch=1.0`, `global_step=626`
- checkpoint-1252: `epoch=2.0`, `global_step=1252`
- checkpoint-1878: `epoch=3.0`, `global_step=1878`, `max_steps=1878`

Each log entry shape (from `log_history`) includes keys:
`entropy`, `epoch`, `grad_norm`, `learning_rate`, `loss`, `mean_token_accuracy`, `num_tokens`, `step`.

---

## 3.5 Flow E — Fine-Tuned Evaluation + Diagnostics

### E1) Reload and evaluate full test split

- Base model reloaded with same quantization config
- Adapter loaded via `PeftModel.from_pretrained(_base, OUTPUT_DIR)`
- Inference loop runs across all `3,080` test rows

**Actual stored notebook output:**
- Fine-tuned Accuracy: `92.37%`
- Fine-tuned Macro F1: `91.97%`

### E2) Base vs Fine-tuned comparison table output

Stored table:

- `Base (zero-shot)` on `200` examples: `4.50%`, `4.18%`
- `QLoRA Fine-tuned (3 epochs)` on `3080` examples: `92.37%`, `91.97%`

Improvement lines:
- Accuracy improvement: `+87.87 pp`
- Macro F1 improvement: `+87.79 pp`

### E3) Per-class diagnostics

Stored outputs include:
- 10 hardest intents (lowest F1)
- 10 easiest intents (highest F1)
- confusion matrix statement for 20 most-confused intents
- qualitative sample with `Correct: 13/15`

---

## 3.6 Flow F — Inference Runtime (`inference.py`)

### F1) Constants and configuration

- `BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"`
- `ADAPTER_DIR = "./qlora-banking77-adapter"`
- `MAX_NEW_TOKENS = 16`
- `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`

### F2) `load_model(adapter_dir)`

Steps:
1. Build same 4-bit `BitsAndBytesConfig`.
2. Load base model with `device_map="auto"`, `torch_dtype=torch.bfloat16`, `trust_remote_code=True`.
3. Attach adapter using `PeftModel.from_pretrained(base, adapter_dir)`.
4. `model.eval()`.
5. Load tokenizer from adapter directory.
6. Ensure `tokenizer.pad_token` exists (fallback to `eos_token`).

### F3) `_normalise(text)`

Regex normalization:
- `re.sub(r"[^a-z0-9_]", "_", text.strip().lower())`
- `re.sub(r"_+", "_", n).strip("_")`

### F4) `predict(text, model, tokenizer)`

- Builds two-message chat (`system` + `user`)
- Applies chat template with `add_generation_prompt=True`
- Tokenizes with `truncation=True`, `max_length=256`
- Greedy generation (`do_sample=False`)
- Decodes generated suffix
- Applies same exact/prefix/char-overlap fallback over `LABEL_NAMES`

### F5) CLI modes (`main`)

Arguments:
- positional `query` (optional)
- `--batch FILE`
- `--adapter PATH`

Control flow:
- if adapter missing -> prints error and exits `sys.exit(1)`
- `--batch` mode prints lines as `"<label>\t<query>"`
- single query mode prints:
  - `Query  : ...`
  - `Intent : ...`
- no args -> interactive loop with prompt `Query> ` and output `Intent: ...`

---

## 3.7 Real Input/Output Structure Inventory

### Dataset row structure

From notebook dataset object and code:

```python
{"text": <str>, "label": <int>}
```

### SFT mapped row structure

```python
{"text": <chat-formatted string>}
```

### Message object structure used for prompting

```python
{"role": "system"|"user"|"assistant", "content": <str>}
```

### Adapter config JSON structure (core keys)

```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "task_type": "CAUSAL_LM",
  "target_modules": ["down_proj","o_proj","up_proj","k_proj","v_proj","q_proj","gate_proj"]
}
```

### Trainer log entry structure (`trainer_state.json`)

```json
{
  "entropy": <float>,
  "epoch": <float>,
  "grad_norm": <float>,
  "learning_rate": <float>,
  "loss": <float>,
  "mean_token_accuracy": <float>,
  "num_tokens": <float>,
  "step": <int>
}
```

---

## 🛠️ Module 4: Step-by-Step Setup & Development Guide (Conceptual, Static)

### 4.1 Clean-Machine Prerequisites (as specified in repo)

From `README.md` and project files:

1. Linux target: Ubuntu 22.04 / 24.04 / 26.04.
2. NVIDIA GPU with >=8 GB VRAM.
3. CUDA 12.x/13.x ecosystem.
4. `uv` package manager installed.
5. Python `3.12.10` (`.python-version`, notebook metadata, venv config).

### 4.2 Dependency Resolution Contract

1. `pyproject.toml` defines declared minimums.
2. `uv.lock` pins exact resolved versions.
3. `.venv/pyvenv.cfg` confirms local interpreter and uv build metadata.

### 4.3 Exact Setup Commands Documented in Repo

These commands are documented (not executed for this handbook):

```bash
git clone https://github.com/pypi-ahmad/qlora-complaint-intent-classifier.git
cd qlora-complaint-intent-classifier
uv sync
uv run python -m ipykernel install --user \
  --name qlora-complaint-intent-classifier \
  --display-name "Python 3.12 (qlora-env)"
```

Notebook launch command documented:

```bash
uv run jupyter notebook qlora_complaint_intent_classifier.ipynb
```

Inference commands documented:

```bash
uv run python inference.py "My card was declined at the supermarket"
uv run python inference.py
uv run python inference.py --batch queries.txt
```

### 4.4 `.env` / Secret Configuration Reality

- No `.env` file exists in visible repository files.
- No required custom env vars are hardcoded in scripts.
- Notebook output does show a Hugging Face warning about unauthenticated requests, implying optional `HF_TOKEN` can improve rate limits, but no mandatory token variable is declared in repo code.

### 4.5 Conceptual Boot Sequence (What Happens at Runtime)

1. **Environment bootstrap**
   - `uv sync` materializes `.venv` and installs pinned stack.
2. **Notebook execution path**
   - loads dataset
   - builds formatted prompt text
   - runs baseline
   - fine-tunes QLoRA adapter
   - saves adapter/tokenizer/checkpoints
   - runs full evaluation and plotting
3. **Serving path via `inference.py`**
   - loads base model + adapter
   - prompts with fixed system policy
   - greedy decode + normalization fallback
   - returns one label per query.

### 4.6 Development Workflow Guidance Grounded in This Repo

1. If you modify notebook logic, authoritative source is `make_notebook.py` (regenerate notebook after changes).
2. Keep label taxonomy synchronized between:
   - notebook `LABEL_NAMES`
   - `inference.py` `LABEL_NAMES`
   - any external service contract.
3. If replacing base model, update consistently in:
   - `make_notebook.py` `MODEL_ID`
   - `inference.py` `BASE_MODEL_ID`
   - adapter metadata location/compatibility assumptions.
4. Keep `MAX_LENGTH` and tokenizer truncation behavior aligned across training and inference.

---

## 💼 Module 5: Tech Interview & Hiring Preparation

### 5.1 Five Core Technical Interview Questions

1. Why does this repository keep training logic primarily in a generated notebook (`make_notebook.py` -> `.ipynb`) instead of a pure module pipeline, and what are the maintenance tradeoffs?
2. Explain how QLoRA is concretely implemented here, including exact quantization and LoRA adapter settings, and why those are appropriate for 8 GB VRAM constraints.
3. Walk through how `predict(...)` in `inference.py` converts free-form model output into one of the 77 labels, and discuss robustness implications.
4. The repository stores multiple checkpoints (`626`, `1252`, `1878`) and a final adapter bundle. How would you design rollback, model selection, and reproducibility policies from these artifacts?
5. There are file-level inconsistencies (for example dataset row counts and model-card completeness). How would you enforce documentation-code consistency in CI for this project?

---

### 5.2 Three Hard Engineering Scenarios (Repo-Specific)

1. **Throughput scenario:** Inference latency doubles under high query volume. You must optimize `load_model` and `predict` behavior without changing label accuracy contract.
2. **Data contract drift scenario:** A future Banking77 revision changes label naming/casing, breaking `LABEL_NAMES` assumptions and causing misclassification from normalization fallback.
3. **Training reliability scenario:** Mid-training interruptions produce partially divergent checkpoints; you must choose a promotion strategy between `checkpoint-1252` and `checkpoint-1878` using only persisted telemetry.

---

### 5.3 Detailed Model Answers

#### Q1 Model Answer
This repo uses a **code-generated notebook pattern**: `make_notebook.py` is the maintainable source, and `qlora_complaint_intent_classifier.ipynb` is the executable artifact with rich outputs/plots. Benefits are pedagogical readability and reproducible narrative flow. Tradeoffs are notebook drift risk and dual-artifact management. A senior engineer should preserve one-way authority (`make_notebook.py` authoritative), and auto-regenerate/check notebook diffs in CI.

#### Q2 Model Answer
QLoRA implementation is explicit and consistent across training and inference prep: 4-bit NF4 quantized base with BF16 compute and double quant (`BitsAndBytesConfig`), then LoRA adapters with `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, target modules spanning attention and MLP projections. Trainer uses `paged_adamw_8bit`, gradient checkpointing, effective batch `16`, and max length `256`. This is coherent for 8 GB-class hardware, as confirmed by stored notebook VRAM prints.

#### Q3 Model Answer
`predict(...)` enforces a deterministic classification pipeline:
1. fixed system instruction requiring one lowercase underscore label,
2. greedy generation with tight token cap,
3. normalization regex,
4. exact match,
5. prefix match,
6. char-overlap fallback.
This design improves resilience when model emits noise around labels. Risk: overlap fallback can mask deeper taxonomy drift; therefore label validation and calibration tests should be added around edge classes (e.g., semantically close transfer/payment labels).

#### Q4 Model Answer
Checkpoint strategy should rely on artifact immutability and telemetry:
- checkpoint 626/1252/1878 each carry full `trainer_state.json`, optimizer, scheduler, RNG, adapter weights.
- final adapter equals checkpoint-1878 adapter hash in this repo.
Promotion policy: prefer highest-step checkpoint with complete telemetry unless eval regression detected. Rollback policy: preserve checkpoint metadata registry (global_step, epoch, hashes) and attach reproducibility manifest linking training args, adapter config, tokenizer config, and source commit.

#### Q5 Model Answer
Consistency controls should include:
- schema lint: verify README dataset split numbers against notebook output artifacts,
- label lint: compare `LABEL_NAMES` across training and inference sources,
- metadata lint: ensure model card fields are populated (checkpoint READMEs are currently generic templates),
- config lint: verify base model ID consistency across scripts/artifacts,
- lock lint: ensure `pyproject.toml` declared ranges remain compatible with `uv.lock` pins.

#### Scenario 1 Model Answer (Inference throughput)
Focus on three layers:
1. **Cold-start amortization:** avoid repeated `load_model` in request path; initialize once per worker.
2. **Batching:** extend `predict` path to support mini-batch tokenization and generation for concurrent requests.
3. **Output post-processing cost:** normalization/fallback loops are cheap, but can be vectorized for batches.
Maintain exact output contract (`one label from LABEL_NAMES`) and keep `do_sample=False` to preserve deterministic class behavior.

#### Scenario 2 Model Answer (Label drift)
Introduce a label contract loader and validation gate:
1. derive authoritative label list from dataset metadata during training,
2. persist it with adapter artifact,
3. load same label list in inference instead of static duplicated list,
4. fail fast if adapter label list mismatches runtime inference list.
This removes manual drift points (currently hard-coded list in `inference.py`) and protects against casing/punctuation shifts.

#### Scenario 3 Model Answer (Checkpoint promotion under interruption)
Use persisted evidence hierarchy:
1. verify artifact integrity (hash/size and required members present),
2. inspect `trainer_state.json` for latest complete step/epoch,
3. compare loss/accuracy trend continuity from `log_history`,
4. prefer checkpoint with complete final state unless interrupted write is detected,
5. run deterministic evaluation on held-out split before production promotion.
In this repository, checkpoint-1878 appears final (`global_step=max_steps=1878`, epoch 3.0), making it default promotion candidate.

---

## Appendix A: Core Operational Metrics Stored in Notebook

- Baseline (200 subset): `Accuracy 4.50%`, `Macro F1 4.18%`
- Fine-tuned (full 3,080 test): `Accuracy 92.37%`, `Macro F1 91.97%`
- Improvement: `+87.87 pp` accuracy, `+87.79 pp` macro F1
- Training: `global steps 1878`, `training loss 0.4705`, `runtime 43.2 min`
- Qualitative sample: `13/15` correct

---

## Appendix B: Repository Consistency Findings (Static)

1. **Dataset split mismatch:**
   - `README.md` dataset table states train `13,083`.
   - Notebook dataset output and logic report train `10,003`.

2. **Adapter size wording mismatch:**
   - README notes adapter about `20–30 MB`.
   - Actual adapter files are `36,981,856` bytes.

3. **Model card completeness mismatch:**
   - Top-level adapter README is concise and usable.
   - Checkpoint README files are generic templates with many `[More Information Needed]` placeholders.

4. **Label literal caveat:**
   - One class label is persisted as `reverted_card_payment?` in inference constants and appears in notebook diagnostics.

---

## Appendix C: Quick Mastery Checklist (Zero to Hero)

1. Read `make_notebook.py` first to understand the true source architecture.
2. Inspect notebook stored outputs to understand achieved performance and failure modes.
3. Trace `inference.py` end-to-end and verify label fallback behavior mentally.
4. Study `adapter_config.json`, `tokenizer_config.json`, and `trainer_state.json` to internalize artifact contracts.
5. Reconcile README claims with artifact evidence before communicating project metrics externally.

# Interview Cram Guide: `qlora-complaint-intent-classifier`

Use this as a rapid prep sheet before technical interviews on this exact repository.

## 1) 60-Second System Summary

- Purpose: fine-tune `Qwen/Qwen2.5-1.5B-Instruct` for Banking77 multi-class intent classification.
- Training orchestration is notebook-centric:
  - Source builder: `make_notebook.py`
  - Executed artifact: `qlora_complaint_intent_classifier.ipynb`
- Inference entrypoint: `inference.py`
- Adapter artifact directory: `qlora-banking77-adapter/`
- Core method: QLoRA (4-bit NF4 base + LoRA adapters) with SFT.

## 2) Architecture-at-a-Glance

```text
make_notebook.py
  -> writes qlora_complaint_intent_classifier.ipynb
      -> loads Banking77 dataset
      -> formats chat training text
      -> baseline zero-shot eval (200 samples)
      -> QLoRA fine-tuning (3 epochs)
      -> saves adapter/tokenizer/checkpoints
      -> evaluates full test split (3080)

inference.py
  -> loads base model + saved adapter
  -> applies fixed system prompt
  -> greedy decoding + normalization fallback
  -> returns one of 77 intent labels
```

## 3) Must-Know Files

- `pyproject.toml`: project/dependency contract, Python >= `3.12.10`, `torch-backend = "cu128"`.
- `uv.lock`: exact pinned versions.
- `make_notebook.py`: authoritative training/eval pipeline definition.
- `qlora_complaint_intent_classifier.ipynb`: stored outputs/metrics from execution.
- `inference.py`: runtime prediction logic and CLI.
- `qlora-banking77-adapter/adapter_config.json`: persisted LoRA hyperparameters.
- `qlora-banking77-adapter/checkpoint-1878/trainer_state.json`: final training telemetry.

## 4) Exact Key Configs (Quote-Ready)

### Base model and quantization
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.bfloat16`
- `bnb_4bit_use_double_quant=True`

### LoRA
- `r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`
- `bias="none"`
- `task_type=TaskType.CAUSAL_LM`
- target modules:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`

### SFT training
- `max_length=256`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=4` (effective batch = 16)
- `num_train_epochs=3`
- `optim="paged_adamw_8bit"`
- `learning_rate=2e-4`
- `lr_scheduler_type="cosine"`
- `warmup_ratio=0.05`
- `weight_decay=0.01`
- `gradient_checkpointing=True`
- `bf16=True`, `fp16=False`
- `logging_steps=50`
- `save_strategy="epoch"`
- `report_to="none"`

### Inference behavior
- `MAX_NEW_TOKENS = 16`
- `do_sample=False` (greedy)
- tokenizer truncation at `max_length=256`

## 5) Dataset + Label Contract

- Loaded dataset revision in notebook logic:
  - `load_dataset("PolyAI/banking77", revision="refs/convert/parquet")`
- Notebook stored output shows:
  - Train: `10,003`
  - Test: `3,080`
  - Intents: `77`
- Labels built by:
  - `LABEL_NAMES = [n.lower() for n in train_ds.features["label"].names]`
- Important literal in inference label list: `reverted_card_payment?`

## 6) Reported Metrics (Stored in Notebook Output)

- Baseline zero-shot (`200` examples):
  - Accuracy: `4.50%`
  - Macro F1: `4.18%`
- Fine-tuned (`3,080` examples):
  - Accuracy: `92.37%`
  - Macro F1: `91.97%`
- Improvements:
  - Accuracy: `+87.87 pp`
  - Macro F1: `+87.79 pp`
- Training summary output:
  - Global steps: `1878`
  - Training loss: `0.4705`
  - Runtime: `43.2 min`

## 7) Inference Logic You Must Explain Clearly

`predict(text, model, tokenizer)` in `inference.py` does:
1. Build chat messages:
   - system classifier instruction
   - user query
2. Serialize with `apply_chat_template(..., add_generation_prompt=True)`
3. Generate greedily
4. Normalize generated text with regex to lowercase underscore form
5. Resolve label with ordered fallback:
   - exact match in `LABEL_NAMES`
   - prefix match
   - char-overlap score fallback

Interview angle: this fallback chain is a pragmatic robustness layer against slightly malformed generations.

## 8) Artifact Knowledge (Production/ML-Ops Questions)

`qlora-banking77-adapter/` includes:
- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `training_args.bin`
- checkpoints:
  - `checkpoint-626/`
  - `checkpoint-1252/`
  - `checkpoint-1878/`

Checkpoint telemetry highlights:
- `checkpoint-626`: `epoch=1.0`, `global_step=626`
- `checkpoint-1252`: `epoch=2.0`, `global_step=1252`
- `checkpoint-1878`: `epoch=3.0`, `global_step=1878`, `max_steps=1878`

## 9) Consistency Gaps (Great Senior-Level Talking Points)

1. README vs notebook data-size mismatch:
- README table states train `13,083`
- notebook output states train `10,003`

2. README adapter size note vs file size:
- README says adapter roughly `20–30 MB`
- actual adapter files are `36,981,856` bytes

3. Checkpoint model cards are generic templates with many placeholder fields.

Use this to discuss documentation governance and CI validation strategies.

## 10) 12 High-Value Interview Flash Q&A

1. Why QLoRA here?
- Enables fine-tuning on 8 GB class GPU by freezing quantized base model and training only LoRA adapters.

2. Why `paged_adamw_8bit`?
- Memory-aware optimizer strategy aligned with constrained VRAM training.

3. Why `max_length=256`?
- Notebook’s query-length analysis shows this comfortably covers Banking77 query lengths.

4. Why deterministic decoding (`do_sample=False`) for inference?
- Classification reliability; avoids stochastic label drift.

5. Why macro-F1 in addition to accuracy?
- Multi-class setup with many intents; macro-F1 gives class-balanced perspective.

6. Why not pure classification head model?
- Repository chooses instruction-following causal LM framing and trains label generation through SFT.

7. Main robustness weakness in current inference?
- Hard-coded label list duplication can drift from training-time label contract.

8. Where is reproducibility evidence stored?
- `training_args.bin`, checkpoint `trainer_state.json`, adapter/tokenizer configs, locked deps.

9. If latency spikes, first optimization?
- Keep model loaded once per worker; avoid reloading in request path; batch requests.

10. If data schema changes upstream?
- Derive/persist label contract from training data and enforce compatibility checks at inference startup.

11. Why keep `make_notebook.py`?
- Maintains notebook as generated artifact from a scriptable source of truth.

12. Biggest governance improvement?
- Automated doc-vs-artifact consistency checks (dataset sizes, metrics, model-card completeness, label parity).

## 11) Fast “Walk Me Through the Repo” Script (2 Minutes)

- Start at `pyproject.toml` and `uv.lock` for reproducible environment.
- Move to `make_notebook.py` as the actual pipeline definition.
- Explain notebook phases: data -> baseline -> QLoRA train -> full eval -> diagnostics.
- Show artifact contract in `qlora-banking77-adapter/`.
- End with `inference.py` deterministic prediction + fallback resolution.
- Close with consistency gaps and how you’d harden CI around them.

## 12) Exact Version Recap (Pinned/Observed)

- Python: `3.12.10`
- torch: `2.12.0`
- transformers: `5.10.2`
- peft: `0.19.1`
- trl: `1.5.1`
- bitsandbytes: `0.49.2`
- datasets: `5.0.0`
- accelerate: `1.13.0`
- scikit-learn: `1.9.0`
- pandas: `3.0.3`
- numpy: `2.4.6`

---

If you want next: I can produce an ultra-short “15-minute pre-interview drill” version (single page, no prose, only prompts and answers).

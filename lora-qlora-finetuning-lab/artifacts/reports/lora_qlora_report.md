# LoRA vs QLoRA Fine-Tuning Report

Generated at: `2026-06-12T16:45:11`

## Experiment Setup

- Dataset: `dair-ai/emotion`
- Task: Emotion classification with generative label prediction
- LoRA model: `distilgpt2`
- QLoRA model: `facebook/opt-350m`
- Seed: `42`

## Evaluation Results

| Run | Model | Accuracy | Macro F1 | Samples |
|---|---|---:|---:|---:|
| baseline_lora | distilgpt2 | 0.0000 | 0.0000 | 40 |
| tuned_lora | distilgpt2 | 0.3125 | 0.0984 | 80 |
| baseline_qlora | facebook/opt-350m | 0.1000 | 0.1365 | 40 |
| tuned_qlora | facebook/opt-350m | 0.3625 | 0.1405 | 80 |


## Delta vs Baseline

- LoRA gain (accuracy): **0.3125**
- QLoRA gain (accuracy): **0.2625**
- LoRA gain (macro F1): **0.0984**
- QLoRA gain (macro F1): **0.0040**

## Notes

- Baselines are measured before adapter training on the same held-out split.
- Train/validation/test are sampled deterministically with a fixed seed.
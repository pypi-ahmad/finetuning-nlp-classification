"""Configuration models and loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    max_length: int = 192
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["c_attn", "c_proj", "c_fc"])


class DataConfig(BaseModel):
    source: str = "synthetic"
    train_size: int = 500
    val_size: int = 120
    test_size: int = 180
    hf_dataset: str = "PolyAI/banking77"
    hf_train_split: str = "train"
    hf_test_split: str = "test"


class TrainConfig(BaseModel):
    sft_epochs: int = 4
    dpo_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 3e-4
    dpo_learning_rate: float = 1.5e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    dpo_beta: float = 0.1


class EvalConfig(BaseModel):
    max_new_tokens: int = 80


class ProjectConfig(BaseModel):
    seed: int = 42
    output_root: str = "outputs"
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load project config from YAML."""

    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return ProjectConfig.model_validate(payload)

"""LoRA and QLoRA training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from lora_qlora_lab.config import Settings
from lora_qlora_lab.data.emotion_dataset import tokenize_for_causal_lm


def _target_modules_for_model(model_name: str) -> list[str]:
    name = model_name.lower()
    if "gpt2" in name:
        return ["c_attn", "c_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_tokenizer(model_name: str, token: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _build_lora_config(model_name: str) -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_target_modules_for_model(model_name),
    )


def _save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_lora(
    settings: Settings,
    sampled_splits: DatasetDict,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Fine-tune with standard LoRA on distilgpt2."""
    set_seed(settings.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(settings.lora_model_name, token=settings.hf_token)
    tokenized_train = tokenize_for_causal_lm(sampled_splits["train"], tokenizer, settings.max_length)
    tokenized_val = tokenize_for_causal_lm(sampled_splits["validation"], tokenizer, settings.max_length)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        settings.lora_model_name,
        torch_dtype=dtype,
        token=settings.hf_token,
    )
    base_model.config.use_cache = False

    model = get_peft_model(base_model, _build_lora_config(settings.lora_model_name))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_steps=settings.lora_max_steps,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=max(4, settings.lora_max_steps // 3),
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        seed=settings.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )

    logger.info("Starting LoRA training on {}", settings.lora_model_name)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "train": train_result.metrics,
        "eval": eval_metrics,
        "model": settings.lora_model_name,
        "method": "lora",
        "train_samples": len(tokenized_train),
        "validation_samples": len(tokenized_val),
    }
    _save_metrics(metrics, output_dir / "train_metrics.json")
    return adapter_dir, metrics


def train_qlora(
    settings: Settings,
    sampled_splits: DatasetDict,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Fine-tune with QLoRA on TinyLlama."""
    set_seed(settings.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(settings.qlora_model_name, token=settings.hf_token)
    tokenized_train = tokenize_for_causal_lm(sampled_splits["train"], tokenizer, settings.max_length)
    tokenized_val = tokenize_for_causal_lm(sampled_splits["validation"], tokenizer, settings.max_length)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        settings.qlora_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        token=settings.hf_token,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False

    model = get_peft_model(base_model, _build_lora_config(settings.qlora_model_name))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=settings.qlora_max_steps,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=max(2, settings.qlora_max_steps // 3),
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
        seed=settings.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )

    logger.info("Starting QLoRA training on {}", settings.qlora_model_name)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "train": train_result.metrics,
        "eval": eval_metrics,
        "model": settings.qlora_model_name,
        "method": "qlora",
        "train_samples": len(tokenized_train),
        "validation_samples": len(tokenized_val),
    }
    _save_metrics(metrics, output_dir / "train_metrics.json")
    return adapter_dir, metrics

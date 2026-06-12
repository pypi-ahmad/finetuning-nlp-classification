"""Model and tokenizer factory utilities."""

from __future__ import annotations

import os
from typing import Any

import torch
from loguru import logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .settings import ModelConfig


def load_tokenizer(model_name: str):
    """Load tokenizer and ensure pad token exists."""

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_cfg: ModelConfig, for_training: bool = True):
    """Load base model. Uses 4-bit path when available and requested."""

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    use_qlora = model_cfg.use_qlora and torch.cuda.is_available()
    if use_qlora:
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg.name,
                quantization_config=quant_cfg,
                device_map="auto",
                trust_remote_code=False,
            )
            if for_training:
                model = prepare_model_for_kbit_training(model)
            logger.info("QLoRA mode enabled (4-bit NF4).")
            return model
        except Exception as exc:  # pragma: no cover - depends on runtime CUDA stack
            logger.warning(
                "QLoRA requested but unavailable in this runtime; falling back to LoRA FP path. Error: {}",
                exc,
            )

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    logger.info("Running standard LoRA path (no 4-bit quantization).")
    return model


def build_lora_model(model_cfg: ModelConfig, for_training: bool = True):
    """Attach LoRA adapters to a base causal LM."""

    base_model = load_base_model(model_cfg=model_cfg, for_training=for_training)
    lora_cfg = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_cfg.target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    if for_training:
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval()
    return model


def load_adapter_for_inference(model_cfg: ModelConfig, adapter_path: str):
    """Load base model + adapter for evaluation/inference."""

    base_model = load_base_model(model_cfg=model_cfg, for_training=False)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()
    return model


def device_of(model: Any) -> torch.device:
    """Resolve primary torch device for a model."""

    return next(model.parameters()).device

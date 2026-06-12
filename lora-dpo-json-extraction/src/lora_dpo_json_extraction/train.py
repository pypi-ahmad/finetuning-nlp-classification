"""Training loops for SFT (LoRA/QLoRA) and DPO."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .datasets import DPOCollator, DPODataset, SFTCollator, SFTDataset
from .models import build_lora_model, device_of, load_base_model, load_tokenizer
from .settings import ProjectConfig
from .utils import ensure_dir, write_json


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _evaluate_sft_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            output = model(**batch)
            losses.append(float(output.loss.detach().cpu().item()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def train_sft(
    cfg: ProjectConfig,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    """Train LoRA/QLoRA adapter with supervised fine-tuning."""

    tokenizer = load_tokenizer(cfg.model.name)
    model = build_lora_model(cfg.model, for_training=True)
    logger.info("LoRA trainable parameters:")
    model.print_trainable_parameters()

    train_dataset = SFTDataset(train_rows, tokenizer, cfg.model.max_length)
    val_dataset = SFTDataset(val_rows, tokenizer, cfg.model.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=SFTCollator(tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=SFTCollator(tokenizer),
    )

    device = device_of(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        parameters,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    total_update_steps = math.ceil(len(train_loader) / cfg.train.gradient_accumulation_steps)
    total_update_steps *= cfg.train.sft_epochs
    warmup_steps = int(total_update_steps * cfg.train.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    history: list[dict[str, float]] = []
    global_step = 0

    for epoch in range(cfg.train.sft_epochs):
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"SFT epoch {epoch + 1}/{cfg.train.sft_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = _to_device(batch, device)
            output = model(**batch)
            loss = output.loss / cfg.train.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu().item())

            if step % cfg.train.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters, cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss = _evaluate_sft_loss(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
            }
        )
        logger.info(
            "SFT epoch {} complete | train_loss={} val_loss={}",
            epoch + 1,
            round(avg_train_loss, 4),
            round(avg_val_loss, 4),
        )

    adapter_dir = run_dir / "sft_adapter"
    tokenizer_dir = run_dir / "tokenizer"
    ensure_dir(adapter_dir)
    ensure_dir(tokenizer_dir)

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    summary = {
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "history": history,
        "global_steps": global_step,
    }
    write_json(run_dir / "sft_training_summary.json", summary)
    return summary


def _sequence_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence log-prob over masked target tokens."""

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :]
    target_tokens = input_ids[:, 1:]
    token_mask = loss_mask[:, 1:]

    token_log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = token_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

    return (token_log_probs * token_mask).sum(dim=-1)


def train_dpo(
    cfg: ProjectConfig,
    preference_rows: list[dict[str, Any]],
    run_dir: Path,
    sft_adapter_dir: str,
) -> dict[str, Any]:
    """Train a second LoRA adapter stage using DPO objective."""

    tokenizer = load_tokenizer(cfg.model.name)

    policy_base = load_base_model(cfg.model, for_training=True)
    policy_model = PeftModel.from_pretrained(policy_base, sft_adapter_dir, is_trainable=True)
    policy_model.train()

    ref_base = load_base_model(cfg.model, for_training=False)
    reference_model = PeftModel.from_pretrained(ref_base, sft_adapter_dir, is_trainable=False)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    device = device_of(policy_model)

    dataset = DPODataset(preference_rows, tokenizer, cfg.model.max_length)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=DPOCollator(tokenizer),
    )

    params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg.train.dpo_learning_rate)

    total_update_steps = math.ceil(len(loader) / cfg.train.gradient_accumulation_steps)
    total_update_steps *= cfg.train.dpo_epochs
    warmup_steps = int(total_update_steps * cfg.train.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    history: list[dict[str, float]] = []
    global_step = 0

    for epoch in range(cfg.train.dpo_epochs):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"DPO epoch {epoch + 1}/{cfg.train.dpo_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = _to_device(batch, device)

            chosen_pi = _sequence_log_probs(
                policy_model,
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                loss_mask=batch["chosen_loss_mask"],
            )
            rejected_pi = _sequence_log_probs(
                policy_model,
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                loss_mask=batch["rejected_loss_mask"],
            )

            with torch.no_grad():
                chosen_ref = _sequence_log_probs(
                    reference_model,
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                    loss_mask=batch["chosen_loss_mask"],
                )
                rejected_ref = _sequence_log_probs(
                    reference_model,
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                    loss_mask=batch["rejected_loss_mask"],
                )

            advantage = (chosen_pi - rejected_pi) - (chosen_ref - rejected_ref)
            loss = -F.logsigmoid(cfg.train.dpo_beta * advantage).mean()
            loss = loss / cfg.train.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu().item())

            if step % cfg.train.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        avg_loss = running_loss / max(len(loader), 1)
        history.append({"epoch": float(epoch + 1), "dpo_loss": float(avg_loss)})
        logger.info("DPO epoch {} complete | dpo_loss={}", epoch + 1, round(avg_loss, 4))

    adapter_dir = run_dir / "dpo_adapter"
    ensure_dir(adapter_dir)
    policy_model.save_pretrained(adapter_dir)

    summary = {
        "adapter_dir": str(adapter_dir),
        "history": history,
        "global_steps": global_step,
    }
    write_json(run_dir / "dpo_training_summary.json", summary)
    return summary

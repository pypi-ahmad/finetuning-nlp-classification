"""
triage_config.py — one place for every knob (model, data sizing, paths, LoRA).
Imported by the data, model, calibration, active-learning and inference modules
so the notebook and CLI never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass
class TriageConfig:
    # ── Model — 1.5B instruct fits 8 GB in 4-bit QLoRA ───────────────────────
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # ── Data curation ────────────────────────────────────────────────────────
    cfpb_parquet: Path = field(default=HERE / "data" / "raw" / "cfpb" / "cfpb_subset.parquet")
    processed_dir: Path = field(default=HERE / "data" / "processed")
    min_l2_count: int = 80          # drop issues (L2) with fewer than this (anti-sparsity)
    max_per_l2: int = 400           # cap an over-represented issue (balance)
    min_chars: int = 60             # drop near-empty narratives
    max_chars: int = 1200           # truncate long narratives before tokenising
    use_banking77: bool = True      # fold Banking77 in as L1 augmentation
    banking77_cap: int = 2000       # subsample B77 so CFPB stays the backbone

    # ── Splits ────────────────────────────────────────────────────────────────
    test_frac: float = 0.15
    val_frac: float = 0.10
    pool_frac: float = 0.25         # held-out unlabeled pool for active learning
    train_cap: int = 6000           # cap labelled train rows (8 GB-friendly run time)
    seed: int = 42

    # ── Tokenisation ──────────────────────────────────────────────────────────
    max_length: int = 704           # rubric + truncated narrative + label (max
    #                                 observed prompt+completion ≈632; margin so
    #                                 keep_start truncation never clips the label)

    # ── LoRA / QLoRA ──────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ── Training (8 GB-friendly) ──────────────────────────────────────────────
    epochs: float = 2.0
    per_device_batch_size: int = 2  # long sequences (640) → small micro-batch
    grad_accum: int = 8             # effective batch 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    eval_batch_size: int = 8        # scorer batch (no grads → headroom)

    # ── Active learning ───────────────────────────────────────────────────────
    al_rounds: int = 2
    al_budget: int = 400            # examples relabelled per round

    # ── Paths ─────────────────────────────────────────────────────────────────
    adapter_dir: Path = field(default=HERE / "outputs" / "qlora-triage-adapter")
    figures_dir: Path = field(default=HERE / "figures")


CONFIG = TriageConfig()

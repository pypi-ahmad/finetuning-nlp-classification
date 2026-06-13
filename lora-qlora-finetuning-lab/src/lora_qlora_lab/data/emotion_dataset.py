"""Dataset loading and prompt formatting."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@dataclass
class EvalRecord:
    """Evaluation record with prompt and expected label."""

    text: str
    gold_label: str
    prompt: str


def build_prompt(text: str) -> str:
    """Create a deterministic classification prompt."""
    labels_text = ", ".join(LABELS)
    return (
        "Classify the emotion in the given text. "
        f"Choose exactly one label from: {labels_text}.\n"
        f"Text: {text}\n"
        "Label:"
    )


def build_train_text(text: str, label: str) -> str:
    """Prompt + target output for supervised LM fine-tuning."""
    return f"{build_prompt(text)} {label}"


def decode_label(label_id: int) -> str:
    return LABELS[label_id]


def extract_label(prediction: str) -> str:
    """Extract first valid label mention from model output."""
    pred_lower = prediction.lower()
    for label in LABELS:
        if f"{label}" in pred_lower:
            return label
    return "unknown"


def load_emotion_splits(dataset_name: str, seed: int) -> DatasetDict:
    """Load the full emotion dataset from Hugging Face Hub."""
    random.seed(seed)
    return load_dataset(dataset_name)


def sample_splits(dataset: DatasetDict, train_n: int, val_n: int, test_n: int, seed: int) -> DatasetDict:
    """Create deterministic sampled splits for faster training."""
    train = dataset["train"].shuffle(seed=seed).select(range(min(train_n, len(dataset["train"]))))
    validation = dataset["validation"].shuffle(seed=seed).select(
        range(min(val_n, len(dataset["validation"])))
    )
    test = dataset["test"].shuffle(seed=seed).select(range(min(test_n, len(dataset["test"]))))
    return DatasetDict({"train": train, "validation": validation, "test": test})


def save_raw_splits(sampled: DatasetDict, output_dir: Path) -> None:
    """Persist sampled rows as JSON for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        path = output_dir / f"emotion_{split}.json"
        path.write_text(json.dumps(sampled[split].to_list(), indent=2), encoding="utf-8")


def build_eval_records(dataset: Dataset, limit: int) -> list[EvalRecord]:
    """Convert dataset rows to evaluation prompt records."""
    records: list[EvalRecord] = []
    for row in dataset.select(range(min(limit, len(dataset)))):
        text = str(row["text"])
        label = decode_label(int(row["label"]))
        records.append(EvalRecord(text=text, gold_label=label, prompt=build_prompt(text)))
    return records


def tokenize_for_causal_lm(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenize prompt-completion strings for causal LM training."""

    def _map_fn(batch: dict[str, list[object]]) -> dict[str, list[list[int]]]:
        texts: list[str] = []
        for text, label_id in zip(batch["text"], batch["label"], strict=True):
            label = decode_label(int(label_id))
            texts.append(build_train_text(str(text), label))
        tokens = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        return tokens

    return dataset.map(_map_fn, batched=True, remove_columns=dataset.column_names)


def save_eval_records(records: Iterable[EvalRecord], output_path: Path) -> None:
    payload = [record.__dict__ for record in records]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

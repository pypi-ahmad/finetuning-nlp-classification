from __future__ import annotations

from datasets import Dataset, DatasetDict

from lora_qlora_lab.data.emotion_dataset import sample_splits


def test_sample_splits_respects_limits() -> None:
    split = Dataset.from_dict({"text": [f"t{i}" for i in range(50)], "label": [0] * 50})
    ds = DatasetDict({"train": split, "validation": split, "test": split})

    sampled = sample_splits(ds, train_n=20, val_n=10, test_n=5, seed=42)

    assert len(sampled["train"]) == 20
    assert len(sampled["validation"]) == 10
    assert len(sampled["test"]) == 5

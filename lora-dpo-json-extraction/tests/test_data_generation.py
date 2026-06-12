import pytest

from lora_dpo_json_extraction.data import build_preference_split, build_splits


def test_splits_have_expected_sizes() -> None:
    splits = build_splits(train_size=10, val_size=4, test_size=6, seed=7, source="synthetic")
    assert len(splits["train"]) == 10
    assert len(splits["val"]) == 4
    assert len(splits["test"]) == 6


def test_preference_pairs_contain_chosen_and_rejected() -> None:
    splits = build_splits(train_size=8, val_size=2, test_size=2, seed=5, source="synthetic")
    prefs = build_preference_split(splits["train"], seed=5)
    assert len(prefs) == 8
    assert all("chosen" in row and "rejected" in row for row in prefs)


def test_unknown_data_source_raises() -> None:
    with pytest.raises(ValueError):
        build_splits(train_size=1, val_size=1, test_size=1, seed=1, source="unknown_source")

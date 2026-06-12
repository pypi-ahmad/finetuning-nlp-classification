"""Torch dataset wrappers for SFT and DPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """Supervised fine-tuning dataset with response-only loss masking."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt_ids = self.tokenizer.encode(row["prompt"], add_special_tokens=False)
        response_ids = self.tokenizer.encode(row["response"], add_special_tokens=False)
        response_ids = response_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]

        # Keep loss on response tokens only. If truncation clipped the prompt,
        # the loss mask remains valid by clamping at sequence length.
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DPODataset(Dataset):
    """Preference dataset for DPO with chosen/rejected responses."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, prompt: str, response: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False) + [
            self.tokenizer.eos_token_id
        ]
        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
        prompt_len = min(len(prompt_ids), len(input_ids))
        loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
        return input_ids, loss_mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        chosen_ids, chosen_mask = self._encode(row["prompt"], row["chosen"])
        rejected_ids, rejected_mask = self._encode(row["prompt"], row["rejected"])

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_loss_mask": torch.tensor(chosen_mask, dtype=torch.float32),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_loss_mask": torch.tensor(rejected_mask, dtype=torch.float32),
        }


@dataclass(slots=True)
class SFTCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].shape[0] for feature in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []

        for feature in features:
            pad_len = max_len - feature["input_ids"].shape[0]
            input_ids.append(
                torch.cat(
                    [feature["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]
                )
            )
            labels.append(
                torch.cat(
                    [
                        feature["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long),
                    ]
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        feature["attention_mask"],
                        torch.zeros((pad_len,), dtype=torch.long),
                    ]
                )
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


@dataclass(slots=True)
class DPOCollator:
    tokenizer: Any

    def _pad(self, tensors: list[torch.Tensor], pad_value: float) -> torch.Tensor:
        max_len = max(x.shape[0] for x in tensors)
        padded: list[torch.Tensor] = []
        for tensor in tensors:
            pad_len = max_len - tensor.shape[0]
            if pad_len:
                padded.append(
                    torch.cat(
                        [tensor, torch.full((pad_len,), pad_value, dtype=tensor.dtype)]
                    )
                )
            else:
                padded.append(tensor)
        return torch.stack(padded)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = float(self.tokenizer.pad_token_id)
        chosen_input_ids = self._pad([f["chosen_input_ids"] for f in features], pad_id).long()
        chosen_loss_mask = self._pad([f["chosen_loss_mask"] for f in features], 0.0)
        rejected_input_ids = self._pad([f["rejected_input_ids"] for f in features], pad_id).long()
        rejected_loss_mask = self._pad([f["rejected_loss_mask"] for f in features], 0.0)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": (chosen_input_ids != int(pad_id)).long(),
            "chosen_loss_mask": chosen_loss_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": (rejected_input_ids != int(pad_id)).long(),
            "rejected_loss_mask": rejected_loss_mask,
        }

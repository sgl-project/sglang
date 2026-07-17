# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PromptToSampleBatchExpander:
    """Expand selected conditioning from prompt order to sample order."""

    prompt_batch_size: int
    sample_batch_size: int

    @classmethod
    def from_batch(cls, batch):
        num_outputs = int(batch.num_outputs_per_prompt or 1)
        if num_outputs <= 1:
            return None
        if isinstance(batch.prompt, list):
            prompt_batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            prompt_batch_size = 1
        else:
            raise ValueError(
                "Multi-output conditioning requires prompt text so the prompt "
                "batch size is unambiguous."
            )
        if prompt_batch_size <= 0:
            raise ValueError("Multi-output conditioning requires at least one prompt.")
        return cls(prompt_batch_size, prompt_batch_size * num_outputs)

    def _expand_tensor(self, value: torch.Tensor, name: str) -> torch.Tensor:
        current_batch_size = value.shape[0]
        if current_batch_size == self.sample_batch_size:
            return value
        if current_batch_size != self.prompt_batch_size:
            raise ValueError(
                f"{name} has batch dim {current_batch_size} (shape "
                f"{tuple(value.shape)}); expected {self.prompt_batch_size} "
                f"(per-prompt) or {self.sample_batch_size} (per-sample)."
            )
        repeats = self.sample_batch_size // self.prompt_batch_size
        return value.repeat_interleave(repeats, dim=0)

    def _expand_tensors(self, value, name: str):
        """Expand a tensor or each tensor in a list, preserving its container."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return self._expand_tensor(value, name)
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a tensor, list of tensors, or None.")
        if any(
            item is not None and not isinstance(item, torch.Tensor) for item in value
        ):
            raise TypeError(f"{name} entries must be tensors or None.")
        return [
            self._expand_tensor(item, f"{name}[{index}]") if item is not None else None
            for index, item in enumerate(value)
        ]

    def _expand_sequence_lengths(
        self, value: list[list[int] | None] | None, name: str
    ) -> list[list[int] | None] | None:
        if value is None:
            return None
        repeats = self.sample_batch_size // self.prompt_batch_size
        expanded = []
        for index, sequence_lengths in enumerate(value):
            if (
                sequence_lengths is None
                or len(sequence_lengths) == self.sample_batch_size
            ):
                expanded.append(sequence_lengths)
            elif len(sequence_lengths) == self.prompt_batch_size:
                expanded.append(
                    [
                        sequence_length
                        for sequence_length in sequence_lengths
                        for _ in range(repeats)
                    ]
                )
            else:
                raise ValueError(
                    f"{name}[{index}] has {len(sequence_lengths)} entries; expected "
                    f"{self.prompt_batch_size} (per-prompt) or "
                    f"{self.sample_batch_size} (per-sample)."
                )
        return expanded

    def expand_field(self, batch, field_name: str) -> None:
        """Expand one field in place, dispatching from its value type."""
        value = getattr(batch, field_name)
        if value is None:
            return
        if isinstance(value, torch.Tensor) or (
            isinstance(value, list)
            and all(item is None or isinstance(item, torch.Tensor) for item in value)
        ):
            expanded = self._expand_tensors(value, field_name)
        elif isinstance(value, list) and all(
            item is None or isinstance(item, list) for item in value
        ):
            expanded = self._expand_sequence_lengths(value, field_name)
        else:
            raise TypeError(
                f"{field_name} must be a tensor, list of tensors, "
                "list of sequence-length lists, or None."
            )
        setattr(batch, field_name, expanded)

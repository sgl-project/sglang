from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedInputs:
    tokens: torch.Tensor
    positions: torch.Tensor

    @classmethod
    def allocate(cls, *, capacity: int, device: torch.device) -> "ExpectedInputs":
        # tokens default to -1 (the WRITE kernel's "skip token check" sentinel). The plan-side
        # gather kernel overwrites the [:total_write] prefix in-place when the validator is on; any
        # tail beyond ``total_write`` (cuda-graph padding) keeps the sentinel and is silently
        # skipped by the WRITE kernel.
        return cls(
            tokens=torch.full((capacity,), -1, dtype=torch.int64, device=device),
            positions=torch.empty(capacity, dtype=torch.int64, device=device),
        )

    def slice(self, num_tokens: int) -> "ExpectedInputs":
        return ExpectedInputs(
            tokens=self.tokens[:num_tokens],
            positions=self.positions[:num_tokens],
        )

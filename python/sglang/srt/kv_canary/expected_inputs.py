from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedInputs:
    tokens: torch.Tensor
    positions: torch.Tensor

    @classmethod
    def allocate(cls, *, capacity: int, device: torch.device) -> ExpectedInputs:
        return cls(
            tokens=torch.empty(capacity, dtype=torch.int64, device=device),
            positions=torch.empty(capacity, dtype=torch.int64, device=device),
        )

    def slice(self, num_tokens: int) -> ExpectedInputs:
        return ExpectedInputs(
            tokens=self.tokens[:num_tokens],
            positions=self.positions[:num_tokens],
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class TokenOracle(Protocol):
    """Deterministic (generalized_req_id, position) -> token_id mapping."""

    def expected_tokens(
        self, *, generalized_req_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class HashOracle:
    """token_id = splitmix64(generalized_req_id XOR position) % vocab_size."""

    vocab_size: int

    def expected_tokens(
        self, *, generalized_req_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        x = generalized_req_ids.to(torch.int64) ^ positions.to(torch.int64)
        x = _splitmix64_tensor(x)
        return _uint64_mod(x, self.vocab_size).to(torch.int32)


_C1: int = -4658895280553007687  # 0xBF58476D1CE4E5B9 as signed int64
_C2: int = -7723592293110705685  # 0x94D049BB133111EB as signed int64


def _splitmix64_tensor(x: torch.Tensor) -> torch.Tensor:
    x = (x ^ _logical_shr(x, 30)) * _C1
    x = (x ^ _logical_shr(x, 27)) * _C2
    x = x ^ _logical_shr(x, 31)
    return x


def _logical_shr(x: torch.Tensor, n: int) -> torch.Tensor:
    return (x >> n) & ((1 << (64 - n)) - 1)


def _uint64_mod(x: torch.Tensor, mod: int) -> torch.Tensor:
    offset = (1 << 64) % mod
    base = x % mod
    correction = (x < 0).to(x.dtype) * offset
    return (base + correction) % mod

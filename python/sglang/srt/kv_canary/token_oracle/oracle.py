from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class TokenOracle(Protocol):
    """Deterministic (req_id, position) -> token_id mapping.

    All inputs/outputs live on device. Element-wise:
    `out[i] = oracle(req_ids[i], positions[i])`. No host sync. Input dtypes int64 or int32;
    output dtype int32 on the same device.

    The oracle has NO knowledge of canary. Both the _OracleSampler (output side) and
    fill_expected_inputs (input-check side) consume the same oracle so the two perspectives
    agree by construction.
    """

    def expected_tokens(
        self, *, req_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class HashOracle:
    """token_id = splitmix64(seed XOR req_id XOR position) % vocab_size. Cheap, stateless.

    Default oracle. The seed is independent of CANARY_CHAIN_ANCHOR (which lives in jit_kernel
    and is not run-time configurable).
    """

    seed: int
    vocab_size: int

    def expected_tokens(
        self, *, req_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        seed = torch.tensor(self.seed, dtype=torch.int64, device=req_ids.device)
        x = req_ids.to(torch.int64) ^ positions.to(torch.int64) ^ seed
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
    correction = torch.where(
        x < 0,
        torch.tensor(offset, dtype=torch.int64, device=x.device),
        torch.tensor(0, dtype=torch.int64, device=x.device),
    )
    return (base + correction) % mod

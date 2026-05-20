from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sglang.jit_kernel.kv_canary.verify_ref import splitmix64


class TokenIdOracle(Protocol):
    """Deterministic (req_id, position) -> token_id mapping.

    The oracle has NO knowledge of canary. Both the _OracleSampler (output side) and
    fill_expected_inputs (input-check side) consume the same oracle so the two perspectives
    agree by construction.
    """

    def expected_token(self, *, req_id: int, position: int) -> int: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class HashOracle:
    """token_id = splitmix64(seed XOR req_id XOR position) % vocab_size. Cheap, stateless.

    Default oracle. The seed is independent of CANARY_CHAIN_ANCHOR (which lives in jit_kernel
    and is not run-time configurable).
    """

    seed: int
    vocab_size: int

    def expected_token(self, *, req_id: int, position: int) -> int:
        return splitmix64(self.seed ^ req_id ^ position) % self.vocab_size

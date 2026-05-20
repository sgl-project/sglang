from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Protocol, Sequence


class Oracle(Protocol):
    """Deterministic (req_id, position) -> token_id mapping.

    The oracle has NO knowledge of canary. Both the OracleSampler (output side) and
    fill_expected_inputs (input-check side) consume the same oracle so the two perspectives
    agree by construction.
    """

    def expected_token(self, *, req_id: int, position: int) -> int: ...

    def expected_tokens_batch(
        self,
        *,
        req_ids: Sequence[int],
        positions: Sequence[int],
    ) -> List[int]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class HashOracle:
    """token_id = splitmix64(seed XOR req_id XOR position) % vocab_size. Cheap, stateless.

    Default oracle. The seed is independent of CANARY_CHAIN_ANCHOR (which lives in jit_kernel
    and is not run-time configurable).
    """

    seed: int
    vocab_size: int

    def expected_token(self, *, req_id: int, position: int) -> int:
        mixed = (self.seed ^ req_id ^ position) & _U64_MASK
        return _splitmix64(mixed) % self.vocab_size

    def expected_tokens_batch(
        self,
        *,
        req_ids: Sequence[int],
        positions: Sequence[int],
    ) -> List[int]:
        if len(req_ids) != len(positions):
            raise ValueError(
                f"req_ids ({len(req_ids)}) and positions ({len(positions)}) must have equal length"
            )
        return [
            self.expected_token(req_id=r, position=p)
            for r, p in zip(req_ids, positions)
        ]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptedOracle:
    """Pre-loaded ``{(req_id, position): token_id}`` table.

    Used by manual edge-case tests that need exact token sequences, e.g.
    'req 7 must see token 42 at position 100'. Lookup outside the table raises KeyError -
    caller is expected to fully cover every (req_id, position) the test will hit.
    """

    table: Mapping[tuple[int, int], int]

    def expected_token(self, *, req_id: int, position: int) -> int:
        return self.table[(req_id, position)]

    def expected_tokens_batch(
        self,
        *,
        req_ids: Sequence[int],
        positions: Sequence[int],
    ) -> List[int]:
        if len(req_ids) != len(positions):
            raise ValueError(
                f"req_ids ({len(req_ids)}) and positions ({len(positions)}) must have equal length"
            )
        return [
            self.expected_token(req_id=r, position=p)
            for r, p in zip(req_ids, positions)
        ]


_U64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    """Standard splitmix64 finalizer over a single uint64.

    Byte-equal mirror of the kernel-side splitmix64_finalize (also mirrored in
    sglang.jit_kernel.kv_canary.verify_ref._splitmix64_python).
    """
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK

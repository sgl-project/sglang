"""Names of the buffers the Mori path registers and writes."""

from __future__ import annotations

import enum
from typing import Iterable, List, Sequence, Tuple

import msgspec
import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.transport.base import BufferKey


class BufferKind(enum.IntEnum):
    KV = 0
    AUX = 1
    STATE = 2


class MoriBufferKey(BufferKey):
    """(kind, index, sub): sub is the per-state-type tensor index for STATE."""

    def __init__(self, kind: BufferKind, index: int, sub: int = 0):
        self._value = (kind, index, sub)
        self._hash = hash(self._value)

    @property
    def kind(self) -> BufferKind:
        return self._value[0]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MoriBufferKey) and self._value == other._value

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        kind, index, sub = self._value
        return f"MoriBufferKey({kind.name}, {index}, {sub})"


class BufferKeys(msgspec.Struct, frozen=True):
    """Keys of one side's buffers, laid out as its registration message lists
    them; built once when the side registers or is imported."""

    kv: Tuple[MoriBufferKey, ...]
    aux: Tuple[MoriBufferKey, ...]
    # One tuple per state type, parallel to kv_args.state_types.
    state: Tuple[Tuple[MoriBufferKey, ...], ...]

    @classmethod
    def for_buffers(
        cls,
        kv_data_ptrs: Sequence[int],
        aux_data_ptrs: Sequence[int],
        state_data_ptrs: Sequence[Sequence[int]],
    ) -> BufferKeys:
        return cls(
            kv=tuple(MoriBufferKey(BufferKind.KV, i) for i in range(len(kv_data_ptrs))),
            aux=tuple(
                MoriBufferKey(BufferKind.AUX, i) for i in range(len(aux_data_ptrs))
            ),
            state=tuple(
                tuple(MoriBufferKey(BufferKind.STATE, i, j) for j in range(len(ptrs)))
                for i, ptrs in enumerate(state_data_ptrs)
            ),
        )

    def pair(
        self,
        kv_data_ptrs: Sequence[int],
        aux_data_ptrs: Sequence[int],
        state_data_ptrs: Sequence[Sequence[int]],
    ) -> List[Tuple[MoriBufferKey, int]]:
        """(key, value) for every buffer, given values in the same layout."""
        pairs = list(zip(self.kv, kv_data_ptrs))
        pairs += zip(self.aux, aux_data_ptrs)
        for keys, values in zip(self.state, state_data_ptrs):
            pairs += zip(keys, values)
        return pairs


def copy_ranges(blocks: Iterable[Tuple[int, int, int]]) -> npt.NDArray[np.uint64]:
    """(N, 3) rows of (src_offset, dst_offset, length) for a Copy."""
    return np.asarray(list(blocks), dtype=np.uint64).reshape(-1, 3)

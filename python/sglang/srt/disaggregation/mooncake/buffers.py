"""Names of the buffers the Mooncake path registers and writes."""

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
    # Staging rings and DCP pack buffers registered through register_fn.
    STAGING = 3


class MooncakeBufferKey(BufferKey):
    """(kind, index, sub): sub is the per-state-type tensor index for STATE."""

    def __init__(self, kind: BufferKind, index: int, sub: int = 0):
        self._value = (kind, index, sub)
        self._hash = hash(self._value)

    @property
    def kind(self) -> BufferKind:
        return self._value[0]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MooncakeBufferKey) and self._value == other._value

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        kind, index, sub = self._value
        return f"MooncakeBufferKey({kind.name}, {index}, {sub})"


class BufferKeys(msgspec.Struct, frozen=True):
    """Keys of one side's buffers, laid out as its registration message lists
    them; built once when the side registers or is imported."""

    kv: Tuple[MooncakeBufferKey, ...]
    aux: Tuple[MooncakeBufferKey, ...]
    # One tuple per state type, parallel to kv_args.state_types.
    state: Tuple[Tuple[MooncakeBufferKey, ...], ...]

    @classmethod
    def for_buffers(
        cls,
        kv_data_ptrs: Sequence[int],
        aux_data_ptrs: Sequence[int],
        state_data_ptrs: Sequence[Sequence[int]],
    ) -> BufferKeys:
        return cls(
            kv=tuple(
                MooncakeBufferKey(BufferKind.KV, i) for i in range(len(kv_data_ptrs))
            ),
            aux=tuple(
                MooncakeBufferKey(BufferKind.AUX, i) for i in range(len(aux_data_ptrs))
            ),
            state=tuple(
                tuple(
                    MooncakeBufferKey(BufferKind.STATE, i, j) for j in range(len(ptrs))
                )
                for i, ptrs in enumerate(state_data_ptrs)
            ),
        )

    def pair(
        self,
        kv_data_ptrs: Sequence[int],
        aux_data_ptrs: Sequence[int],
        state_data_ptrs: Sequence[Sequence[int]],
    ) -> List[Tuple[MooncakeBufferKey, int]]:
        """(key, value) for every buffer, given values in the same layout."""
        pairs = list(zip(self.kv, kv_data_ptrs))
        pairs += zip(self.aux, aux_data_ptrs)
        for keys, values in zip(self.state, state_data_ptrs):
            pairs += zip(keys, values)
        return pairs


def copy_ranges(blocks: Iterable[Tuple[int, int, int]]) -> npt.NDArray[np.uint64]:
    """(N, 3) rows of (src_offset, dst_offset, length) for a Copy."""
    return np.asarray(list(blocks), dtype=np.uint64).reshape(-1, 3)


def group_units(src_groups, dst_groups) -> npt.NDArray[np.uint64]:
    """Return (first src slot, first dst slot, slot count) for each paired group.
    Multiplying by an entry's item length gives Copy ranges, allowing entries
    to share one traversal of the contiguous groups."""
    return np.array(
        [(src[0], dst[0], len(src)) for src, dst in zip(src_groups, dst_groups)],
        dtype=np.uint64,
    ).reshape(-1, 3)


# A decode registers a single staging ring.
PEER_STAGING_KEY = MooncakeBufferKey(BufferKind.STAGING, 0)

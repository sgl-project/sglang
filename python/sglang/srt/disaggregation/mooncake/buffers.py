"""Names of the buffers the Mooncake path registers and writes."""

from __future__ import annotations

import enum
from typing import Callable, Iterable, List, Tuple

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
        self._value = (BufferKind(kind), index, sub)

    @property
    def kind(self) -> BufferKind:
        return self._value[0]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MooncakeBufferKey) and self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        kind, index, sub = self._value
        return f"MooncakeBufferKey({kind.name}, {index}, {sub})"


def kv_args_buffers(
    kv_data_ptrs: List[int],
    aux_data_ptrs: List[int],
    state_data_ptrs: List[List[int]],
) -> List[Tuple[MooncakeBufferKey, int]]:
    """Key every buffer a registration message lists, in the order it lists them."""
    entries = [
        (MooncakeBufferKey(BufferKind.KV, i), p) for i, p in enumerate(kv_data_ptrs)
    ]
    entries += [
        (MooncakeBufferKey(BufferKind.AUX, i), p) for i, p in enumerate(aux_data_ptrs)
    ]
    for i, ptrs in enumerate(state_data_ptrs):
        entries += [
            (MooncakeBufferKey(BufferKind.STATE, i, j), p) for j, p in enumerate(ptrs)
        ]
    return entries


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


def kv_key(index: int) -> MooncakeBufferKey:
    return MooncakeBufferKey(BufferKind.KV, index)


def aux_key(index: int) -> MooncakeBufferKey:
    return MooncakeBufferKey(BufferKind.AUX, index)


def state_keys(state_index: int) -> Callable[[int], MooncakeBufferKey]:
    """Key of each tensor of the state type at `state_index` in kv_args.state_types."""
    return lambda index: MooncakeBufferKey(BufferKind.STATE, state_index, index)


# A decode registers a single staging ring.
PEER_STAGING_KEY = MooncakeBufferKey(BufferKind.STAGING, 0)

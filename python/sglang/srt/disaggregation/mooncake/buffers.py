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


def group_contiguous_units(
    src_indices: npt.NDArray[np.integer], dst_indices: npt.NDArray[np.integer]
) -> npt.NDArray[np.uint64]:
    """Return (first src slot, first dst slot, slot count) for each paired run."""
    size = src_indices.size
    if size == 0 or dst_indices.size == 0:
        return np.empty((0, 3), dtype=np.uint64)
    if size != dst_indices.size:
        raise ValueError(
            "group_concurrent_contiguous requires equal-length src/dst index arrays, "
            f"got {size} and {dst_indices.size}"
        )

    if size > 1:
        # Detect boundaries in the original dtype, as in group_concurrent_contiguous.
        cuts = (
            np.flatnonzero((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))
            + 1
        )
        if cuts.size:
            bounds = np.empty(cuts.size + 2, dtype=np.intp)
            bounds[0], bounds[1:-1], bounds[-1] = 0, cuts, size
            starts = bounds[:-1]
            src_starts, dst_starts = src_indices[starts], dst_indices[starts]
            lengths = np.diff(bounds)
            # Keep the existing Python-to-uint64 conversion behavior for
            # negative offsets, including rejection with NumPy 2.
            if src_starts.min() < 0 or dst_starts.min() < 0:
                rows = zip(src_starts.tolist(), dst_starts.tolist(), lengths.tolist())
                return np.array(list(rows), dtype=np.uint64)
            units = np.empty((starts.size, 3), dtype=np.uint64)
            units[:, 0] = src_starts
            units[:, 1] = dst_starts
            units[:, 2] = lengths
            return units
    return np.array([(int(src_indices[0]), int(dst_indices[0]), size)], dtype=np.uint64)


# A decode registers a single staging ring.
PEER_STAGING_KEY = MooncakeBufferKey(BufferKind.STAGING, 0)

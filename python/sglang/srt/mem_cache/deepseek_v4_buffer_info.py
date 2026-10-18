"""Native DSV4 page regions, without host names, budgets or logical indices."""

from __future__ import annotations

from typing import Literal, Sequence

import msgspec
import torch


class DSV4PagedBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer storage views whose first axis addresses physical pages.

    ``page_slots`` counts native compressed slots, not tree tokens. ``layout``
    identifies the format a transfer consumer must support. Views retain the
    original dtype and storage; constructing a description never copies KV.
    """

    buffers: tuple[torch.Tensor, ...]
    page_bytes: int
    page_slots: int
    layout: Literal[
        "v4",
        "v41",
        "v41_fp4",
        "uniform_fp8",
        "unified",
        "index_fused",
        "index_fp4",
        "pa_nd",
    ]

    @classmethod
    def from_buffers(cls, buffers: Sequence[torch.Tensor], *, page_slots: int, layout):
        return cls(
            buffers=tuple(buffers),
            page_bytes=buffers[0][0].numel() * buffers[0].element_size(),
            page_slots=page_slots,
            layout=layout,
        )


class DSV4CompressedBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    kv: DSV4PagedBufferInfo
    rope: DSV4PagedBufferInfo | None = None


class DSV4IndexerBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    key: DSV4PagedBufferInfo
    scale: DSV4PagedBufferInfo | None = None


class DSV4StateBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Byte pages of persistent compressor rings, with their native element type.

    Each page holds one complete ring. Request-private state has no checkpoint
    description: it is rebuilt when a cached prefix is restored.
    """

    pages: torch.Tensor
    ring_size: int
    dtype: torch.dtype


class DSV4CheckpointGroupInfo(msgspec.Struct, frozen=True, kw_only=True):
    attention: tuple[DSV4StateBufferInfo, ...]
    indexer: tuple[DSV4StateBufferInfo, ...]


class DSV4CompressedGroupInfo(msgspec.Struct, frozen=True, kw_only=True):
    """One physical store per owner, in buffer-axis order.

    C1/C2 readers share their owner's store and do not add duplicate buffers.
    Checkpoint tuples, when present, follow the same model-layer order.
    """

    ratio: Literal[1, 2, 4, 128]
    layer_ids: tuple[int, ...]
    kv: DSV4CompressedBufferInfo
    indexer: DSV4IndexerBufferInfo | None
    checkpoint: DSV4CheckpointGroupInfo | None


class DSV4RootBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    layer_ids: tuple[int, ...]
    swa: DSV4PagedBufferInfo | None
    compressed: tuple[DSV4CompressedGroupInfo, ...]

    def compressed_group(self, ratio: int) -> DSV4CompressedGroupInfo | None:
        return next((group for group in self.compressed if group.ratio == ratio), None)

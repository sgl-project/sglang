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
        "v41_new",
        "v41_fp4_new",
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


class DSV4CheckpointBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Persistent compressor rings of one region, one entry per owner layer."""

    states: tuple[DSV4StateBufferInfo, ...]

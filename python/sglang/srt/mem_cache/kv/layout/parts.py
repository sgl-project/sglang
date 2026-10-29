"""Typed parts in an entry, with dtype alignment and no vector-row restriction."""

import math
from typing import Sequence

import torch

from sglang.srt.mem_cache.kv.decl import TensorPartDecl


class TensorEntryLayout:
    def __init__(self, parts: Sequence[TensorPartDecl]):
        self.parts = tuple(parts)
        if not parts or len({p.name for p in parts}) != len(parts):
            raise ValueError("an entry requires distinct named parts")
        self.offsets = {}
        offset = 0
        for part in self.parts:
            alignment = part.dtype.itemsize
            offset = -(-offset // alignment) * alignment
            self.offsets[part.name] = offset
            offset += part.row_bytes
        alignment = max(p.dtype.itemsize for p in self.parts)
        self.entry_bytes = -(-offset // alignment) * alignment

    def views(self, raw: torch.Tensor, rows: int, *, entry_bytes: int | None = None):
        stride_bytes = self.entry_bytes if entry_bytes is None else entry_bytes
        if stride_bytes < self.entry_bytes or rows * stride_bytes > raw.numel():
            raise ValueError("entry parts exceed their backing")
        views = {}
        for part in self.parts:
            itemsize = part.dtype.itemsize
            if stride_bytes % itemsize:
                raise ValueError(f"entry stride is not aligned for {part.name!r}")
            inner_strides = tuple(
                math.prod(part.shape[i + 1 :]) for i in range(len(part.shape))
            )
            strides = (stride_bytes // itemsize, *inner_strides)
            if part.layer_ids:
                strides = (math.prod(part.shape), *strides)
            views[part.name] = torch.as_strided(
                raw[: raw.numel() // itemsize * itemsize].view(part.dtype),
                size=part.tensor_shape(rows),
                stride=strides,
                storage_offset=self.offsets[part.name] // itemsize,
            )
        return views

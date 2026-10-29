"""Typed entry parts with physical row alignment supplied by their consumers."""

import math
from typing import Mapping, Sequence

import torch

from sglang.srt.mem_cache.kv.decl import TensorPartDecl


class TensorEntryLayout:
    def __init__(
        self,
        parts: Sequence[TensorPartDecl],
        *,
        row_alignments: Mapping[str, int] | None = None,
    ):
        self.parts = tuple(parts)
        if not parts or len({p.name for p in parts}) != len(parts):
            raise ValueError("an entry requires distinct named parts")
        row_alignments = {} if row_alignments is None else row_alignments
        if set(row_alignments) - {p.name for p in parts} or any(
            alignment <= 0 for alignment in row_alignments.values()
        ):
            raise ValueError("row alignments must be positive and name entry parts")
        self.offsets = {}
        self.layer_strides = {}
        self.alignment = math.lcm(
            *(p.dtype.itemsize for p in parts), *row_alignments.values()
        )
        offset = 0
        for part in self.parts:
            alignment = math.lcm(part.dtype.itemsize, row_alignments.get(part.name, 1))
            offset = -(-offset // alignment) * alignment
            self.offsets[part.name] = offset
            row_bytes = math.prod(part.shape) * part.dtype.itemsize
            layer_stride = -(-row_bytes // alignment) * alignment
            self.layer_strides[part.name] = layer_stride
            offset += (
                layer_stride * len(part.layer_ids) if part.layer_ids else row_bytes
            )
        self.entry_bytes = -(-offset // self.alignment) * self.alignment

    def views(
        self,
        raw: torch.Tensor,
        rows: int,
        *,
        entry_bytes: int | None = None,
        offset_bytes: int = 0,
    ):
        stride_bytes = self.entry_bytes if entry_bytes is None else entry_bytes
        end = offset_bytes + max(0, rows - 1) * stride_bytes + self.entry_bytes
        if stride_bytes < self.entry_bytes or end > raw.numel():
            raise ValueError("entry parts exceed their backing")
        if (
            stride_bytes % self.alignment
            or (raw.data_ptr() + offset_bytes) % self.alignment
        ):
            raise ValueError("entry stride or base violates its row alignment")
        views = {}
        for part in self.parts:
            itemsize = part.dtype.itemsize
            if stride_bytes % itemsize or offset_bytes % itemsize:
                raise ValueError(f"entry stride is not aligned for {part.name!r}")
            inner_strides = tuple(
                math.prod(part.shape[i + 1 :]) for i in range(len(part.shape))
            )
            strides = (stride_bytes // itemsize, *inner_strides)
            if part.layer_ids:
                strides = (self.layer_strides[part.name] // itemsize, *strides)
            typed = raw[: raw.numel() // itemsize * itemsize].view(part.dtype)
            views[part.name] = torch.as_strided(
                typed,
                size=part.tensor_shape(rows),
                stride=strides,
                storage_offset=typed.storage_offset()
                + (offset_bytes + self.offsets[part.name]) // itemsize,
            )
        return views

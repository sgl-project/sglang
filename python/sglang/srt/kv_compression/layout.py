"""Qwen3 MHA/GQA, one token per native page, all layers per object."""

from __future__ import annotations

import json

import torch
from sglang.srt.kv_compression.types import FORMAT_VERSION


class KVLayoutAdapter:
    def __init__(self, k_buffers, v_buffers, page_size=1):
        self.buffers = tuple(k_buffers) + tuple(v_buffers)
        if (
            page_size != 1
            or not k_buffers
            or len(k_buffers) != len(v_buffers)
            or any(t.dtype != torch.bfloat16 for t in self.buffers)
            or any(t.shape[1:] != k_buffers[0].shape[1:] for t in self.buffers)
        ):
            raise ValueError("Compression v2 requires BF16 MHA/GQA with page_size=1")
        self.device = k_buffers[0].device
        self.slot_bytes = k_buffers[0][0].numel() * k_buffers[0].element_size()
        self.page_bytes = self.slot_bytes * len(self.buffers)
        self.tag = json.dumps(
            dict(
                version=FORMAT_VERSION,
                order="page,K-layers,V-layers",
                page_size=1,
                shape=list(k_buffers[0].shape[1:]),
                layers=len(k_buffers),
                dtype="bf16",
            ),
            sort_keys=True,
        )

    def pack_pages(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        return torch.cat(
            [
                t.index_select(0, indices).view(torch.uint8).reshape(len(indices), -1)
                for t in self.buffers
            ],
            dim=1,
        )

    def unpack_pages(self, raw, indices):
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        raw = raw.reshape(len(indices), self.page_bytes)
        for slot, tensor in enumerate(self.buffers):
            value = raw[:, slot * self.slot_bytes : (slot + 1) * self.slot_bytes]
            value = (
                value.contiguous()
                .view(tensor.dtype)
                .reshape(len(indices), *tensor.shape[1:])
            )
            tensor.index_copy_(0, indices, value)

    def to_staging_order(self, raw):
        """Existing Decode scatter consumes slot-major, not page-major bytes."""
        return (
            raw.reshape(-1, len(self.buffers), self.slot_bytes)
            .transpose(0, 1)
            .contiguous()
            .reshape(-1)
        )

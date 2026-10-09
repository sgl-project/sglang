# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Packed recurrent KV for the opt-in FlashLoop Ouro model."""

import torch

from sglang.srt.mem_cache.flashloop_quantization import QuantizedStorage
from sglang.srt.mem_cache.memory_pool import KVCache


class FlashLoopKVPool(KVCache):
    def __init__(
        self,
        size,
        page_size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        device,
        enable_memory_saver=False,
        v_head_dim=None,
        start_layer=None,
        end_layer=None,
        enable_alt_stream=True,
        enable_kv_cache_copy=False,
        max_requests=None,
        post_capture_active=False,
        **kwargs,
    ):
        if (
            page_size != 64
            or layer_num % 4
            or dtype != torch.bfloat16
            or enable_kv_cache_copy
            or enable_memory_saver
        ):
            raise ValueError(
                "INT4 recurrence pool requires BF16 compute, page64, four loops and no KV-copy/offload"
            )
        if v_head_dim not in (None, head_dim) or start_layer not in (None, 0):
            raise ValueError(
                "Only equal K/V head dimensions and single-rank execution are supported"
            )
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.capacity, self.head_num, self.head_dim = (
            size + page_size,
            head_num,
            head_dim,
        )
        self.physical_layers = layer_num // 4
        if max_requests is None or max_requests < 1 or post_capture_active:
            raise ValueError(
                "FlashLoop requires a bounded request count and ordinary KV allocation"
            )
        requests = max_requests + 1
        self.storage = [
            QuantizedStorage(self.capacity, head_num, head_dim, requests, device)
            for _ in range(self.physical_layers)
        ]
        # Native extend attends to its input K/V when prefix length is zero.
        # This single row supplies shape/stride metadata, not a BF16 cache.
        self.dummy = torch.zeros((1, head_num, head_dim), device=device, dtype=dtype)
        self._finalize_allocation_log(size)

    def get_key_buffer(self, layer_id):
        return self.dummy

    def get_value_buffer(self, layer_id):
        return self.dummy

    def get_kv_buffer(self, layer_id):
        return self.dummy, self.dummy

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("Packed recurrent KV must be written with request positions")

    def views(self, layer_id):
        loop, physical = divmod(layer_id, self.physical_layers)
        storage = self.storage[physical]
        return storage, loop, storage.view(loop, True), storage.view(loop, False)

    def get_kv_size_bytes(self):
        return (
            sum(s.nbytes() for s in self.storage)
            + self.dummy.numel() * self.dummy.element_size()
        )


def bytes_per_token(heads, dims, layers):
    # Two KV streams, 4-bit codes, two FP16 values per group, one row flag.
    return heads * dims * layers * 2 * 9 // 16 + layers * 2


def fixed_bytes(heads, dims, layers, max_requests):
    return (
        (max_requests + 1) * 128 * heads * dims * layers * 4
        + 64 * bytes_per_token(heads, dims, layers)
        + heads * dims * 2
    )

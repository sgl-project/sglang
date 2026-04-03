"""
INT8 KV cache memory pool for MHA models.

Phase-1 constraints:
- MHA only (no MLA)
- triton attention backend only
- uses per-token per-head asymmetric quantization:
    zp    = (min + max) / 2
    scale = (max - min) / 255
    q     = clamp(round((x - zp) / scale), -128, 127)
    x_hat = q * scale + zp
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.int8_kv_kernels import (
    dequant_int8_kv,
    gather_dequant_kv_from_pool,
    scatter_quant_kv_to_pool,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


class INT8MHATokenToKVPool(MHATokenToKVPool):
    """INT8 storage + FP16 scale/zp metadata KV pool."""

    scale_dtype = torch.float16

    @staticmethod
    def get_bytes_per_token(
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        scale_dtype: torch.dtype = torch.float16,
        v_head_dim: Optional[int] = None,
    ) -> int:
        if v_head_dim is None:
            v_head_dim = head_dim
        scale_bytes = torch._utils._element_size(scale_dtype)
        data_bytes = num_kv_heads * (head_dim + v_head_dim) * num_layers
        meta_bytes = num_kv_heads * num_layers * 4 * scale_bytes
        return int(data_bytes + meta_bytes)

    @staticmethod
    def get_baseline_bytes_per_token(
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        kv_dtype: torch.dtype,
        v_head_dim: Optional[int] = None,
    ) -> int:
        if v_head_dim is None:
            v_head_dim = head_dim
        return int(
            num_kv_heads
            * (head_dim + v_head_dim)
            * num_layers
            * torch._utils._element_size(kv_dtype)
        )

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                f"INT8 KV cache expects fp16/bf16/fp32 compute dtype, got {dtype}."
            )
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )
        self.int8_kv_cache_enabled = True
        self.kv_capacity_factor = (
            self.get_baseline_bytes_per_token(
                self.head_num,
                self.head_dim,
                self.layer_num,
                self.dtype,
                v_head_dim=self.v_head_dim,
            )
            / self.get_bytes_per_token(
                self.head_num,
                self.head_dim,
                self.layer_num,
                v_head_dim=self.v_head_dim,
            )
        )
        self._write_version = [0 for _ in range(self.layer_num)]
        self._last_dequant_layer = -1
        self._last_dequant_version = -1
        self._last_k_cache = None
        self._last_v_cache = None

        logger.info(
            "[INT8 KV] Initialized with estimated token-capacity factor %.2fx",
            self.kv_capacity_factor,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                k_shape = (self.size + self.page_size, self.head_num, self.head_dim)
                v_shape = (self.size + self.page_size, self.head_num, self.v_head_dim)
                scale_shape = (self.size + self.page_size, self.head_num, 1)

                self.k_buffer = [
                    torch.zeros(k_shape, dtype=torch.int8, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(v_shape, dtype=torch.int8, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.k_scale_buffer = [
                    torch.ones(scale_shape, dtype=self.scale_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.k_zp_buffer = [
                    torch.zeros(scale_shape, dtype=self.scale_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.ones(scale_shape, dtype=self.scale_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_zp_buffer = [
                    torch.zeros(scale_shape, dtype=self.scale_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        ptr_groups = [
            self.k_buffer,
            self.v_buffer,
            self.k_scale_buffer,
            self.k_zp_buffer,
            self.v_scale_buffer,
            self.v_zp_buffer,
        ]
        self.data_ptrs = torch.cat(
            [
                torch.tensor(
                    [x.data_ptr() for x in group],
                    dtype=torch.uint64,
                    device=self.device,
                )
                for group in ptr_groups
            ],
            dim=0,
        )
        self.data_strides = torch.tensor(
            [np.prod(x.shape[1:]) * x.dtype.itemsize for group in ptr_groups for x in group],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.k_zp_buffer
        del self.v_scale_buffer
        del self.v_zp_buffer

    def get_kv_size_bytes(self):
        k_size_bytes = 0
        v_size_bytes = 0
        for i in range(self.layer_num):
            k_size_bytes += (
                np.prod(self.k_buffer[i].shape) * self.k_buffer[i].dtype.itemsize
                + np.prod(self.k_scale_buffer[i].shape)
                * self.k_scale_buffer[i].dtype.itemsize
                + np.prod(self.k_zp_buffer[i].shape) * self.k_zp_buffer[i].dtype.itemsize
            )
            v_size_bytes += (
                np.prod(self.v_buffer[i].shape) * self.v_buffer[i].dtype.itemsize
                + np.prod(self.v_scale_buffer[i].shape)
                * self.v_scale_buffer[i].dtype.itemsize
                + np.prod(self.v_zp_buffer[i].shape) * self.v_zp_buffer[i].dtype.itemsize
            )
        return k_size_bytes, v_size_bytes

    def gather_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        out_k: Optional[torch.Tensor] = None,
        out_v: Optional[torch.Tensor] = None,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        layer_idx = layer_id - self.start_layer
        k = gather_dequant_kv_from_pool(
            loc=loc,
            q_pool=self.k_buffer[layer_idx],
            scale_pool=self.k_scale_buffer[layer_idx],
            zp_pool=self.k_zp_buffer[layer_idx],
            out=out_k,
            out_dtype=self.dtype,
        )
        v = gather_dequant_kv_from_pool(
            loc=loc,
            q_pool=self.v_buffer[layer_idx],
            scale_pool=self.v_scale_buffer[layer_idx],
            zp_pool=self.v_zp_buffer[layer_idx],
            out=out_v,
            out_dtype=self.dtype,
        )
        return k, v

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "INT8 KV cache does not yet support contiguous export for disaggregation."
        )

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cache_cpu[-1].append(
                    [
                        self.k_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        self.v_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        self.k_scale_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        self.k_zp_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        self.v_scale_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                        self.v_zp_buffer[layer_id][chunk_indices].to(
                            "cpu", non_blocking=True
                        ),
                    ]
                )
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_i8, v_i8, k_s, k_z, v_s, v_z = kv_cache_cpu[layer_id][
                    i // chunk_size
                ]
                self.k_buffer[layer_id][chunk_indices] = k_i8.to(
                    self.device, non_blocking=True
                )
                self.v_buffer[layer_id][chunk_indices] = v_i8.to(
                    self.device, non_blocking=True
                )
                self.k_scale_buffer[layer_id][chunk_indices] = k_s.to(
                    self.device, non_blocking=True
                )
                self.k_zp_buffer[layer_id][chunk_indices] = k_z.to(
                    self.device, non_blocking=True
                )
                self.v_scale_buffer[layer_id][chunk_indices] = v_s.to(
                    self.device, non_blocking=True
                )
                self.v_zp_buffer[layer_id][chunk_indices] = v_z.to(
                    self.device, non_blocking=True
                )
        torch.cuda.synchronize()

    def _dequantize_layer(self, layer_idx: int):
        k = dequant_int8_kv(
            self.k_buffer[layer_idx],
            self.k_scale_buffer[layer_idx],
            self.k_zp_buffer[layer_idx],
            out_dtype=self.dtype,
        )
        v = dequant_int8_kv(
            self.v_buffer[layer_idx],
            self.v_scale_buffer[layer_idx],
            self.v_zp_buffer[layer_idx],
            out_dtype=self.dtype,
        )
        return k, v

    def _get_kv_buffer_cached(self, layer_id: int):
        layer_idx = layer_id - self.start_layer
        version = self._write_version[layer_idx]
        if (
            self._last_dequant_layer == layer_idx
            and self._last_dequant_version == version
            and self._last_k_cache is not None
            and self._last_v_cache is not None
        ):
            return self._last_k_cache, self._last_v_cache

        k, v = self._dequantize_layer(layer_idx)
        self._last_dequant_layer = layer_idx
        self._last_dequant_version = version
        self._last_k_cache = k
        self._last_v_cache = v
        return k, v

    def _get_key_buffer(self, layer_id: int):
        return self._get_kv_buffer_cached(layer_id)[0]

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        return self._get_kv_buffer_cached(layer_id)[1]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_kv_buffer_cached(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        layer_idx = layer_id - self.start_layer
        scatter_quant_kv_to_pool(
            cache_k=cache_k,
            cache_v=cache_v,
            loc=loc,
            k_int8_pool=self.k_buffer[layer_idx],
            v_int8_pool=self.v_buffer[layer_idx],
            k_scale_pool=self.k_scale_buffer[layer_idx],
            k_zp_pool=self.k_zp_buffer[layer_idx],
            v_scale_pool=self.v_scale_buffer[layer_idx],
            v_zp_pool=self.v_zp_buffer[layer_idx],
        )

        self._write_version[layer_idx] += 1
        if self._last_dequant_layer == layer_idx:
            self._last_dequant_version = -1
            self._last_k_cache = None
            self._last_v_cache = None

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        super().move_kv_cache(tgt_loc, src_loc)
        self._last_dequant_layer = -1
        self._last_dequant_version = -1
        self._last_k_cache = None
        self._last_v_cache = None
        for i in range(self.layer_num):
            self._write_version[i] += 1

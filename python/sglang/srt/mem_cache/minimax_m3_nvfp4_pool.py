# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    KVCache,
    MiniMaxSparseKVPool,
    UnquantizedKVCacheMethod,
    maybe_detect_oob,
    unwrap_write_loc,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo

logger = logging.getLogger(__name__)

MSA_V1_PAGE_SIZE = 128
_SCALE_GROUP_SIZE = 16
# Component offsets and row widths stay 128-byte multiples: MSA kernels need
# 16/32-byte aligned cache views and the HiCache page-row copy moves 128-byte lines.
_PAGE_ROW_ALIGN = 128


def _main_page_components(head_num: int, head_dim: int, page_size: int):
    payload = (head_num, page_size, head_dim // 2)
    scale = (head_num, page_size, head_dim // _SCALE_GROUP_SIZE)
    return (
        ("k_buffer", payload),
        ("k_scale_buffer", scale),
        ("v_buffer", payload),
        ("v_scale_buffer", scale),
    )


def _index_page_components(head_dim: int, page_size: int):
    return (
        ("k_buffer", (page_size, head_dim // 2)),
        ("k_scale_buffer", (page_size, head_dim // _SCALE_GROUP_SIZE)),
    )


def _page_row_layout(components) -> tuple[list[int], int]:
    """Byte offset of each ``(name, per-page byte shape)`` component inside one
    page row, and the row width."""
    offsets = []
    offset = 0
    for _, shape in components:
        offsets.append(offset)
        offset += -(-math.prod(shape) // _PAGE_ROW_ALIGN) * _PAGE_ROW_ALIGN
    return offsets, offset


def msa_v1_page_row_bytes(
    kv_heads: int, head_dim: int, index_head_dim: int, page_size: int
) -> tuple[int, int]:
    """Main and index row widths of one sparse layer's page."""
    return (
        _page_row_layout(_main_page_components(kv_heads, head_dim, page_size))[1],
        _page_row_layout(_index_page_components(index_head_dim, page_size))[1],
    )


def _allocate_page_rows(pool: KVCache, components, num_pages: int, device: str):
    """One ``[layers, pages, row_bytes]`` slab and per-layer component views.

    A view keeps the exact per-page shape the kernels consume and differs from a
    dedicated tensor only in its page stride, which is the row width.
    """
    offsets, row_bytes = _page_row_layout(components)
    with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(pool.custom_mem_pool)
            if pool.enable_custom_mem_pool
            else nullcontext()
        ):
            slab = torch.zeros(
                (pool.layer_num, num_pages, row_bytes), dtype=torch.uint8, device=device
            )
    rows = list(slab.unbind(0))
    views = {
        name: [
            row[:, offset : offset + math.prod(shape)].view(num_pages, *shape)
            for row in rows
        ]
        for (name, shape), offset in zip(components, offsets)
    }
    return slab, rows, row_bytes, views


def _page_row_device_pool_info(pool, pool_name) -> DevicePoolInfo:
    from sglang.srt.mem_cache.device_pool_info import (
        DevicePoolInfo,
        PageRowBufferInfo,
    )
    from sglang.srt.mem_cache.hicache_storage import PoolName

    return DevicePoolInfo(
        pool_name=pool_name,
        indices_from_pool=PoolName.KV,
        layer_ids=tuple(range(pool.start_layer, pool.start_layer + pool.layer_num)),
        buffer_info=PageRowBufferInfo(
            slab=pool.slab,
            rows=tuple(pool.rows),
            row_bytes=pool.row_bytes,
            page_size=pool.page_size,
            num_pages=pool.num_pages,
        ),
    )


class MiniMaxM3Nvfp4IndexKPool(KVCache):
    """Paged NVFP4 K-only cache consumed by a MiniMax sparse indexer.

    Each (layer, page) is one row holding ``[index K, index K scale]``.
    """

    PAYLOAD_BYTES_PER_TOKEN = 64
    SCALE_BYTES_PER_TOKEN = 8
    INDEX_HEAD_DIM = 128
    SCALE_GROUP_SIZE = _SCALE_GROUP_SIZE

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
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        msa_name = "MSA-v1"
        if page_size != MSA_V1_PAGE_SIZE:
            raise ValueError(
                f"{msa_name} NVFP4 index cache requires page_size=128, got {page_size}."
            )
        if head_num != 1 or head_dim != self.INDEX_HEAD_DIM:
            raise ValueError(
                f"{msa_name} NVFP4 index cache requires one 128-dimensional K head, "
                f"got head_num={head_num}, head_dim={head_dim}."
            )

        super().__init__(
            size,
            page_size,
            torch.uint8,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            allocation_label=f"MiniMax {msa_name} index",
        )
        self.source_dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.is_minimax_m3_nvfp4_index = True

        total_slots = size + page_size
        self.num_pages = (total_slots + page_size - 1) // page_size
        self.total_slots = self.num_pages * page_size
        self.slab, self.rows, self.row_bytes, views = _allocate_page_rows(
            self,
            _index_page_components(self.INDEX_HEAD_DIM, page_size),
            self.num_pages,
            device,
        )
        self.k_buffer = views["k_buffer"]
        self.k_scale_buffer = views["k_scale_buffer"]
        self.global_scale = torch.ones(1, dtype=torch.float32, device=device)
        self._finalize_allocation_log(size)

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def _wait_for_layer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def _page_row(self, loc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = loc.reshape(-1).long()
        return loc // self.page_size, loc % self.page_size

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.k_buffer[self._layer_index(layer_id)]

    def get_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.k_scale_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("MiniMaxM3Nvfp4IndexKPool does not allocate V")

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("MiniMaxM3Nvfp4IndexKPool does not allocate V")

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "Use MiniMaxSparseKVPool.set_index_k_buffer for the NVFP4 index cache"
        )

    def set_quantized_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        packed_k: torch.Tensor,
        block_scales: torch.Tensor,
    ) -> None:
        packed_k = packed_k.reshape(-1, self.PAYLOAD_BYTES_PER_TOKEN).view(torch.uint8)
        block_scales = block_scales.reshape(-1, self.SCALE_BYTES_PER_TOKEN).view(
            torch.uint8
        )
        if packed_k.shape[0] != loc.numel() or block_scales.shape[0] != loc.numel():
            raise ValueError(
                "MiniMax NVFP4 index cache write expects one packed K and scale row per location."
            )

        page, row = self._page_row(loc)
        layer_idx = self._layer_index(layer_id)
        self.k_buffer[layer_idx][page, row] = packed_k
        self.k_scale_buffer[layer_idx][page, row] = block_scales

    def set_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if cache_k.shape[-1] != self.INDEX_HEAD_DIM:
            raise ValueError(
                f"MSA-v1 index K must have dim {self.INDEX_HEAD_DIM}, "
                f"got shape={tuple(cache_k.shape)}."
            )
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        cache_k = cache_k.reshape(-1, 1, self.INDEX_HEAD_DIM).contiguous()
        packed_k, block_scales, _ = NVFP4KVQuantizeUtil.quantize(
            cache_k, self.global_scale
        )
        self.set_quantized_k_buffer(layer_id, loc, packed_k, block_scales)

    def get_quantized_rows(
        self, layer_id: int, loc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        page, row = self._page_row(loc)
        layer_idx = self._layer_index(layer_id)
        packed_k = self.k_buffer[layer_idx][page, row]
        block_scales = self.k_scale_buffer[layer_idx][page, row]
        return packed_k, block_scales.reshape(-1, self.SCALE_BYTES_PER_TOKEN).view(
            torch.float8_e4m3fn
        )

    def get_raw_k_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed payload and E4M3 block scales as raw uint8, for fused writers.

        Write-side accessor: it does not wait on the layer transfer counter,
        matching `set_quantized_k_buffer`. Readers use `get_key_buffer`.
        """
        layer_idx = self._layer_index(layer_id)
        return self.k_buffer[layer_idx], self.k_scale_buffer[layer_idx]

    def move_slots(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            packed_k, block_scales = self.get_quantized_rows(layer_id, src_loc)
            self.set_quantized_k_buffer(
                layer_id, tgt_loc, packed_k.clone(), block_scales.clone()
            )

    def get_device_pool_infos(self) -> tuple[DevicePoolInfo, ...]:
        from sglang.srt.mem_cache.hicache_storage import PoolName

        return (_page_row_device_pool_info(self, PoolName.INDEXER),)

    def get_kv_size_bytes(self):
        return self.slab.nbytes, 0

    def get_contiguous_buf_infos(self):
        """One packed page row per item, one buffer per layer."""
        return (
            [row.data_ptr() for row in self.rows],
            [row.nbytes for row in self.rows],
            [self.row_bytes] * self.layer_num,
        )

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                packed_k, block_scales = self.get_quantized_rows(
                    layer_id, chunk_indices
                )
                cache_cpu[-1].append(
                    [
                        packed_k.to("cpu", non_blocking=True),
                        block_scales.to("cpu", non_blocking=True),
                    ]
                )
        current_platform.synchronize()
        return cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id, layer_id in enumerate(
            range(self.start_layer, self.start_layer + self.layer_num)
        ):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                packed_cpu, scales_cpu = kv_cache_cpu[local_layer_id][i // chunk_size]
                self.set_quantized_k_buffer(
                    layer_id,
                    chunk_indices,
                    packed_cpu.to(self.device, non_blocking=True),
                    scales_cpu.to(self.device, non_blocking=True),
                )
        current_platform.synchronize()


class MiniMaxM3Nvfp4MainKVPool(KVCache):
    """MSA-v1 paged main K/V cache in the Q8KV4 kernel's native layout.

    Each (layer, page) is one row holding ``[K, K scale, V, V scale]``.
    """

    SCALE_GROUP_SIZE = _SCALE_GROUP_SIZE

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
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        if page_size != MSA_V1_PAGE_SIZE:
            raise ValueError(
                f"MSA-v1 NVFP4 main cache requires page_size=128, got {page_size}."
            )
        if head_dim != 128 or head_dim % self.SCALE_GROUP_SIZE != 0:
            raise ValueError(
                f"MSA-v1 NVFP4 main cache requires head_dim=128, got {head_dim}."
            )
        if head_num <= 0:
            raise ValueError("MSA-v1 NVFP4 main cache requires at least one KV head.")

        super().__init__(
            size,
            page_size,
            torch.uint8,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            allocation_label="MiniMax MSA-v1 main NVFP4",
        )
        self.source_dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.payload_bytes_per_head = head_dim // 2
        self.scale_bytes_per_head = head_dim // self.SCALE_GROUP_SIZE
        self.is_minimax_m3_nvfp4_main = True

        total_slots = size + page_size
        self.num_pages = (total_slots + page_size - 1) // page_size
        self.total_slots = self.num_pages * page_size
        # FP8 scales are stored as bytes because advanced-index assignment is
        # not implemented for every float8 dtype in PyTorch.
        self.slab, self.rows, self.row_bytes, views = _allocate_page_rows(
            self,
            _main_page_components(head_num, head_dim, page_size),
            self.num_pages,
            device,
        )
        self.k_buffer = views["k_buffer"]
        self.v_buffer = views["v_buffer"]
        self.k_scale_buffer = views["k_scale_buffer"]
        self.v_scale_buffer = views["v_scale_buffer"]
        # M3.1 checkpoints use unit outer scales. Keep these device-resident so
        # quantize/dequantize launches do not create scalar tensors in graphs.
        self.global_scale = torch.ones(1, dtype=torch.float32, device=device)
        self.quant_method = UnquantizedKVCacheMethod()
        self._finalize_allocation_log(size)

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def _wait_for_layer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def _page_row(self, loc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = loc.reshape(-1).long()
        return loc // self.page_size, loc % self.page_size

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.k_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.v_buffer[self._layer_index(layer_id)]

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_k_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.k_scale_buffer[self._layer_index(layer_id)].view(
            torch.float8_e4m3fn
        )

    def get_v_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.v_scale_buffer[self._layer_index(layer_id)].view(
            torch.float8_e4m3fn
        )

    def set_quantized_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        packed_k: torch.Tensor,
        packed_v: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
    ) -> None:
        rows = loc.numel()
        packed_shape = (rows, self.head_num, self.payload_bytes_per_head)
        scale_shape = (rows, self.head_num, self.scale_bytes_per_head)
        packed_k = packed_k.reshape(packed_shape).view(torch.uint8)
        packed_v = packed_v.reshape(packed_shape).view(torch.uint8)
        k_scales = k_scales.reshape(scale_shape).view(torch.uint8)
        v_scales = v_scales.reshape(scale_shape).view(torch.uint8)
        page, row = self._page_row(loc)
        layer_idx = self._layer_index(layer_id)
        self.k_buffer[layer_idx][page, :, row, :] = packed_k
        self.v_buffer[layer_idx][page, :, row, :] = packed_v
        self.k_scale_buffer[layer_idx][page, :, row, :] = k_scales
        self.v_scale_buffer[layer_idx][page, :, row, :] = v_scales

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ) -> None:
        if k_scale not in (None, 1.0) or v_scale not in (None, 1.0):
            raise ValueError(
                "MSA-v1 NVFP4 main K/V uses fixed unit outer scales, got "
                f"k_scale={k_scale}, v_scale={v_scale}."
            )
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(
            loc,
            0,
            self.size + self.page_size,
            "set_kv_buffer (MiniMax MSA-v1 NVFP4)",
        )
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        expected_tail = (self.head_num, self.head_dim)
        if tuple(cache_k.shape[1:]) != expected_tail or cache_v.shape != cache_k.shape:
            raise ValueError(
                "MSA-v1 NVFP4 main cache expects equal K/V tensors with shape "
                f"[tokens, {self.head_num}, {self.head_dim}], got "
                f"K={tuple(cache_k.shape)}, V={tuple(cache_v.shape)}."
            )
        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        packed_k, k_scales, _ = NVFP4KVQuantizeUtil.quantize(
            cache_k.contiguous(), self.global_scale
        )
        packed_v, v_scales, _ = NVFP4KVQuantizeUtil.quantize(
            cache_v.contiguous(), self.global_scale
        )
        self.set_quantized_kv_buffer(
            layer_id, loc, packed_k, packed_v, k_scales, v_scales
        )

    def get_quantized_rows(
        self, layer_id: int, loc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        page, row = self._page_row(loc)
        layer_idx = self._layer_index(layer_id)
        return (
            self.k_buffer[layer_idx][page, :, row, :],
            self.v_buffer[layer_idx][page, :, row, :],
            self.k_scale_buffer[layer_idx][page, :, row, :].view(torch.float8_e4m3fn),
            self.v_scale_buffer[layer_idx][page, :, row, :].view(torch.float8_e4m3fn),
        )

    def get_raw_kv_buffers(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packed payloads and E4M3 block scales as raw uint8, for fused writers.

        Write-side accessor: it does not wait on the layer transfer counter,
        matching `set_quantized_kv_buffer`. Readers use `get_kv_buffer`.
        """
        layer_idx = self._layer_index(layer_id)
        return (
            self.k_buffer[layer_idx],
            self.v_buffer[layer_idx],
            self.k_scale_buffer[layer_idx],
            self.v_scale_buffer[layer_idx],
        )

    def move_slots(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            rows = self.get_quantized_rows(layer_id, src_loc)
            self.set_quantized_kv_buffer(
                layer_id, tgt_loc, *(row.clone() for row in rows)
            )

    def get_device_pool_infos(self) -> tuple[DevicePoolInfo, ...]:
        from sglang.srt.mem_cache.hicache_storage import PoolName

        return (_page_row_device_pool_info(self, PoolName.KV),)

    def get_kv_size_bytes(self):
        return self.slab.nbytes, 0

    def get_contiguous_buf_infos(self):
        """One packed page row per item, one buffer per layer."""
        return (
            [row.data_ptr() for row in self.rows],
            [row.nbytes for row in self.rows],
            [self.row_bytes] * self.layer_num,
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        current_platform.synchronize()
        cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                rows = self.get_quantized_rows(layer_id, indices[i : i + chunk_size])
                cache_cpu[-1].append([row.to("cpu", non_blocking=True) for row in rows])
        current_platform.synchronize()
        return cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id, layer_id in enumerate(
            range(self.start_layer, self.start_layer + self.layer_num)
        ):
            for i in range(0, len(indices), chunk_size):
                rows = kv_cache_cpu[local_layer_id][i // chunk_size]
                self.set_quantized_kv_buffer(
                    layer_id,
                    indices[i : i + chunk_size],
                    *(row.to(self.device, non_blocking=True) for row in rows),
                )
        current_platform.synchronize()


class MiniMaxM3Nvfp4KVPool(MiniMaxSparseKVPool):
    """MSA-v1 main K/V and single-head index K, all packed as NVFP4."""

    def __init__(self, **kwargs):
        if kwargs["dense_layer_ids"]:
            raise ValueError("M3.1 NVFP4 requires sparse attention in every layer")
        if set(kwargs["disable_value_sparse_layer_ids"]) != set(
            kwargs["sparse_layer_ids"]
        ):
            raise ValueError("M3.1 NVFP4 requires a K-only sparse index")
        super().__init__(
            **kwargs,
            main_pool_cls=MiniMaxM3Nvfp4MainKVPool,
            index_k_pool_cls=MiniMaxM3Nvfp4IndexKPool,
        )
        # The base fast path is a raw byte copy, which cannot serve a quantizing
        # cache; this pool dispatches to its own fused quantize+store instead.
        self.use_minimax_fused_kv_index_store = False
        # The fused kernel reproduces `flashinfer.nvfp4_kv_quantize`, which is
        # the branch `NVFP4KVQuantizeUtil.quantize` takes on SM100/SM103 only.
        # SM120 shares the API but has no bit-exact acceptance yet, and SM90
        # takes a different quantizer entirely.
        self.use_nvfp4_fused_quantize_store = (
            envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get()
            and get_platform().is_sm100
            and self.main_pool.head_dim == MiniMaxM3Nvfp4IndexKPool.INDEX_HEAD_DIM
            and self.main_pool.page_size == self.index_k_pool.page_size
            # One kernel argument carries the outer scale for all three
            # streams, so main and index must agree on it.
            and torch.equal(self.main_pool.global_scale, self.index_k_pool.global_scale)
        )
        logger.info(
            "MiniMax M3.1 NVFP4 KV store: %s",
            (
                "fused quantize+store"
                if self.use_nvfp4_fused_quantize_store
                else "per-buffer quantize then scatter"
            ),
        )

    def get_index_k_scale_buffer(self, layer_id):
        return self.index_k_pool.get_scale_buffer(
            self.index_k_layer_id_mapping[layer_id]
        )

    def get_main_k_scale_buffer(self, layer_id):
        return self.main_pool.get_k_scale_buffer(layer_id)

    def get_main_v_scale_buffer(self, layer_id):
        return self.main_pool.get_v_scale_buffer(layer_id)

    def set_fused_kv_index_buffer(
        self,
        layer: RadixAttention,
        loc,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: Optional[torch.Tensor],
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        idx_k_scale: Optional[float] = None,
        idx_v_scale: Optional[float] = None,
    ) -> None:
        """Quantize and store main K/V and index K in one launch.

        Falls back to the separate per-cache stores whenever the fused kernel is
        not enabled for this platform or input, so the eager path stays the
        reference and remains available for A/B.
        """
        if cache_idx_v is not None:
            raise ValueError("M3.1 NVFP4 requires a K-only sparse index")
        for name, scale in (
            ("k_scale", k_scale),
            ("v_scale", v_scale),
            ("idx_k_scale", idx_k_scale),
        ):
            if scale not in (None, 1.0):
                raise ValueError(
                    f"MSA-v1 NVFP4 uses fixed unit outer scales, got {name}={scale}."
                )

        if not (
            self.use_nvfp4_fused_quantize_store
            and cache_k.dtype == torch.bfloat16
            and cache_v.dtype == cache_k.dtype
            and cache_idx_k.dtype == cache_k.dtype
        ):
            self.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)
            self.set_index_k_buffer(layer, loc, cache_idx_k, idx_k_scale)
            return

        from sglang.kernels.ops.kvcache.nvfp4_store_kv_index import (
            nvfp4_store_kv_index,
        )

        main = self.main_pool
        index = self.index_k_pool
        write_loc, _, _ = unwrap_write_loc(loc)
        maybe_detect_oob(
            write_loc,
            0,
            main.size + main.page_size,
            "set_fused_kv_index_buffer (MiniMax MSA-v1 NVFP4)",
        )
        expected_tail = (main.head_num, main.head_dim)
        if tuple(cache_k.shape[1:]) != expected_tail or cache_v.shape != cache_k.shape:
            raise ValueError(
                "MSA-v1 NVFP4 main cache expects equal K/V tensors with shape "
                f"[tokens, {main.head_num}, {main.head_dim}], got "
                f"K={tuple(cache_k.shape)}, V={tuple(cache_v.shape)}."
            )

        k_payload, v_payload, k_scale_buf, v_scale_buf = main.get_raw_kv_buffers(
            layer.layer_id
        )
        idx_payload, idx_scale_buf = index.get_raw_k_buffers(
            self.index_k_layer_id_mapping[layer.layer_id]
        )
        nvfp4_store_kv_index(
            cache_k,
            cache_v,
            cache_idx_k.reshape(-1, main.head_dim),
            k_payload,
            k_scale_buf,
            v_payload,
            v_scale_buf,
            idx_payload,
            idx_scale_buf,
            write_loc.reshape(-1),
            main.global_scale,
            page_size=main.page_size,
        )

    def set_index_k_buffer(self, layer, loc, cache_idx_k, k_scale=None):
        if k_scale not in (None, 1.0):
            raise ValueError("M3.1 index K requires unit outer scale")
        self.index_k_pool.set_k_buffer(
            self.index_k_layer_id_mapping[layer.layer_id], loc, cache_idx_k
        )

    def get_sparse_kv_buf_infos(self):
        return self.main_pool.get_contiguous_buf_infos()

    def get_kv_layer_ids(self):
        # One main page row per layer has no K/V halves; peers pair rows by id.
        return list(range(self.main_pool.start_layer, self.main_pool.end_layer))

    def get_index_k_state_buf_infos(self):
        return self.index_k_pool.get_contiguous_buf_infos()

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        return {
            "main": self.main_pool.get_cpu_copy(indices),
            "index_k": self.index_k_pool.get_cpu_copy(indices),
        }

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        self.main_pool.load_cpu_copy(kv_cache_cpu["main"], indices)
        self.index_k_pool.load_cpu_copy(kv_cache_cpu["index_k"], indices)

    def get_v_head_dim(self):
        return self.main_pool.v_head_dim

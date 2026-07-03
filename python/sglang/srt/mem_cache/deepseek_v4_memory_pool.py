from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch

from sglang.jit_kernel.dsv4 import (
    clear_unaccepted_c128_draft_states,
    fused_k_norm_rope_flashmla,
    fused_store_cache,
)
from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.runtime_context import get_server_args
from sglang.srt.platforms import current_platform
from sglang.srt.utils import ceil_div, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # is incompatible with speculative decode for now.
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError("online c128 does not support MTP")
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


class DeepSeekV4SingleKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
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
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        self.scale_pad = 1
        self.quantize_block_size = 64
        self.rope_storage_dtype = torch.bfloat16
        self.k_with_scale_buffer_dtype = torch.int8
        self._create_buffers()

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

        assert bytes_per_token == 448 + 64 * 2 + 8, (
            "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
            "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
        )
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        write_mask: Optional[torch.Tensor] = None,
    ):
        dsv4_index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            write_mask=write_mask,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
        )

    def set_key_buffer_fused_fallback_triton(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """DCP fallback path: quantize the bf16 ``cache_k`` to a NopeFp8RopeBf16
        pack on-device, then write through the Triton kernel which honors
        ``dcp_world_size``/``dcp_rank``. Used when ``dcp_size > 1`` because the
        fused C++ flashmla store kernel is not yet DCP-aware.
        """
        from sglang.srt.layers.attention.dsv4.quant_k_cache import (
            quant_to_nope_fp8_rope_bf16_pack_triton,
        )

        pack = quant_to_nope_fp8_rope_bf16_pack_triton(cache_k.bfloat16())
        self.set_key_buffer(
            layer_id,
            loc,
            pack,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            write_mask=write_mask,
        )

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")

    def _value_bytes_per_token(self) -> int:
        return (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
        )

    def _scale_bytes_per_token(self) -> int:
        return self.qk_nope_head_dim // self.quantize_block_size + self.scale_pad

    def _copy_token_rows_to_cpu(self, buf: torch.Tensor, indices: torch.Tensor):
        value_bytes = self._value_bytes_per_token()
        scale_bytes = self._scale_bytes_per_token()
        pages = (indices // self.page_size).to(torch.long)
        offsets = (indices % self.page_size).to(torch.long)

        value_byte_offsets = torch.arange(value_bytes, device=buf.device)
        value_offsets = offsets[:, None] * value_bytes + value_byte_offsets[None, :]
        values_cpu = buf[pages[:, None], value_offsets].to("cpu", non_blocking=True)

        scale_byte_offsets = torch.arange(scale_bytes, device=buf.device)
        scale_base = self.page_size * value_bytes
        scale_offsets = (
            scale_base
            + offsets[:, None] * scale_bytes
            + scale_byte_offsets[None, :]
        )
        scales_cpu = buf[pages[:, None], scale_offsets].to("cpu", non_blocking=True)
        return values_cpu, scales_cpu

    def _load_token_rows_from_cpu(
        self,
        buf: torch.Tensor,
        indices: torch.Tensor,
        values_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
    ) -> None:
        value_bytes = self._value_bytes_per_token()
        scale_bytes = self._scale_bytes_per_token()
        pages = (indices // self.page_size).to(torch.long)
        offsets = (indices % self.page_size).to(torch.long)

        value_byte_offsets = torch.arange(value_bytes, device=buf.device)
        value_offsets = offsets[:, None] * value_bytes + value_byte_offsets[None, :]
        values = values_cpu.to(buf.device, non_blocking=True)
        buf[pages[:, None], value_offsets] = values

        scale_byte_offsets = torch.arange(scale_bytes, device=buf.device)
        scale_base = self.page_size * value_bytes
        scale_offsets = (
            scale_base
            + offsets[:, None] * scale_bytes
            + scale_byte_offsets[None, :]
        )
        scales = scales_cpu.to(buf.device, non_blocking=True)
        buf[pages[:, None], scale_offsets] = scales

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Move packed DSV4 token rows between cache slots."""
        if self.layer_num == 0 or tgt_loc.numel() == 0:
            return

        tgt_loc = tgt_loc.view(-1).long()
        src_loc = src_loc.view(-1).long()

        value_bytes = self._value_bytes_per_token()
        scale_bytes = self._scale_bytes_per_token()
        value_byte_offsets = torch.arange(value_bytes, device=self.device)
        scale_byte_offsets = torch.arange(scale_bytes, device=self.device)
        scale_base = self.page_size * value_bytes

        pages_t = tgt_loc // self.page_size
        offsets_t = tgt_loc % self.page_size
        pages_s = src_loc // self.page_size
        offsets_s = src_loc % self.page_size

        value_offsets_t = offsets_t[:, None] * value_bytes + value_byte_offsets[None, :]
        value_offsets_s = offsets_s[:, None] * value_bytes + value_byte_offsets[None, :]
        scale_offsets_t = (
            scale_base + offsets_t[:, None] * scale_bytes + scale_byte_offsets[None, :]
        )
        scale_offsets_s = (
            scale_base + offsets_s[:, None] * scale_bytes + scale_byte_offsets[None, :]
        )

        for buf in self.kv_buffer:
            values = buf[pages_s[:, None], value_offsets_s].clone()
            scales = buf[pages_s[:, None], scale_offsets_s].clone()
            buf[pages_t[:, None], value_offsets_t] = values
            buf[pages_t[:, None], scale_offsets_t] = scales

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            layer_chunks = []
            buf = self.kv_buffer[layer_id]
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                layer_chunks.append(self._copy_token_rows_to_cpu(buf, chunk_indices))
            kv_cache_cpu.append(layer_chunks)
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            buf = self.kv_buffer[layer_id]
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                values_cpu, scales_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert values_cpu.shape[0] == scales_cpu.shape[0] == len(chunk_indices)
                self._load_token_rows_from_cpu(
                    buf, chunk_indices, values_cpu, scales_cpu
                )
        current_platform.synchronize()


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        write_mask: Optional[torch.Tensor] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_key_buffer(
            layer_id,
            loc,
            cache_nope_fp8_rope_bf16_pack,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            write_mask=write_mask,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k)

    def set_key_buffer_fused_fallback_triton(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused_fallback_triton(
            layer_id,
            loc,
            cache_k,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            write_mask=write_mask,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
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
        self.index_head_dim = index_head_dim
        self.use_fp4_indexer = get_server_args().enable_deepseek_v4_fp4_indexer

        self._create_buffer()

    def get_bytes_per_token(self) -> int:
        if self.use_fp4_indexer:
            return self.index_head_dim // 2 + 4
        return self.index_head_dim + 4

    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        (self.size + self.page_size + 1) // self.page_size,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Move C4 indexer key+scale rows between cache slots."""
        if self.layer_num == 0 or tgt_loc.numel() == 0:
            return

        tgt_loc = tgt_loc.view(-1).long()
        src_loc = src_loc.view(-1).long()

        k_bytes = (
            self.index_head_dim // 2 if self.use_fp4_indexer else self.index_head_dim
        )
        scale_bytes = 4
        k_offsets = torch.arange(k_bytes, device=self.device)
        scale_offsets = torch.arange(scale_bytes, device=self.device)
        scale_base = self.page_size * k_bytes

        pages_t = tgt_loc // self.page_size
        offsets_t = tgt_loc % self.page_size
        pages_s = src_loc // self.page_size
        offsets_s = src_loc % self.page_size

        k_offsets_t = offsets_t[:, None] * k_bytes + k_offsets[None, :]
        k_offsets_s = offsets_s[:, None] * k_bytes + k_offsets[None, :]
        scale_offsets_t = (
            scale_base + offsets_t[:, None] * scale_bytes + scale_offsets[None, :]
        )
        scale_offsets_s = (
            scale_base + offsets_s[:, None] * scale_bytes + scale_offsets[None, :]
        )

        for buf in self.index_k_with_scale_buffer:
            keys = buf[pages_s[:, None], k_offsets_s].clone()
            scales = buf[pages_s[:, None], scale_offsets_s].clone()
            buf[pages_t[:, None], k_offsets_t] = keys
            buf[pages_t[:, None], scale_offsets_t] = scales

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            write_mask=write_mask,
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )

    def set_index_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        return store_fp4_index_k_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
        )

    def _scale_bytes_per_token(self) -> int:
        return (self.index_head_dim // self.quant_block_size) * 4

    def _copy_token_rows_to_cpu(self, buf: torch.Tensor, indices: torch.Tensor):
        value_bytes = self.index_head_dim
        scale_bytes = self._scale_bytes_per_token()
        pages = (indices // self.page_size).to(torch.long)
        offsets = (indices % self.page_size).to(torch.long)

        value_byte_offsets = torch.arange(value_bytes, device=buf.device)
        value_offsets = offsets[:, None] * value_bytes + value_byte_offsets[None, :]
        values_cpu = buf[pages[:, None], value_offsets].to("cpu", non_blocking=True)

        scale_byte_offsets = torch.arange(scale_bytes, device=buf.device)
        scale_base = self.page_size * value_bytes
        scale_offsets = (
            scale_base
            + offsets[:, None] * scale_bytes
            + scale_byte_offsets[None, :]
        )
        scales_cpu = buf[pages[:, None], scale_offsets].to("cpu", non_blocking=True)
        return values_cpu, scales_cpu

    def _load_token_rows_from_cpu(
        self,
        buf: torch.Tensor,
        indices: torch.Tensor,
        values_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
    ) -> None:
        value_bytes = self.index_head_dim
        scale_bytes = self._scale_bytes_per_token()
        pages = (indices // self.page_size).to(torch.long)
        offsets = (indices % self.page_size).to(torch.long)

        value_byte_offsets = torch.arange(value_bytes, device=buf.device)
        value_offsets = offsets[:, None] * value_bytes + value_byte_offsets[None, :]
        values = values_cpu.to(buf.device, non_blocking=True)
        buf[pages[:, None], value_offsets] = values

        scale_byte_offsets = torch.arange(scale_bytes, device=buf.device)
        scale_base = self.page_size * value_bytes
        scale_offsets = (
            scale_base
            + offsets[:, None] * scale_bytes
            + scale_byte_offsets[None, :]
        )
        scales = scales_cpu.to(buf.device, non_blocking=True)
        buf[pages[:, None], scale_offsets] = scales

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            layer_chunks = []
            buf = self.index_k_with_scale_buffer[layer_id]
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                layer_chunks.append(self._copy_token_rows_to_cpu(buf, chunk_indices))
            kv_cache_cpu.append(layer_chunks)
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            buf = self.index_k_with_scale_buffer[layer_id]
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                values_cpu, scales_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert values_cpu.shape[0] == scales_cpu.shape[0] == len(chunk_indices)
                self._load_token_rows_from_cpu(
                    buf, chunk_indices, values_cpu, scales_cpu
                )
        current_platform.synchronize()


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


# The following kv pool follows ATOM's unified_kv kernel layout.
class DeepSeekV4UnifiedKVPool:
    """
    Layout:
    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16
    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)
    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)
    """

    K_PER_BLOCK = {0: 0, 4: 32, 128: 1}

    def __init__(
        self,
        *,
        stage_ratios: List[int],
        num_slots: int,
        num_blocks: int,
        page_size: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        device: str,
        memory_saver_adapter,
        custom_mem_pool,
        swa_ring_size: int,
    ):
        self.swa_ring_size = swa_ring_size
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_slots = num_slots
        self.swa_pages = num_slots * self.swa_ring_size
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.k_per_block = dict(self.K_PER_BLOCK)

        bufs = []
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(custom_mem_pool)
                if custom_mem_pool
                else nullcontext()
            ):
                for ratio in stage_ratios:
                    # Pad by one extra page. The KV pool reserves a null slot
                    # (token indices run 1..size).
                    compress_rows = self.num_blocks * self.k_per_block[ratio]
                    rows_per_page = self.page_size // ratio if ratio else 0
                    padded_compress_rows = compress_rows + rows_per_page
                    bufs.append(
                        torch.zeros(
                            self.swa_pages + padded_compress_rows,
                            self.head_dim,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
        self.kv_buffer = bufs

    def get_unified_kv(self, local_layer_id: int) -> torch.Tensor:
        return self.kv_buffer[local_layer_id]

    def get_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs = [b.data_ptr() for b in self.kv_buffer]
        data_lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes for b in self.kv_buffer]
        return data_ptrs, data_lens, item_lens


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        c4_state_dtype: torch.dtype,
        c128_state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        sliding_window: int = 128,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        online_mtp_max_draft_tokens: int = 0,
        c4_indexer_size: Optional[int] = None,
        c4_indexer_state_pool_size: Optional[int] = None,
        num_req_slots: Optional[int] = None,
    ):
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        c4_logical_size = (
            c4_indexer_size if c4_indexer_size is not None else c128_size * 32
        )

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} "
            f"{c4_indexer_state_pool_size=} "
            f"{c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        # SWA ring needs one slot per addressable req_pool_idx. PD decode inflates
        # req_to_token past max_num_reqs (pre-alloc), so the caller passes the real
        # capacity; sizing as max_num_reqs+1 overflows ("length out of range").
        self.num_req_slots = (
            num_req_slots if num_req_slots is not None else max_num_reqs + 1
        )
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        self.c4_indexer_state_pool_size = (
            c4_indexer_state_pool_size
            if c4_indexer_state_pool_size is not None
            else c4_state_pool_size
        )
        c128_ring_size = self.get_ring_size(128)
        if ONLINE_C128:
            # Request-scoped online C128 state is indexed by req_pool_idx.
            # PD decode can allocate pre-transfer slots beyond
            # max_num_reqs, so size to the actual req_to_token row count.
            c128_state_pool_size = max(c128_state_pool_size, self.num_req_slots)
        else:
            # Offline C128 keeps a per-request raw state ring.
            c128_state_pool_size = max(
                c128_state_pool_size, self.num_req_slots * c128_ring_size
            )
        self.c128_state_pool_size = c128_state_pool_size
        self.c4_state_dtype = c4_state_dtype
        self.c128_state_dtype = c128_state_dtype
        self.compression_ratios = compression_ratios
        self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
        self.online_c128_state_num_req_slots = c128_state_pool_size
        self.online_c128_mtp_pending_seq_lens: Optional[torch.Tensor] = None
        if ONLINE_C128 and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            self.online_c128_mtp_pending_seq_lens = torch.empty(
                self.online_c128_state_num_req_slots, dtype=torch.int64, device=device
            )
            self.online_c128_mtp_pending_seq_lens.fill_(-1)

        # Determine this PP stage's absolute layer range
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer
            self._stage_end = end_layer
        else:
            self._stage_start = 0
            self._stage_end = len(compression_ratios)
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]

        assert page_size % swa_page_size == 0
        self.sliding_window = sliding_window

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = 1

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        stage_layer_num = len(stage_ratios)
        c4_layer_num = sum(1 for r in stage_ratios if r == 4)
        c128_layer_num = sum(1 for r in stage_ratios if r == 128)
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        self._unified_kv = is_unified_kv_triton()

        if self._unified_kv:
            self.swa_kv_pool = None
            self.c4_kv_pool = None
            self.c128_kv_pool = None
            server_args = get_server_args()
            spec_extra = (
                (server_args.speculative_num_draft_tokens - 1)
                if server_args.speculative_algorithm is not None
                else 0
            )
            self.unified_kv_pool = DeepSeekV4UnifiedKVPool(
                stage_ratios=stage_ratios,
                num_slots=self.num_req_slots,
                num_blocks=self.c128_size,
                page_size=page_size,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                device=device,
                memory_saver_adapter=self.memory_saver_adapter,
                custom_mem_pool=self.custom_mem_pool,
                swa_ring_size=self.sliding_window + spec_extra,
            )

            self.unified_swa_window = self.sliding_window
            self.unified_swa_ring_size = self.sliding_window + spec_extra
            self.unified_swa_pages = self.unified_kv_pool.swa_pages
        else:
            self.unified_kv_pool = None
            self.swa_kv_pool = self._make_kv_pool(
                size=swa_size,
                page_size=swa_page_size,
                dtype=dtype,
                layer_num=stage_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=swa_page_size,
            )

            c4_kv_pool_type = DeepSeekV4SingleKVPool
            if enable_hisparse:
                c4_kv_pool_type = HiSparseC4DevicePool
            self.c4_kv_pool = self._make_kv_pool(
                size=c4_size,
                page_size=c4_page_size,
                dtype=dtype,
                layer_num=c4_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=page_size,
                cls=c4_kv_pool_type,
            )

            self.c128_kv_pool = self._make_kv_pool(
                size=c128_size,
                page_size=c128_page_size,
                dtype=dtype,
                layer_num=c128_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=page_size,
            )

        indexer_size = self.c4_logical_size
        self.c4_indexer_kv_pool = self._make_indexer_pool(
            indexer_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self._init_compressed_layer_mapping()

        if _is_hip:
            self._init_paged_compress_states(False)
        else:
            self._init_paged_compress_states(enable_memory_saver)

    def get_unified_kv(self, layer_id: int) -> torch.Tensor:
        # Under HiCache the compressed region is loaded H->D per layer; wait for this
        # layer's transfer before attention reads it. No-op when HiCache is off.
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        server_args = get_server_args()
        is_speculative = server_args.speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def _localize_dcp_move_locs(
        self, tgt_loc: torch.Tensor, src_loc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dcp_world_size, dcp_rank = self._dcp_world_rank()
        if dcp_world_size == 1 or tgt_loc.numel() == 0:
            return tgt_loc, src_loc

        local_mask = (tgt_loc % dcp_world_size == dcp_rank) & (
            src_loc % dcp_world_size == dcp_rank
        )
        if not torch.any(local_mask):
            return tgt_loc.new_empty((0,)), src_loc.new_empty((0,))
        return (
            tgt_loc[local_mask] // dcp_world_size,
            src_loc[local_mask] // dcp_world_size,
        )

    @staticmethod
    def _compressed_move_locs_from_full(
        tgt_loc: torch.Tensor, src_loc: torch.Tensor, compress_ratio: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tgt_loc.numel() == 0:
            return tgt_loc, src_loc
        mask = ((tgt_loc + 1) % compress_ratio == 0) & (
            (src_loc + 1) % compress_ratio == 0
        )
        return (
            (tgt_loc[mask] // compress_ratio).to(tgt_loc.dtype),
            (src_loc[mask] // compress_ratio).to(src_loc.dtype),
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Move accepted speculative DSV4 KV rows into their committed slots."""
        if tgt_loc.numel() == 0:
            return

        tgt_loc = tgt_loc.view(-1).long()
        src_loc = src_loc.view(-1).long()

        if self._unified_kv:
            raise NotImplementedError("DSV4 unified KV move is not supported yet.")

        tgt_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_swa = self.translate_loc_from_full_to_swa(src_loc)
        tgt_swa_local, src_swa_local = self._localize_dcp_move_locs(tgt_swa, src_swa)
        self.swa_kv_pool.move_kv_cache(tgt_swa_local, src_swa_local)

        tgt_c4, src_c4 = self._compressed_move_locs_from_full(tgt_loc, src_loc, 4)
        if tgt_c4.numel() and src_c4.numel():
            # C4 indexer cache is replicated in global c4-index space.
            self.c4_indexer_kv_pool.move_kv_cache(tgt_c4, src_c4)
            tgt_c4_local, src_c4_local = self._localize_dcp_move_locs(tgt_c4, src_c4)
            self.c4_kv_pool.move_kv_cache(tgt_c4_local, src_c4_local)

        tgt_c128, src_c128 = self._compressed_move_locs_from_full(tgt_loc, src_loc, 128)
        if tgt_c128.numel() and src_c128.numel():
            tgt_c128_local, src_c128_local = self._localize_dcp_move_locs(
                tgt_c128, src_c128
            )
            self.c128_kv_pool.move_kv_cache(tgt_c128_local, src_c128_local)

    def move_accepted_kv_cache(
        self,
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        accepted_seq_lens: torch.Tensor,
    ):
        """Move speculative accepted rows using logical seq-len boundaries.

        DSV4 compressed KV is written when the logical sequence length hits a
        compression boundary. The physical full-cache slot number is allocator
        state and must not be used to infer C4/C128 boundaries.
        """
        if tgt_loc.numel() == 0:
            return

        tgt_loc = tgt_loc.view(-1).long()
        src_loc = src_loc.view(-1).long()
        accepted_seq_lens = accepted_seq_lens.view(-1).to(device=tgt_loc.device)

        if self._unified_kv:
            raise NotImplementedError("DSV4 unified KV move is not supported yet.")

        tgt_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_swa = self.translate_loc_from_full_to_swa(src_loc)
        tgt_swa_local, src_swa_local = self._localize_dcp_move_locs(tgt_swa, src_swa)
        self.swa_kv_pool.move_kv_cache(tgt_swa_local, src_swa_local)

        tgt_c4, src_c4 = self._compressed_move_locs_from_boundary_mask(
            tgt_loc, src_loc, accepted_seq_lens, 4
        )
        if tgt_c4.numel() and src_c4.numel():
            self.c4_indexer_kv_pool.move_kv_cache(tgt_c4, src_c4)
            tgt_c4_local, src_c4_local = self._localize_dcp_move_locs(tgt_c4, src_c4)
            self.c4_kv_pool.move_kv_cache(tgt_c4_local, src_c4_local)

        tgt_c128, src_c128 = self._compressed_move_locs_from_boundary_mask(
            tgt_loc, src_loc, accepted_seq_lens, 128
        )
        if tgt_c128.numel() and src_c128.numel():
            tgt_c128_local, src_c128_local = self._localize_dcp_move_locs(
                tgt_c128, src_c128
            )
            self.c128_kv_pool.move_kv_cache(tgt_c128_local, src_c128_local)

    @staticmethod
    def _compressed_move_locs_from_boundary_mask(
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        accepted_seq_lens: torch.Tensor,
        compress_ratio: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tgt_loc.numel() == 0:
            return tgt_loc, src_loc
        token_len = min(tgt_loc.numel(), src_loc.numel(), accepted_seq_lens.numel())
        tgt_loc = tgt_loc[:token_len]
        src_loc = src_loc[:token_len]
        accepted_seq_lens = accepted_seq_lens[:token_len]
        mask = accepted_seq_lens % compress_ratio == 0
        return (
            (tgt_loc[mask] // compress_ratio).to(tgt_loc.dtype),
            (src_loc[mask] // compress_ratio).to(src_loc.dtype),
        )

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self._unified_kv:
            # Unified buffer per layer: [swa_pages + padded_compress_rows, head_dim].
            # Compressed region [swa_pages:] is page-contiguous (row swa_pages +
            # loc//ratio), so reuse the page-block PD transfer by offsetting the ptr
            # past the SWA ring and setting item_len = one page of rows. The SWA ring
            # ships separately as StateType.SWA_RING. Order [c4, c4_indexer, c128]
            # mirrors the non-unified kv_data layout (keeps PP ptr-slicing valid).
            stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
            swa_pages = self.unified_kv_pool.swa_pages

            def _append_compressed_entry(local_layer_id: int, ratio: int) -> None:
                buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                row_bytes = buf[0].nbytes
                rows_per_page = self.page_size // ratio
                compress_rows = buf.shape[0] - swa_pages
                data_ptrs.append(buf.data_ptr() + swa_pages * row_bytes)
                data_lens.append(compress_rows * row_bytes)
                item_lens.append(rows_per_page * row_bytes)

            c4_locals = [i for i, r in enumerate(stage_ratios) if r == 4]
            c128_locals = [i for i, r in enumerate(stage_ratios) if r == 128]

            for i in c4_locals:
                _append_compressed_entry(i, 4)
            for buf in self.c4_indexer_kv_pool.index_k_with_scale_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
            for i in c128_locals:
                _append_compressed_entry(i, 128)

            return data_ptrs, data_lens, item_lens

        buf_groups = [
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]

        for bufs in buf_groups:
            for buf in bufs:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_unified_swa_ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """SWA-ring region [0, swa_pages) of every unified_kv layer, addressed
        per-row by ring slot. Shipped as the StateType.SWA_RING PD component."""
        # TODO(billishyahao): validate PP layer-slicing for SWA_RING.
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if not self._unified_kv:
            return data_ptrs, data_lens, item_lens
        swa_pages = self.unified_kv_pool.swa_pages
        for buf in self.unified_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            row_bytes = buf[0].nbytes
            data_ptrs.append(buf.data_ptr())
            data_lens.append(swa_pages * row_bytes)
            item_lens.append(row_bytes)
        return data_ptrs, data_lens, item_lens

    def unified_region_buffers(self, ratio: int) -> Tuple[List[torch.Tensor], int]:
        """
        In unified_kv, swa/c4/c128 share one buffer with one slot per row. But the
        HiCache host pool transfers a whole page per indexed row, so we reshape the
        compressed region into the layout it expects: skip the SWA segment, reshape to
        one row per page, then cast to uint8.
        """
        assert self._unified_kv, "unified_region_buffers requires unified_kv layout"
        assert ratio in (4, 128), f"unsupported compression ratio: {ratio}"

        swa_pages = self.unified_kv_pool.swa_pages
        head_dim = self.unified_kv_pool.head_dim
        rows_per_page = self.page_size // ratio
        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        local_layer_ids = [i for i, r in enumerate(stage_ratios) if r == ratio]

        views: List[torch.Tensor] = []
        for local_layer_id in local_layer_ids:
            buf = self.unified_kv_pool.kv_buffer[local_layer_id]
            compress_rows = buf.shape[0] - swa_pages
            assert compress_rows % rows_per_page == 0, (
                f"compressed rows {compress_rows} not a multiple of "
                f"rows_per_page {rows_per_page} for ratio {ratio}"
            )
            num_pages = compress_rows // rows_per_page
            page_view = (
                buf.narrow(0, swa_pages, compress_rows)
                .reshape(num_pages, rows_per_page * head_dim)
                .view(torch.uint8)
            )
            views.append(page_view)

        item_bytes = (
            rows_per_page * head_dim * self.unified_kv_pool.kv_buffer[0].element_size()
        )
        return views, item_bytes

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if not self._unified_kv:
            for buf in self.swa_kv_pool.kv_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                if pool.ratio == 128:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_c128_state_buf_infos(
        self,
    ) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue
            t = pool.kv_score_buffer.kv_score
            assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
            data_ptrs.append(t.data_ptr())
            data_lens.append(t.nbytes)
            item_lens.append(t[0].nbytes if ONLINE_C128 else t[0].nbytes * 128)
        return data_ptrs, data_lens, item_lens

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
    ) -> DeepSeekV4SingleKVPool:
        """Build a full / SWA / c4 / c128 single-KV pool. ``global_page_size``
        is the model-wide page_size (== ``page_size`` for the SWA pool, larger
        for the per-ratio c4/c128 pools); the default CUDA pool ignores it.
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the NPU bf16
        PA_ND variant, which needs ``global_page_size`` for its kernel view."""
        del global_page_size  # CUDA pools key only off their own page_size
        return cls(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ) -> DeepSeekV4IndexerPool:
        """Build the c4 lightning-indexer K pool (packed CUDA layout).
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the
        dedicated-buffer NPU variant (int8 K + fp16 scale)."""
        return DeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _state_pool_size(self, ratio: int) -> int:
        return self.c4_state_pool_size if ratio == 4 else self.c128_state_pool_size

    def _make_attn_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer attention compress-state pool for ``ratio``
        (4 or 128). Overridden by :class:`DSV4NPUTokenToKVPool` to swap the
        ring-buffered pool for the NPU paged one."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=(ratio == 128 and ONLINE_C128),
            swa_page_size=self.swa_page_size,
            online_mtp_max_draft_tokens=(
                self.online_mtp_max_draft_tokens if ratio == 128 else 0
            ),
        )

    def _make_indexer_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer indexer compress-state pool (c4 only)."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.indexer_head_dim,
            device=self.device,
            dtype=self.c4_state_dtype,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            swa_page_size=self.swa_page_size,
        )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        c4_state_pool_size = self.c4_state_pool_size
        c4_indexer_state_pool_size = self.c4_indexer_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        total_L = len(self.compression_ratios)
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue

            self.compress_state_pools[idx] = self._make_attn_state_pool(
                ratio, enable_memory_saver
            )

            if ratio == 4:
                self.indexer_compress_state_pools[idx] = self._make_indexer_state_pool(
                    ratio, enable_memory_saver
                )

    def _init_compressed_layer_mapping(self):
        c1_cnt = c4_cnt = c128_cnt = 0
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=c1_cnt,
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,
                    compress_layer_id=c4_cnt,
                    compress_kv_pool=self.c4_kv_pool,
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=c128_cnt,
                    compress_kv_pool=self.c128_kv_pool,
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def wait_layer_transfer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert (
            compress_state_pool is not None
        ), "Only c4/c128 layers have attention states."
        return compress_state_pool

    def get_online_c128_mtp_state_slot_offset(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_state_slot_offset)
        return 0

    def get_online_c128_mtp_max_draft_tokens(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_max_draft_tokens)
        return 0

    def get_online_c128_state_num_req_slots(self) -> int:
        return self.online_c128_state_num_req_slots

    def get_online_c128_mtp_pending_seq_lens(self) -> torch.Tensor:
        assert self.online_c128_mtp_pending_seq_lens is not None
        return self.online_c128_mtp_pending_seq_lens

    def clear_c128_req_state(self, req_pool_idx: int) -> None:
        """Reset request-scoped C128 state for one req slot."""
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            state = pool.kv_score_buffer.kv_score
            if ONLINE_C128:
                rows = [req_pool_idx]
                if pool.online_mtp_max_draft_tokens > 0:
                    rows = [
                        req_pool_idx + i * pool.online_mtp_state_slot_offset
                        for i in range(pool.online_mtp_max_draft_tokens + 1)
                    ]
                rows = torch.as_tensor(rows, dtype=torch.long, device=state.device)
                head_dim = state.shape[-1] // 3
                state[rows, :head_dim] = float("-inf")
                state[rows, head_dim:] = 0
            else:
                start = req_pool_idx * pool.ring_size
                rows = state[start : start + pool.ring_size]
                half = rows.shape[-1] // 2
                rows[:, :half].zero_()
                rows[:, half:].fill_(float("-inf"))

        if self.online_c128_mtp_pending_seq_lens is not None:
            self.online_c128_mtp_pending_seq_lens[req_pool_idx] = -1

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        """Clear offline C128 ring slots written for rejected speculative tokens."""
        if ONLINE_C128 or num_draft_tokens <= 1 or req_pool_indices.numel() == 0:
            return

        bs = req_pool_indices.numel()
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            clear_unaccepted_c128_draft_states(
                pool.kv_score_buffer.kv_score,
                req_pool_indices,
                seq_lens,
                accept_lens,
                ring_size=pool.ring_size,
                num_draft_tokens=num_draft_tokens,
            )

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert (
            indexer_compress_state_pool is not None
        ), "Only c4 layers have indexer states."
        return indexer_compress_state_pool

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Convert absolute model layer_id to SWA-pool-local (PP-stage-local) index."""
        return layer_id - self._stage_start

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def _dcp_world_rank(self) -> Tuple[int, int]:
        """Resolve current DCP (decode context parallel) world size and rank.

        Imported lazily so non-DCP setups don't pay an import cost. Returns
        ``(world_size, rank)``; when DCP is disabled the group is ``None``
        and we fall back to ``(1, 0)`` which makes the Triton kernel a no-op
        on the dcp dimension.
        """
        from sglang.srt.distributed.parallel_state import get_dcp_group_no_assert

        group = get_dcp_group_no_assert()
        if group is None or group.world_size == 1:
            return 1, 0
        return group.world_size, group.rank_in_group

    def _dcp_write_context(self) -> Tuple[int, int, bool]:
        dcp_world_size, dcp_rank = self._dcp_world_rank()
        return dcp_world_size, dcp_rank, dcp_world_size > 1

    @staticmethod
    def _dcp_loc_write_mask(loc: torch.Tensor) -> torch.Tensor:
        # Keep only the validity part here: full-cache loc 0 is the allocator's
        # dummy/padding slot and must not be persisted as KV. Callers that write
        # translated locations (for example SWA ring slots) must pass the raw
        # full-cache loc instead, because translated slot 0 can be valid.
        return (loc > 0).contiguous()

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ) -> None:
        dcp_world_size, dcp_rank, dcp_enabled = self._dcp_write_context()
        if dcp_enabled or dcp_kv_mask is not None:
            write_mask = self._dcp_loc_write_mask(loc)
        else:
            write_mask = None
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id),
            loc,
            cache_nope_fp8_rope_bf16_pack,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            write_mask=write_mask,
        )

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        self.wait_layer_transfer(layer_id)
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
        dcp_kv_mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        dcp_world_size, dcp_rank, _ = self._dcp_write_context()
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id,
            loc,
            cache_nope_fp8_rope_bf16_pack,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            write_mask=(write_mask.contiguous() if write_mask is not None else None),
        )

    def get_index_k_page_size(self) -> int:
        return self.c4_indexer_kv_pool.page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id,
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
        dcp_kv_mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        # The c4 indexer is a scoring cache used by paged_mqa_logits with the
        # global c4 page_table, and its topk result is later localized for DCP
        # attention. Keep this cache replicated in global c4-index space.
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id,
            loc,
            index_k,
            index_k_scale,
            write_mask=write_mask.contiguous() if write_mask is not None else None,
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
        dcp_kv_mask: Optional[torch.Tensor] = None,
        raw_loc: Optional[torch.Tensor] = None,
    ) -> None:
        dcp_world_size, dcp_rank, dcp_enabled = self._dcp_write_context()
        if dcp_enabled or dcp_kv_mask is not None:
            write_mask = self._dcp_loc_write_mask(
                raw_loc if raw_loc is not None else swa_loc
            )
        else:
            write_mask = None
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id),
            swa_loc,
            cache_nope_fp8_rope_bf16_pack,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            write_mask=write_mask,
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
        dcp_kv_mask: Optional[torch.Tensor] = None,
        raw_loc: Optional[torch.Tensor] = None,
    ) -> None:
        dcp_world_size, dcp_rank, dcp_enabled = self._dcp_write_context()
        if dcp_enabled or dcp_kv_mask is not None:
            return self.swa_kv_pool.set_key_buffer_fused_fallback_triton(
                self._swa_local_layer_id(layer_id),
                swa_loc,
                cache_k,
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                write_mask=self._dcp_loc_write_mask(
                    raw_loc if raw_loc is not None else swa_loc
                ),
            )
        return self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        dcp_kv_mask: Optional[torch.Tensor] = None,
        raw_loc: Optional[torch.Tensor] = None,
    ) -> None:
        dcp_world_size, dcp_rank, dcp_enabled = self._dcp_write_context()
        if dcp_enabled or dcp_kv_mask is not None:
            from sglang.jit_kernel.dsv4 import fused_norm_rope_inplace

            kv = kv.contiguous()
            fused_norm_rope_inplace(kv, kv_weight, eps, freqs_cis, positions)
            return self.swa_kv_pool.set_key_buffer_fused_fallback_triton(
                self._swa_local_layer_id(layer_id),
                swa_loc,
                kv,
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                write_mask=self._dcp_loc_write_mask(
                    raw_loc if raw_loc is not None else swa_loc
                ),
            )
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],
            page_size=self.swa_kv_pool.page_size,
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        dcp_kv_mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        compress_ratio, compress_layer_id, compress_kv_pool = self.layer_mapping[
            layer_id
        ]
        assert compress_kv_pool is not None
        dcp_world_size, dcp_rank, dcp_enabled = self._dcp_write_context()

        if dcp_enabled or dcp_kv_mask is not None or write_mask is not None:
            return compress_kv_pool.set_key_buffer_fused_fallback_triton(
                compress_layer_id,
                loc,
                cache_k,
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                write_mask=write_mask.contiguous() if write_mask is not None else None,
            )
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        dcp_kv_mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        if write_mask is not None:
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

            index_k, index_k_scale = act_quant(cache_k)
            return self.c4_indexer_kv_pool.set_index_k_scale_buffer(
                compress_layer_id,
                loc,
                index_k,
                index_k_scale,
                write_mask=write_mask.contiguous(),
            )
        # See set_index_k_scale_buffer: the indexer KV must stay global because
        # the logits path reads it through the global c4 page_table.
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fp4(compress_layer_id, loc, cache_k)

    def _compressed_indices_from_full_indices(
        self, indices: torch.Tensor, compress_ratio: int
    ) -> torch.Tensor:
        if len(indices) == 0:
            return indices
        positions = torch.arange(
            1, len(indices) + 1, dtype=torch.long, device=indices.device
        )
        mask = (positions % compress_ratio) == 0
        return (indices[mask] // compress_ratio).to(indices.dtype)

    def _copy_state_pool_to_cpu(
        self, pool: Optional[CompressStatePool], indices: torch.Tensor
    ):
        if (
            pool is None
            or self.full_to_swa_index_mapping is None
            or len(indices) == 0
        ):
            return None
        swa_indices = self.full_to_swa_index_mapping[indices]
        mask = swa_indices >= 0
        if not torch.any(mask):
            return None
        state_locs = pool.translate_from_swa_loc_to_state_loc(swa_indices[mask])
        bank_offsets = [0]
        if pool.online_mtp_max_draft_tokens > 0:
            bank_offsets = [
                i * pool.online_mtp_state_slot_offset
                for i in range(pool.online_mtp_max_draft_tokens + 1)
            ]
        cpu_banks = [
            pool.kv_score_buffer.kv_score[state_locs + offset].to(
                "cpu", non_blocking=True
            )
            for offset in bank_offsets
        ]
        return {
            "mask": mask.cpu(),
            "bank_offsets": bank_offsets,
            "kv_score": cpu_banks,
        }

    def _load_state_pool_from_cpu(
        self,
        pool: Optional[CompressStatePool],
        state_cpu,
        indices: torch.Tensor,
    ) -> None:
        if (
            pool is None
            or state_cpu is None
            or self.full_to_swa_index_mapping is None
            or len(indices) == 0
        ):
            return
        old_mask = state_cpu["mask"].to(indices.device)
        if not torch.any(old_mask):
            return
        swa_indices = self.full_to_swa_index_mapping[indices]
        new_mask = swa_indices >= 0
        row_mask = new_mask[old_mask]
        if not torch.any(row_mask):
            return
        state_locs = pool.translate_from_swa_loc_to_state_loc(
            swa_indices[old_mask][row_mask]
        )
        for bank_cpu, offset in zip(
            state_cpu["kv_score"], state_cpu["bank_offsets"]
        ):
            pool.kv_score_buffer.kv_score[state_locs + offset] = bank_cpu[
                row_mask.cpu()
            ].to(pool.kv_score_buffer.kv_score.device, non_blocking=True)
        pool.kv_score_buffer[-1].clear()

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        swa_cpu = None
        swa_mask = None
        if self.full_to_swa_index_mapping is not None and len(indices) > 0:
            swa_indices = self.full_to_swa_index_mapping[indices]
            swa_mask = swa_indices >= 0
            if torch.any(swa_mask):
                swa_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices[swa_mask])
                swa_mask = swa_mask.cpu()

        c4_indices = self._compressed_indices_from_full_indices(indices, 4)
        c128_indices = self._compressed_indices_from_full_indices(indices, 128)
        c4_cpu = (
            self.c4_kv_pool.get_cpu_copy(c4_indices)
            if len(c4_indices) > 0
            else None
        )
        c128_cpu = (
            self.c128_kv_pool.get_cpu_copy(c128_indices)
            if len(c128_indices) > 0
            else None
        )
        c4_indexer_cpu = (
            self.c4_indexer_kv_pool.get_cpu_copy(c4_indices)
            if len(c4_indices) > 0
            else None
        )
        state_cpu = [
            self._copy_state_pool_to_cpu(pool, indices)
            for pool in self.compress_state_pools
        ]
        indexer_state_cpu = [
            self._copy_state_pool_to_cpu(pool, indices)
            for pool in self.indexer_compress_state_pools
        ]
        current_platform.synchronize()
        return {
            "swa": swa_cpu,
            "swa_mask": swa_mask,
            "c4": c4_cpu,
            "c4_indices_len": len(c4_indices),
            "c128": c128_cpu,
            "c128_indices_len": len(c128_indices),
            "c4_indexer": c4_indexer_cpu,
            "state": state_cpu,
            "indexer_state": indexer_state_cpu,
        }

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        swa_cpu = kv_cache_cpu["swa"]
        if swa_cpu is not None and self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            new_swa_mask = swa_indices >= 0
            old_swa_mask = kv_cache_cpu.get("swa_mask")
            if old_swa_mask is not None:
                old_swa_mask = old_swa_mask.to(indices.device)
                row_mask = new_swa_mask[old_swa_mask].cpu()
                swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
            else:
                row_mask = new_swa_mask.cpu()
                swa_indices = swa_indices[new_swa_mask]
            if swa_indices.numel() > 0:
                swa_cpu = self._filter_layer_chunks(swa_cpu, row_mask)
                self.swa_kv_pool.load_cpu_copy(swa_cpu, swa_indices)

        c4_indices = self._compressed_indices_from_full_indices(indices, 4)
        c128_indices = self._compressed_indices_from_full_indices(indices, 128)
        if kv_cache_cpu["c4"] is not None and len(c4_indices) > 0:
            c4_indices = c4_indices[: kv_cache_cpu["c4_indices_len"]]
            self.c4_kv_pool.load_cpu_copy(kv_cache_cpu["c4"], c4_indices)
        if kv_cache_cpu["c4_indexer"] is not None and len(c4_indices) > 0:
            c4_indices = c4_indices[: kv_cache_cpu["c4_indices_len"]]
            self.c4_indexer_kv_pool.load_cpu_copy(
                kv_cache_cpu["c4_indexer"], c4_indices
            )
        if kv_cache_cpu["c128"] is not None and len(c128_indices) > 0:
            c128_indices = c128_indices[: kv_cache_cpu["c128_indices_len"]]
            self.c128_kv_pool.load_cpu_copy(kv_cache_cpu["c128"], c128_indices)

        for pool, state_cpu in zip(self.compress_state_pools, kv_cache_cpu["state"]):
            self._load_state_pool_from_cpu(pool, state_cpu, indices)
        for pool, state_cpu in zip(
            self.indexer_compress_state_pools, kv_cache_cpu["indexer_state"]
        ):
            self._load_state_pool_from_cpu(pool, state_cpu, indices)
        current_platform.synchronize()

    def _filter_layer_chunks(self, kv_cpu, row_mask: torch.Tensor):
        if kv_cpu is None:
            return None
        if row_mask is None or bool(torch.all(row_mask).item()):
            return kv_cpu
        chunk_size = self.cpu_offloading_chunk_size
        filtered = []
        for layer_chunks in kv_cpu:
            if len(layer_chunks) == 0:
                filtered.append([])
                continue
            values_cpu = torch.cat([chunk[0] for chunk in layer_chunks], dim=0)
            scales_cpu = torch.cat([chunk[1] for chunk in layer_chunks], dim=0)
            values_cpu = values_cpu[row_mask]
            scales_cpu = scales_cpu[row_mask]
            filtered_layer = []
            for i in range(0, len(values_cpu), chunk_size):
                filtered_layer.append(
                    (values_cpu[i : i + chunk_size], scales_cpu[i : i + chunk_size])
                )
            filtered.append(filtered_layer)
        return filtered

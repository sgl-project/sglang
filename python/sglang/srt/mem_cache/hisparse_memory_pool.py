from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.kvcache.cache_move import copy_all_layer_kv_cache_func
from sglang.kernels.ops.kvcache.hisparse import (
    load_cache_to_device_buffer_mla,
)
from sglang.kernels.ops.kvcache.hisparse_slot_mapping import (
    translate_padded_hisparse_locations,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.utils import is_cuda, is_hip, is_xpu

# mapping on device memory, host memory and memory allocator


logger = logging.getLogger(__name__)

# sgl_kernel.kvcacheio is only available in CUDA/ROCm/XPU sgl-kernel builds (not MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio "
            "(CUDA/ROCm/XPU). It is not available on this backend."
        )


class HiSparseDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_kpool: int = 1,
        index_kpool_compress: bool = False,
        tail_extra_slots: int = 0,
        max_running_requests: Optional[int] = None,
        skip_topk_layers: Optional[list[bool]] = None,
        host_to_device_ratio: int = 2,
        kv_scale_ue8m0: bool = False,
        use_fp4_indexer: bool = False,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=size * host_to_device_ratio,
            index_kpool=index_kpool,
            index_kpool_compress=index_kpool_compress,
            tail_extra_slots=tail_extra_slots,
            max_running_requests=max_running_requests,
            skip_topk_layers=skip_topk_layers,
            kv_scale_ue8m0=kv_scale_ue8m0,
            use_fp4_indexer=use_fp4_indexer,
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize
        self.data_strides = torch.full_like(self.data_ptrs, self.bytes_per_token)

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(
        self, compressed_indices: torch.Tensor
    ) -> torch.Tensor:
        """Map logical locations to physical slots with the same shape.

        CUDA and ROCm use a fused kernel for 1D GPU slot lists, preserving
        negative padding. Page tables and CPU inputs keep the direct gather.
        """
        if compressed_indices.is_cuda and compressed_indices.ndim == 1:
            return translate_padded_hisparse_locations(
                self.full_to_hisparse_device_index_mapping, compressed_indices
            )
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(full_indices)

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        return full_indices

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(
            layer,
            loc,
            cache_k_nope,
            cache_k_rope,
            layer_id_override=layer_id_override,
        )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(
            layer, loc, dst_dtype, layer_id_override=layer_id_override
        )

    def transfer_values_on_device(
        self, dst_indices: torch.Tensor, src_indices: torch.Tensor
    ) -> None:
        """Copy target MLA KV between physical slots, without remapping IDs."""
        if src_indices.numel() == 0:
            return
        config = {
            "bytes_per_tile": 128,
            "byte_tiles": (self.bytes_per_token + 127) // 128,
            "num_warps": 4,
        }
        for start in range(0, src_indices.numel(), 256):
            src = src_indices[start : start + 256].contiguous()
            dst = dst_indices[start : start + 256].contiguous()
            copy_all_layer_kv_cache_func(
                self.data_ptrs,
                self.data_strides,
                dst,
                src,
                src.numel(),
                1 << (src.numel() - 1).bit_length(),
                config,
            )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")


class HiSparseMHAMainPool(MHATokenToKVPool):
    """MHA KV pool with HiSparse logical-to-device mapping.

    Used by MiniMax M3 HiSparse. The index pools (index_kv_pool, index_k_pool)
    stay fully resident on the device and do not use this mapping.
    """

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
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.full_to_hisparse_device_index_mapping: Optional[torch.Tensor] = None
        self.bytes_per_token_k = head_num * head_dim * self.store_dtype.itemsize
        self.bytes_per_token_v = head_num * self.v_head_dim * self.store_dtype.itemsize

    def register_mapping(
        self, full_to_hisparse_device_index_mapping: torch.Tensor
    ) -> None:
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_hisparse_device_index_mapping is not None
        return self.full_to_hisparse_device_index_mapping[indices]

    def _translate_loc_to_hisparse_device(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_hisparse_device_index_mapping is not None
        return self.full_to_hisparse_device_index_mapping[indices]

    def translate_loc_from_full_to_hisparse_device(
        self, full_indices: torch.Tensor
    ) -> torch.Tensor:
        assert self.full_to_hisparse_device_index_mapping is not None
        return self.full_to_hisparse_device_index_mapping[full_indices]

    def translate_loc_from_full_to_compressed(
        self, full_indices: torch.Tensor
    ) -> torch.Tensor:
        return full_indices

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        *args,
        **kwargs,
    ):
        raw_loc, _, _ = unwrap_write_loc(loc)
        translated = self.translate_loc_to_hisparse_device(raw_loc)
        super().set_kv_buffer(layer, translated, cache_k, cache_v, *args, **kwargs)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("HiSparseMHAMainPool does not support get_cpu_copy")

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError("HiSparseMHAMainPool does not support load_cpu_copy")


class HiSparseHiCacheBuffer:
    @torch.inference_mode(False)
    def __init__(
        self,
        *,
        req_to_token_pool,
        token_to_kv_pool,
        top_k: int,
        buffer_tokens: int,
        block_size: int,
        device: str,
    ):
        self.pool = token_to_kv_pool
        self.req_to_token = req_to_token_pool.req_to_token
        self.page_size = self.pool.page_size
        self.top_k = top_k
        self.buffer_tokens = buffer_tokens
        self.block_size = block_size
        self.device = device
        self.reserved_buffer_tokens = buffer_tokens
        buffer_columns = buffer_tokens + 1
        rows, context = self.req_to_token.shape
        layers = self.pool.layer_num
        if buffer_tokens < top_k or buffer_tokens % self.page_size:
            raise ValueError(
                "HiSparse HiCache backend buffer must be page-aligned and cover top_k"
            )
        if buffer_tokens > 32767:
            raise ValueError("HiSparse HiCache backend LRU slots must fit int16")
        self.index_buffers = [
            b for b in self.pool.index_k_with_scale_buffer if b.shape[0] > 0
        ]
        self.base_pages = self.pool.size // self.page_size + 1
        self.index_pages = self.index_buffers[0].shape[0]
        if self.index_pages <= self.base_pages:
            raise ValueError(
                "HiSparse HiCache backend requires expanded indexer storage"
            )
        self.host_locs = torch.full(
            (rows, context), -1, dtype=torch.int64, device=device
        )
        self.protected_lens = torch.zeros(rows, dtype=torch.int64, device=device)
        self.temp_locs = torch.full(
            (layers, rows, buffer_columns), -1, dtype=torch.int32, device=device
        )
        self.residency = torch.full(
            (layers, rows, buffer_columns),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(buffer_tokens, dtype=torch.int16, device=device)
        self.lru = self._lru_init.expand(layers, rows, -1).clone()
        self.result = torch.full((rows, top_k), -1, dtype=torch.int32, device=device)
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        # HiCache attaches after decode graphs are captured.
        self.host_bindings = torch.zeros((layers, 2), dtype=torch.int64, device=device)
        self.host_pool = None

    def attach_host_pool(self, host_pool, *, layer_offset: int = 0) -> None:
        if self.base_pages + host_pool.size // self.page_size > self.index_pages:
            raise ValueError(
                "HiSparse HiCache backend index expansion must cover the HiCache host pool; "
                "increase expansion_ratio or reduce the host pool size"
            )
        refs = host_pool.data_refs[layer_offset : layer_offset + self.pool.layer_num]
        if len(refs) != self.pool.layer_num:
            raise ValueError("HiSparse HiCache backend host/device layer counts differ")
        bindings = []
        for host, device in zip(refs, self.pool.kv_buffer):
            host_u8, device_u8 = host.view(torch.uint8), device.view(torch.uint8)
            if not host.is_pinned() or host_u8.stride(0) < device_u8.stride(0):
                raise ValueError(
                    "HiSparse HiCache backend needs pinned host rows covering device KV"
                )
            bindings.append((host_u8.data_ptr(), host_u8.stride(0)))
        self.host_bindings.copy_(
            torch.tensor(bindings, dtype=torch.int64, device=self.device)
        )
        self.host_pool = host_pool

    def prepare_forward(self, forward_batch) -> None:
        if self.num_real_reqs.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        self.num_real_reqs.fill_(
            0 if forward_batch.forward_mode.is_idle() else forward_batch.batch_size
        )

    def assign_temp_locations(self, req_id: int, temp: torch.Tensor) -> None:
        self.temp_locs[:, req_id, : self.buffer_tokens] = temp[: self.buffer_tokens].to(
            torch.int32
        )

    def preserve_index(self, device_tokens: torch.Tensor, host_tokens: torch.Tensor):
        source = device_tokens[:: self.page_size] // self.page_size
        destination = self.base_pages + host_tokens[:: self.page_size] // self.page_size
        # Host locks keep these rows alive for every request sharing the prefix.
        for buffer in self.index_buffers:
            buffer[destination] = buffer[source]

    def release(self, req_id: int) -> None:
        self.protected_lens[req_id] = 0
        self.host_locs[req_id].fill_(-1)
        self.temp_locs[:, req_id].fill_(-1)
        self.residency[:, req_id].fill_(-1)
        self.lru[:, req_id].copy_(self._lru_init)

    def indexer_page_table(
        self, *, req_pool_indices: torch.Tensor, num_pages: int
    ) -> torch.Tensor:
        req_ids = req_pool_indices.long()
        tokens = self.req_to_token[
            req_ids, : num_pages * self.page_size : self.page_size
        ]
        hosts = self.host_locs[req_ids, : num_pages * self.page_size : self.page_size]
        host_pages = torch.where(
            hosts >= 0, self.base_pages + hosts // self.page_size, -1
        )
        pages = torch.where(tokens < 0, host_pages, tokens // self.page_size)
        starts = torch.arange(num_pages, device=self.device) * self.page_size
        protected = starts[None, :] < self.protected_lens[req_ids, None]
        valid = (pages >= 0) & (pages < self.index_pages)
        torch._assert_async(
            (~protected | valid).all(),
            "HiSparse HiCache backend lost an active index page",
        )
        return torch.where(valid, pages, 0).to(torch.int32)

    def swap_in_selected_pages(
        self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id
    ):
        local_layer = layer_id - self.pool.start_layer
        num_reqs = req_pool_indices.numel()
        num_queries = top_k_result.shape[0]
        num_query_steps = num_queries // num_reqs if num_reqs else 1
        if num_query_steps > 1:
            raise ValueError(
                "HiSparse HiCache backend query count exceeds its reserved workspace"
            )
        result = self.result[:num_queries]
        device_buffer = self.pool.kv_buffer[local_layer]
        load_cache_to_device_buffer_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.residency[local_layer],
            host_cache_locs=self.host_locs,
            device_buffer_locs=self.temp_locs[local_layer],
            host_cache=None,
            device_buffer=device_buffer,
            top_k_device_locs=result,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru[local_layer],
            item_size_bytes=device_buffer.stride(0) * device_buffer.element_size(),
            num_top_k=self.top_k,
            hot_buffer_size=self.buffer_tokens,
            page_size=1,
            block_size=self.block_size,
            num_real_reqs=self.num_real_reqs,
            device_locs=self.req_to_token,
            host_binding=self.host_bindings[local_layer],
        )
        return result

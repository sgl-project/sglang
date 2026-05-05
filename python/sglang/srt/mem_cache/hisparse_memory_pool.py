import logging
import weakref
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import get_num_new_pages

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


logger = logging.getLogger(__name__)


class HiSparseNSATokenToKVPool(NSATokenToKVPool):
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
        host_to_device_ratio: int = 2,
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
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: NSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.need_sort = need_sort

        self.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_full,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )

        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.page_size,
            self.dtype,
            self.device,
            kvcache,
            need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._size_full + self.page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self._kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def size_full(self) -> int:
        return self._size_full

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "Page size = 1 is not supported in HiSparse allocator"
        )

    def alloc_logical_only(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate only logical indices without hisparse device indices.

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging.
        """
        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        if logical_indices is not None:
            self.full_to_hisparse_device_index_mapping[logical_indices] = 0
        return logical_indices

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        # Filter valid hisparse indices. Page 0 is reserved by the paged
        # allocator, so indices below page_size must never be reused or freed.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        hisparse_indices = hisparse_indices[hisparse_indices >= self.page_size]
        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            self.free_hisparse_indices(hisparse_indices[need_size:])
        else:
            # page alignment, claiming the residual space for an incomplete page
            page_residual_length = len(hisparse_indices) % self.page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.page_size
                            - page_residual_length
                            + 1,
                            device=self.device,
                        ),
                    ]
                )
            extra_indices = self.hisparse_attn_allocator.alloc(
                need_size - len(hisparse_indices)
            )
            assert (
                extra_indices is not None
            ), "Hisparse allocation failed in alloc_device_buffer"
            buffer_indices = torch.cat([hisparse_indices, extra_indices])

        # CRITICAL: Map buffer indices back to logical indices for the used portion.
        # This ensures free_hisparse() can find the mapping when release_kv_cache
        # is called, preventing memory leaks from page alignment mismatches.
        map_len = min(need_size, len(allocated_indices))
        if map_len > 0:
            self.full_to_hisparse_device_index_mapping[allocated_indices[:map_len]] = (
                buffer_indices[:map_len]
            )

        return buffer_indices

    def free_hisparse_indices(
        self, buffer_indices: torch.Tensor, source: str = "unknown"
    ):
        # Request teardown can observe stale aliases (for example reserved newest
        # slot / speculative draft slots) that point multiple logical tokens to
        # the same physical HiSparse buffer slot. Free each physical slot once.
        valid_indices = buffer_indices[buffer_indices >= self.page_size]
        if valid_indices.numel() == 0:
            return
        free_page_indices = torch.unique(valid_indices // self.page_size, sorted=False)

        already_free_mask = torch.zeros_like(free_page_indices, dtype=torch.bool)
        if self.hisparse_attn_allocator.free_pages.numel() > 0:
            already_free_mask |= torch.isin(
                free_page_indices, self.hisparse_attn_allocator.free_pages
            )
        if self.hisparse_attn_allocator.release_pages.numel() > 0:
            already_free_mask |= torch.isin(
                free_page_indices, self.hisparse_attn_allocator.release_pages
            )
        if torch.any(already_free_mask):
            already_free_pages = free_page_indices[already_free_mask]
            logger.warning(
                "HiSparse double-free candidate skipped: pages already free before "
                "release. source=%s pages=%s physical_indices_sample=%s total_physical=%d",
                source,
                already_free_pages[:16].tolist(),
                valid_indices[:32].tolist(),
                int(valid_indices.numel()),
            )
            free_page_indices = free_page_indices[~already_free_mask]
            if free_page_indices.numel() == 0:
                return

        # PagedTokenToKVPoolAllocator.free() releases whole pages from any token
        # index in that page. Pass one representative token per page explicitly so
        # HiSparse cleanup stays page-granular and idempotent.
        valid_indices = free_page_indices * self.page_size

        # disable free group mechanism for device buffer free
        self.hisparse_attn_allocator.is_not_in_free_group = True
        self.hisparse_attn_allocator.free(valid_indices)

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        hisparse_last_locs = self._kvcache._translate_loc_to_hisparse_device(last_locs)
        return hisparse_last_locs

    def alloc_extend_with_device_mapping(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        device_slots: torch.Tensor,
        backup_state: bool = False,
    ):
        """Allocate logical indices and map them to hisparse device slots atomically.

        Combines logical allocation + device mapping into one call to prevent
        callers from forgetting the mapping step (which causes silent corruption).
        """
        # Pre-flight capacity check to avoid launching a Triton kernel
        # only to discover there aren't enough free pages.
        avail = self.logical_attn_allocator.available_size()
        if avail < extend_num_tokens:
            raise RuntimeError(
                f"HiSparse logical alloc: need {extend_num_tokens} tokens but only "
                f"{avail} available"
            )
        state = self.backup_state() if backup_state else None
        out = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        if out is None:
            raise RuntimeError(
                f"HiSparse logical alloc failed for {extend_num_tokens} tokens. "
                f"Logical pool available: {self.logical_attn_allocator.available_size()}"
            )
        self.full_to_hisparse_device_index_mapping[out] = device_slots
        if backup_state:
            return out, state
        return out

    def clear_device_mapping(self, logical_indices: torch.Tensor):
        """Clear hisparse device mapping. Must be called before free() for
        indices whose device slots were not allocated from hisparse_attn_allocator,
        otherwise free_hisparse() would corrupt the hisparse allocator's free list."""
        self.full_to_hisparse_device_index_mapping[logical_indices] = 0

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if (
            num_new_pages
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages
            > self.hisparse_attn_allocator.available_size() // self.page_size
        ):
            return None

        logical_indices = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert logical_indices is not None, "Logical allocation failed in alloc_extend"

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        invalid_partial_prefix = (hisparse_last_loc < self.page_size) & (
            prefix_lens % self.page_size != 0
        )
        if torch.any(invalid_partial_prefix):
            # Direct-to-host requests can have logical prefix pages without a
            # matching device page. Allocate fresh HiSparse pages for the new
            # target/draft tokens instead of extending from reserved page 0.
            zero_prefix_lens = torch.zeros_like(prefix_lens)
            zero_prefix_lens_cpu = torch.zeros_like(prefix_lens_cpu)
            extend_lens = seq_lens - prefix_lens
            extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu
            hisparse_last_loc = torch.full_like(last_loc, -1)
            hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
                zero_prefix_lens,
                zero_prefix_lens_cpu,
                extend_lens,
                extend_lens_cpu,
                hisparse_last_loc,
                len(logical_indices),
            )
            assert (
                hisparse_indices is not None
            ), "Hisparse allocation failed in alloc_extend"
            self.full_to_hisparse_device_index_mapping[logical_indices] = (
                hisparse_indices
            )
            return logical_indices

        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
            len(logical_indices),
            num_new_pages=num_new_pages,
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        return logical_indices

    def alloc_decode_debug(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        logical_indices = self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_decode(
            seq_lens,
            seq_lens_cpu,
            hisparse_last_loc,
        )

        if logical_indices is None or hisparse_indices is None:
            return None

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices

        return logical_indices

    def free_hisparse(self, free_indices: torch.Tensor):
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_indices)
        hisparse_indices = hisparse_indices[hisparse_indices >= self.page_size]
        if hisparse_indices.numel() > 0:
            logical_pages = torch.unique(free_indices // self.page_size)
            physical_pages = torch.unique(hisparse_indices // self.page_size)
            if logical_pages.numel() != physical_pages.numel():
                logger.warning(
                    "HiSparse free logical/physical page fanout mismatch. "
                    "logical_pages=%d physical_pages=%d logical_indices_sample=%s physical_indices_sample=%s",
                    int(logical_pages.numel()),
                    int(physical_pages.numel()),
                    free_indices[:32].tolist(),
                    hisparse_indices[:32].tolist(),
                )
        self.free_hisparse_indices(hisparse_indices, source="logical_release")
        self.full_to_hisparse_device_index_mapping[free_indices] = 0

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def backup_state(self):
        return [
            self.logical_attn_allocator.backup_state(),
            self.hisparse_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        logical_state = state[0]
        restored_free_pages = torch.cat(logical_state)
        current_free_pages = torch.cat(self.logical_attn_allocator.backup_state())
        page_lookup = torch.zeros(
            self.logical_attn_allocator.num_pages + 1,
            dtype=torch.bool,
            device=self.device,
        )
        page_lookup[current_free_pages] = True
        restored_pages = restored_free_pages[~page_lookup[restored_free_pages]]
        if restored_pages.numel() > 0:
            restored_indices = (
                restored_pages[:, None] * self.page_size
                + torch.arange(self.page_size, device=self.device)
            ).reshape(-1)
            self.full_to_hisparse_device_index_mapping[restored_indices] = 0

        self.logical_attn_allocator.restore_state(state[0])
        self.hisparse_attn_allocator.restore_state(state[1])

    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
            self.free_hisparse(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.logical_attn_allocator.available_size()
            <= self.logical_attn_allocator.size
        )
        assert (
            self.hisparse_attn_allocator.available_size()
            <= self.hisparse_attn_allocator.size
        )

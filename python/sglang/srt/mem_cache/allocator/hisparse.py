import weakref

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.utils.common import get_num_new_pages


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kvcache: HiSparseDSATokenToKVPool,
        need_sort: bool,
        host_to_device_ratio: int = 2,
    ):
        self._kvcache = kvcache
        self._size_full = size * host_to_device_ratio
        self._size_hisparse = size
        self.compress_ratio = 1
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

    @property
    def size(self) -> int:
        return self._size_full

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int) -> torch.Tensor | None:
        assert need_size % self.page_size == 0, (
            f"HiSparse alloc expects page-aligned size: {need_size=}, "
            f"page_size={self.page_size}"
        )

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None

        hisparse_indices = self.hisparse_attn_allocator.alloc(need_size)
        if hisparse_indices is None:
            self.logical_attn_allocator.free(logical_indices)
            return None

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_logical_only(self, need_size: int) -> torch.Tensor | None:
        """Allocate only logical indices without hisparse device indices.

        Used in the direct-to-host transfer path where KV data is written
        directly to host memory by the prefill node, skipping GPU staging.
        """
        assert need_size % self.page_size == 0, (
            f"HiSparse alloc_logical_only expects page-aligned size: {need_size=}, "
            f"page_size={self.page_size}"
        )
        return self.logical_attn_allocator.alloc(need_size)

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
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
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor) -> None:
        self.hisparse_attn_allocator.free_pages_by_any_member_legacy(
            buffer_indices[buffer_indices > 0]
        )

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return last_locs

    def free_hisparse(self, free_indices: torch.Tensor):
        hisparse_indices = self._kvcache._translate_loc_to_hisparse_device(free_indices)
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[free_indices] = 0

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return

        if not self.uses_legacy_real_length_alloc:
            assert free_index.numel() % self.page_size == 0, (
                f"HiSparse free expects whole-page input: {free_index.numel()=}, "
                f"page_size={self.page_size}"
            )

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


class DeepSeekV4HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):

    def __init__(
        self,
        logical_attn_allocator: BaseTokenToKVPoolAllocator,
    ):
        assert isinstance(logical_attn_allocator._kvcache, DeepSeekV4TokenToKVPool)
        assert isinstance(
            logical_attn_allocator._kvcache.c4_kv_pool, HiSparseC4DevicePool
        )
        self.compress_ratio = 4

        self.hisparse_kvcache = logical_attn_allocator._kvcache.c4_kv_pool
        self._size_full = logical_attn_allocator.size_full
        self._size_hisparse = self.hisparse_kvcache.size

        self.dtype = self.hisparse_kvcache.dtype
        self.device = self.hisparse_kvcache.device
        # Keep the public page_size as the logical DSV4 full/SWA page size.
        # C4 HiSparse allocation/device-buffer code must use the compressed page size.
        self.page_size = logical_attn_allocator.page_size
        self.hisparse_page_size = self.hisparse_kvcache.page_size

        self.logical_attn_allocator = logical_attn_allocator
        self._kvcache = logical_attn_allocator._kvcache
        self.hisparse_attn_allocator = PagedTokenToKVPoolAllocator(
            self._size_hisparse,
            self.hisparse_page_size,
            self.dtype,
            self.device,
            self.hisparse_kvcache,
            logical_attn_allocator.need_sort,
        )

        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(
                    self._kvcache.c4_logical_size + self.hisparse_page_size,
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=self.device),
            ]
        )

        self.need_sort = logical_attn_allocator.need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        self.hisparse_kvcache.register_mapping(
            weakref.proxy(self.full_to_hisparse_device_index_mapping)
        )

    @property
    def uses_legacy_real_length_alloc(self) -> bool:
        return self.logical_attn_allocator.uses_legacy_real_length_alloc

    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size(self) -> int:
        return self.logical_attn_allocator.size

    @property
    def size_swa(self) -> int:
        return self.logical_attn_allocator.size_swa

    @property
    def full_to_swa_index_mapping(self):
        return self.logical_attn_allocator.full_to_swa_index_mapping

    def debug_print(self) -> str:
        msg = self.logical_attn_allocator.debug_print()
        msg += (
            f"#hisparse-available-size: "
            f"{self.hisparse_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return self.logical_attn_allocator.translate_loc_from_full_to_swa(kv_indices)

    def full_available_size(self):
        return min(
            self.logical_attn_allocator.full_available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def swa_available_size(self):
        return self.logical_attn_allocator.swa_available_size()

    def free_swa(self, free_indices: torch.Tensor):
        self.logical_attn_allocator.free_swa(free_indices)

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def alloc(self, need_size: int) -> torch.Tensor | None:
        assert need_size % self.page_size == 0, (
            f"DSV4 HiSparse alloc expects page-aligned size: {need_size=}, "
            f"page_size={self.page_size}"
        )

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None

        compressed_logical_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )
        assert len(compressed_logical_indices) == need_size // self.compress_ratio

        hisparse_indices = self.hisparse_attn_allocator.alloc(
            len(compressed_logical_indices)
        )
        if hisparse_indices is None:
            self.logical_attn_allocator.free(logical_indices)
            return None

        self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = (
            hisparse_indices
        )
        return logical_indices

    def alloc_logical_only(self, need_size: int) -> torch.Tensor | None:
        assert need_size % self.page_size == 0, (
            f"DSV4 HiSparse alloc_logical_only expects page-aligned size: "
            f"{need_size=}, page_size={self.page_size}"
        )
        return self.logical_attn_allocator.alloc(need_size)

    def alloc_logical_only_legacy(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        """Allocate decode logical indices without allocating C4 hisparse device pages."""
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_extend_swa_tail(
        self, *, seq_len: int, swa_tail_len: int
    ) -> torch.Tensor | None:
        return self.logical_attn_allocator.alloc_extend_swa_tail(
            seq_len=seq_len,
            swa_tail_len=swa_tail_len,
        )

    def alloc_extend_swa_tail_legacy(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        swa_tail_len: int,
    ):
        return self.logical_attn_allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
            swa_tail_len=swa_tail_len,
        )

    def alloc_device_buffer(self, allocated_indices, need_size: int):
        assert need_size % self.hisparse_page_size == 0
        hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
        self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
        hisparse_indices = hisparse_indices[hisparse_indices > 0]

        device_buffer_size = need_size - self.hisparse_page_size
        P = len(hisparse_indices)
        if P > device_buffer_size + 1:
            newest_src = hisparse_indices[P - 1].clone()
            old_at_dbs = hisparse_indices[device_buffer_size].clone()
            hisparse_indices[device_buffer_size] = newest_src
            hisparse_indices[P - 1] = old_at_dbs

        if len(hisparse_indices) >= need_size:
            buffer_indices = hisparse_indices[:need_size]
            surplus = hisparse_indices[need_size:]
            if surplus.numel() > 0:
                buffer_pages = torch.unique(buffer_indices // self.hisparse_page_size)
                surplus_pages = torch.unique(surplus // self.hisparse_page_size)
                pure_surplus = surplus_pages[~torch.isin(surplus_pages, buffer_pages)]
                if pure_surplus.numel() > 0:
                    self.hisparse_attn_allocator.free_pages_by_any_member_legacy(
                        pure_surplus * self.hisparse_page_size
                    )
        else:
            page_residual_length = len(hisparse_indices) % self.hisparse_page_size
            if page_residual_length != 0:
                hisparse_indices = torch.cat(
                    [
                        hisparse_indices,
                        torch.arange(
                            hisparse_indices[-1] + 1,
                            hisparse_indices[-1]
                            + self.hisparse_page_size
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
        return buffer_indices

    def free_hisparse_indices(self, buffer_indices: torch.Tensor) -> None:
        self.hisparse_attn_allocator.free_pages_by_any_member_legacy(
            buffer_indices[buffer_indices > 0]
        )

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return (last_locs - 3) // self.compress_ratio

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self.hisparse_kvcache._translate_loc_to_hisparse_device(
            self.get_last_loc_compressed(last_locs)
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages_logical = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_new_pages_hisparse = get_num_new_pages(
            seq_lens=seq_lens_cpu // self.compress_ratio,
            page_size=self.hisparse_page_size,
            prefix_lens=prefix_lens_cpu // self.compress_ratio,
        )
        if (
            num_new_pages_logical
            > self.logical_attn_allocator.available_size() // self.page_size
        ):
            return None
        if (
            num_new_pages_hisparse
            > self.hisparse_attn_allocator.available_size() // self.hisparse_page_size
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

        compressed_logical_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )
        hisparse_last_loc = self.get_last_loc_hisparse_device(last_loc)
        hisparse_indices = self.hisparse_attn_allocator.alloc_extend(
            prefix_lens // self.compress_ratio,
            prefix_lens_cpu // self.compress_ratio,
            seq_lens // self.compress_ratio,
            seq_lens_cpu // self.compress_ratio,
            hisparse_last_loc,
            len(compressed_logical_indices),
        )
        assert (
            hisparse_indices is not None
        ), "Hisparse allocation failed in alloc_extend"

        self.full_to_hisparse_device_index_mapping[compressed_logical_indices] = (
            hisparse_indices.to(torch.int64)
        )
        return logical_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_compressed(self, compressed_indices: torch.Tensor):
        hisparse_indices = self.hisparse_kvcache.translate_loc_to_hisparse_device(
            compressed_indices
        )
        hisparse_indices = hisparse_indices[hisparse_indices > 0]
        self.free_hisparse_indices(hisparse_indices)
        self.full_to_hisparse_device_index_mapping[compressed_indices] = 0

    def free_hisparse(self, free_indices: torch.Tensor):
        compressed_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(free_indices)
        )
        self.free_compressed(compressed_indices)

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()

        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return

        if not self.uses_legacy_real_length_alloc:
            assert free_index.numel() % self.page_size == 0, (
                f"DSV4 HiSparse free expects whole-page input: {free_index.numel()=}, "
                f"page_size={self.page_size}"
            )

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
        else:
            self.free_group.append(free_index)

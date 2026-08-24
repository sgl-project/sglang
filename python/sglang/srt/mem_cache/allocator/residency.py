"""Paged allocator with disjoint replicated and DCP-striped KV regions."""

from __future__ import annotations

import torch

from sglang.srt.attn_parallel import KvResidency
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator


class ResidencyAwarePagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Partition one physical MLA pool into two residency namespaces.

    Replicated locs directly address the first physical region. Striped locs
    use widened virtual IDs whose division by ``dcp_size`` addresses the second
    region. Both allocators expose the same logical page size, so one radix tree
    can safely hold entries from both namespaces when ``extra_key`` carries the
    residency tag.
    """

    def __init__(
        self,
        *,
        physical_size: int,
        physical_page_size: int,
        dcp_size: int,
        replicated_fraction: float,
        dtype: torch.dtype,
        device: str,
        kvcache,
        need_sort: bool,
    ):
        if dcp_size <= 1:
            raise ValueError("residency-aware allocator requires dcp_size > 1")
        if not 0.0 < replicated_fraction < 1.0:
            raise ValueError("replicated_fraction must be in (0, 1)")

        logical_page_size = physical_page_size * dcp_size
        replicated_size = (
            int(physical_size * replicated_fraction) // logical_page_size
        ) * logical_page_size
        striped_physical_size = (
            (physical_size - replicated_size) // physical_page_size
        ) * physical_page_size
        if (
            replicated_size < logical_page_size
            or striped_physical_size < physical_page_size
        ):
            raise ValueError("KV pool is too small for both residency regions")

        super().__init__(
            size=replicated_size + striped_physical_size * dcp_size,
            page_size=logical_page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.dcp_size = dcp_size
        self.physical_page_size = physical_page_size
        self.logical_page_size = logical_page_size
        self.physical_size = replicated_size + striped_physical_size
        self.replicated_size = replicated_size
        self.striped_physical_size = striped_physical_size
        self.active_residency = KvResidency.REPLICATED

        self.replicated = PagedTokenToKVPoolAllocator(
            size=replicated_size,
            page_size=logical_page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.striped = PagedTokenToKVPoolAllocator(
            size=striped_physical_size * dcp_size,
            page_size=logical_page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

        # Paged allocator page IDs become physical addresses after loc/dcp.
        # Offset the striped IDs past the replicated physical region.
        striped_page_offset = replicated_size // physical_page_size
        striped_pages = striped_physical_size // physical_page_size
        self.striped.free_pages = torch.arange(
            striped_page_offset + 1,
            striped_page_offset + striped_pages + 1,
            dtype=torch.int64,
            device=device,
        )
        self.striped.release_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.striped_virtual_start = (striped_page_offset + 1) * logical_page_size
        self.num_pages = replicated_size // logical_page_size + striped_pages
        self.set_active_residency(KvResidency.REPLICATED)

    def set_active_residency(self, residency: KvResidency) -> None:
        residency = KvResidency(residency)
        if residency is KvResidency.TRANSITIONING:
            raise RuntimeError("cannot allocate while KV residency is transitioning")
        self.active_residency = residency
        if hasattr(self, "replicated"):
            self.size = self._active.size
            self.page_size = self._active.page_size

    @property
    def _active(self) -> PagedTokenToKVPoolAllocator:
        return (
            self.striped
            if self.active_residency is KvResidency.STRIPED
            else self.replicated
        )

    @property
    def free_pages(self):
        return torch.cat((self.replicated.free_pages, self.striped.free_pages))

    @free_pages.setter
    def free_pages(self, _value):
        # BaseTokenToKVPoolAllocator initializes this before child allocators.
        pass

    @property
    def release_pages(self):
        return torch.cat((self.replicated.release_pages, self.striped.release_pages))

    @release_pages.setter
    def release_pages(self, _value):
        pass

    def available_size(self):
        return self._active.available_size()

    def total_available_size(self):
        return self.replicated.available_size() + self.striped.available_size()

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        return self._active.alloc(need_size)

    def alloc_extend(self, *args, **kwargs):
        return self._active.alloc_extend(*args, **kwargs)

    def alloc_decode(self, *args, **kwargs):
        return self._active.alloc_decode(*args, **kwargs)

    def _split_indices(self, indices: torch.Tensor):
        replicated_mask = indices < self.striped_virtual_start
        return indices[replicated_mask], indices[~replicated_mask]

    def free(self, free_index: torch.Tensor):
        replicated, striped = self._split_indices(free_index)
        self.replicated.free(replicated)
        self.striped.free(striped)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        replicated, striped = self._split_indices(free_index)
        if replicated.numel():
            self.replicated.free_segment(replicated, start_pos=start_pos)
        if striped.numel():
            self.striped.free_segment(striped, start_pos=start_pos)

    def free_segments(self, segments):
        for free_index, start_pos in segments:
            self.free_segment(free_index, start_pos=start_pos)

    def free_group_begin(self):
        self.replicated.free_group_begin()
        self.striped.free_group_begin()

    def free_group_end(self):
        self.replicated.free_group_end()
        self.striped.free_group_end()

    def merge_and_sort_free(self):
        self.replicated.merge_and_sort_free()
        self.striped.merge_and_sort_free()

    def clear(self):
        self.replicated.clear()
        self.striped.clear()
        striped_page_offset = self.replicated_size // self.physical_page_size
        striped_pages = self.striped_physical_size // self.physical_page_size
        self.striped.free_pages = torch.arange(
            striped_page_offset + 1,
            striped_page_offset + striped_pages + 1,
            dtype=torch.int64,
            device=self.device,
        )

    def translate_kv_indices_for_transfer(self, kv_indices: torch.Tensor):
        replicated, striped = self._split_indices(kv_indices)
        if replicated.numel() == 0:
            return striped // self.dcp_size
        if striped.numel() == 0:
            return replicated
        # Mixed transfers preserve request order; callers should split by
        # residency before transfer rather than concatenate two address spaces.
        raise RuntimeError("mixed-residency KV transfer is not supported")

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )

    def resize(self, config) -> None:
        raise NotImplementedError(
            "runtime resize is not supported for residency-aware KV allocation"
        )

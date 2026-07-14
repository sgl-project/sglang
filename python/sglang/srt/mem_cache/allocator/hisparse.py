import weakref
from collections.abc import Callable

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.utils.common import get_num_new_pages


def _stable_unique_page_ids(page_ids: torch.Tensor) -> torch.Tensor:
    if page_ids.numel() == 0:
        return page_ids.to(dtype=torch.int64)

    unique_page_ids, inverse = torch.unique(
        page_ids.to(dtype=torch.int64), sorted=False, return_inverse=True
    )
    positions = torch.arange(
        page_ids.numel(), dtype=torch.int64, device=page_ids.device
    )
    first_positions = torch.full_like(unique_page_ids, page_ids.numel())
    first_positions.scatter_reduce_(
        0, inverse, positions, reduce="amin", include_self=True
    )
    return unique_page_ids[torch.argsort(first_positions)]


class _HiSparsePageOwnership:
    def __init__(
        self,
        *,
        mapping: torch.Tensor,
        child_allocator: PagedTokenToKVPoolAllocator,
        page_size: int,
    ) -> None:
        assert child_allocator.is_not_in_free_group
        assert page_size > 0
        self.mapping = mapping
        self.child_allocator = child_allocator
        self.page_size = page_size

    def release(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
        clear_extra_owner: Callable[[], None] | None = None,
        unique_page_owners: bool = False,
    ) -> None:
        coordinates = self.mapping[mapping_indices]
        if extra_owned_coordinates is not None:
            coordinates = torch.cat([coordinates, extra_owned_coordinates])
        owned_page_ids = self._owned_page_ids(
            coordinates, unique_page_owners=unique_page_owners
        )
        self._clear_owners_and_release(
            mapping_indices=mapping_indices,
            owned_page_ids=owned_page_ids,
            clear_extra_owner=clear_extra_owner,
        )

    def take_device_buffer(
        self,
        *,
        ordered_real_mapping_indices: torch.Tensor,
        allocated_mapping_indices: torch.Tensor,
        need_size: int,
        newest_position: int | None,
    ) -> torch.Tensor | None:
        assert need_size % self.page_size == 0
        ordered_real_coordinates = self.mapping[ordered_real_mapping_indices]
        ordered_real_coordinates = ordered_real_coordinates[
            ordered_real_coordinates > 0
        ].to(dtype=torch.int64)
        allocated_coordinates = self.mapping[allocated_mapping_indices]
        allocated_page_ids = self._owned_page_ids(allocated_coordinates)

        if (
            newest_position is not None
            and ordered_real_coordinates.numel() > newest_position + 1
        ):
            return self._take_device_buffer_with_reserved_newest_page(
                ordered_real_coordinates=ordered_real_coordinates,
                allocated_mapping_indices=allocated_mapping_indices,
                allocated_page_ids=allocated_page_ids,
                need_size=need_size,
                newest_position=newest_position,
            )

        ordered_prefix = ordered_real_coordinates[:need_size]
        semantic_page_ids = self._owned_page_ids(ordered_prefix)
        semantic_blocks = self._full_page_blocks(semantic_page_ids)
        completion_candidates = semantic_blocks[
            ~torch.isin(semantic_blocks, ordered_prefix)
        ]
        completion_size = need_size - ordered_prefix.numel()
        semantic_completion = completion_candidates[:completion_size]

        remaining_size = (
            need_size - ordered_prefix.numel() - semantic_completion.numel()
        )
        assert remaining_size % self.page_size == 0
        padding_page_ids = allocated_page_ids[
            ~torch.isin(allocated_page_ids, semantic_page_ids)
        ]
        retained_padding_page_ids = padding_page_ids[: remaining_size // self.page_size]
        retained_padding_blocks = self._full_page_blocks(retained_padding_page_ids)

        new_size = remaining_size - retained_padding_blocks.numel()
        assert new_size % self.page_size == 0
        if new_size > 0:
            new_blocks = self.child_allocator.alloc(new_size)
            if new_blocks is None:
                return None
            new_blocks = new_blocks.to(dtype=torch.int64)
        else:
            new_blocks = allocated_page_ids[:0]

        retained_page_ids = torch.cat([semantic_page_ids, retained_padding_page_ids])
        surplus_page_ids = allocated_page_ids[
            ~torch.isin(allocated_page_ids, retained_page_ids)
        ]
        buffer_indices = torch.cat(
            [
                ordered_prefix,
                semantic_completion,
                retained_padding_blocks,
                new_blocks,
            ]
        )
        assert buffer_indices.numel() == need_size
        torch._assert_async(
            torch.all(buffer_indices > 0),
            "HiSparse device buffers must contain positive coordinates",
        )

        self._clear_owners_and_release(
            mapping_indices=allocated_mapping_indices,
            owned_page_ids=surplus_page_ids,
        )
        return buffer_indices

    def _take_device_buffer_with_reserved_newest_page(
        self,
        *,
        ordered_real_coordinates: torch.Tensor,
        allocated_mapping_indices: torch.Tensor,
        allocated_page_ids: torch.Tensor,
        need_size: int,
        newest_position: int,
    ) -> torch.Tensor:
        assert newest_position == need_size - self.page_size
        newest_coordinate = ordered_real_coordinates[-1]
        newest_page_start = (newest_coordinate // self.page_size) * self.page_size
        newest_page = newest_page_start + torch.arange(
            self.page_size,
            dtype=torch.int64,
            device=ordered_real_coordinates.device,
        )
        reserved_page = torch.cat(
            [newest_coordinate.view(1), newest_page[newest_page != newest_coordinate]]
        )
        buffer_indices = torch.cat(
            [ordered_real_coordinates[:newest_position], reserved_page]
        )
        retained_page_ids = self._owned_page_ids(buffer_indices)
        retained_blocks = self._full_page_blocks(retained_page_ids)
        assert retained_page_ids.numel() == need_size // self.page_size
        torch._assert_async(
            torch.all(torch.isin(retained_blocks, buffer_indices)),
            "HiSparse device buffer must retain complete pages",
        )

        surplus_page_ids = allocated_page_ids[
            ~torch.isin(allocated_page_ids, retained_page_ids)
        ]
        self._clear_owners_and_release(
            mapping_indices=allocated_mapping_indices,
            owned_page_ids=surplus_page_ids,
        )
        return buffer_indices

    def _owned_page_ids(
        self,
        coordinates: torch.Tensor,
        *,
        unique_page_owners: bool = False,
    ) -> torch.Tensor:
        positive_coordinates = coordinates[coordinates > 0].to(dtype=torch.int64)
        page_ids = positive_coordinates // self.page_size
        if unique_page_owners:
            assert self.page_size == 1
            return page_ids
        return _stable_unique_page_ids(page_ids)

    def _full_page_blocks(self, owned_page_ids: torch.Tensor) -> torch.Tensor:
        if owned_page_ids.numel() == 0:
            return owned_page_ids

        offsets = torch.arange(
            self.page_size, dtype=torch.int64, device=owned_page_ids.device
        )
        return (owned_page_ids[:, None] * self.page_size + offsets).reshape(-1)

    def _release_owned_page_ids(self, owned_page_ids: torch.Tensor) -> None:
        assert self.child_allocator.is_not_in_free_group
        full_page_blocks = self._full_page_blocks(owned_page_ids)
        assert full_page_blocks.numel() % self.page_size == 0
        if full_page_blocks.numel() == 0:
            return
        self.child_allocator.free(full_page_blocks)

    def _clear_owners_and_release(
        self,
        *,
        mapping_indices: torch.Tensor,
        owned_page_ids: torch.Tensor,
        clear_extra_owner: Callable[[], None] | None = None,
    ) -> None:
        self.mapping[mapping_indices] = 0
        if clear_extra_owner is not None:
            clear_extra_owner()
        self._release_owned_page_ids(owned_page_ids)


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
        self._page_ownership = _HiSparsePageOwnership(
            mapping=self.full_to_hisparse_device_index_mapping,
            child_allocator=self.hisparse_attn_allocator,
            page_size=self.page_size,
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

    @property
    def hisparse_device_page_size(self) -> int:
        return self.page_size

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size(),
        )

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        if self.page_size != 1:
            raise NotImplementedError(
                "HiSparse generic allocation is only supported for page_size=1. "
                "Use alloc_extend for paged allocation."
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
        return self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )

    def alloc_device_buffer(
        self,
        *,
        ordered_real_mapping_indices: torch.Tensor,
        allocated_mapping_indices: torch.Tensor,
        need_size: int,
    ) -> torch.Tensor | None:
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        return self._page_ownership.take_device_buffer(
            ordered_real_mapping_indices=ordered_real_mapping_indices,
            allocated_mapping_indices=allocated_mapping_indices,
            need_size=need_size,
            newest_position=None,
        )

    def release_hisparse_ownership(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
        clear_extra_owner: Callable[[], None] | None = None,
        unique_page_owners: bool = False,
    ) -> None:
        self._page_ownership.release(
            mapping_indices=mapping_indices,
            extra_owned_coordinates=extra_owned_coordinates,
            clear_extra_owner=clear_extra_owner,
            unique_page_owners=unique_page_owners,
        )

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return last_locs

    def get_last_loc_hisparse_device(self, last_locs: torch.Tensor):
        return self._kvcache._translate_loc_to_hisparse_device(last_locs)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
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
        return self.logical_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def free_hisparse(self, free_indices: torch.Tensor):
        self.release_hisparse_ownership(mapping_indices=free_indices)

    def clear(self):
        self.logical_attn_allocator.clear()
        self.hisparse_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_hisparse_device_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

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
        self._page_ownership = _HiSparsePageOwnership(
            mapping=self.full_to_hisparse_device_index_mapping,
            child_allocator=self.hisparse_attn_allocator,
            page_size=self.hisparse_page_size,
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
    def size_full(self) -> int:
        return self._size_full

    @property
    def size(self) -> int:
        return self.logical_attn_allocator.size

    @property
    def size_swa(self) -> int:
        return self.logical_attn_allocator.size_swa

    @property
    def hisparse_device_page_size(self) -> int:
        return self.hisparse_page_size

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

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "DeepSeek V4 HiSparse allocator does not support direct token allocation; "
            "use alloc_extend or alloc_decode instead."
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
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        swa_tail_len: int,
        swa_tail_end: int,
    ):
        return self.logical_attn_allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
            swa_tail_len=swa_tail_len,
            swa_tail_end=swa_tail_end,
        )

    def alloc_device_buffer(
        self,
        *,
        ordered_real_mapping_indices: torch.Tensor,
        allocated_mapping_indices: torch.Tensor,
        need_size: int,
    ) -> torch.Tensor | None:
        return self._page_ownership.take_device_buffer(
            ordered_real_mapping_indices=ordered_real_mapping_indices,
            allocated_mapping_indices=allocated_mapping_indices,
            need_size=need_size,
            newest_position=need_size - self.hisparse_page_size,
        )

    def release_hisparse_ownership(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
        clear_extra_owner: Callable[[], None] | None = None,
        unique_page_owners: bool = False,
    ) -> None:
        self._page_ownership.release(
            mapping_indices=mapping_indices,
            extra_owned_coordinates=extra_owned_coordinates,
            clear_extra_owner=clear_extra_owner,
            unique_page_owners=unique_page_owners,
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
        self.release_hisparse_ownership(mapping_indices=compressed_indices)

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

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
        else:
            self.free_group.append(free_index)

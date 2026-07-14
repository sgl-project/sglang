import weakref

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool


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


def _coordinates_to_owned_page_ids(
    coordinates: torch.Tensor,
    *,
    device_page_size: int,
) -> torch.Tensor:
    assert coordinates.ndim == 1
    assert coordinates.dtype in (torch.int32, torch.int64)
    assert device_page_size > 0

    positive_coordinates = coordinates[coordinates > 0].to(dtype=torch.int64)
    return _stable_unique_page_ids(positive_coordinates // device_page_size)


def _owned_page_ids_to_full_blocks(
    owned_page_ids: torch.Tensor,
    *,
    device_page_size: int,
) -> torch.Tensor:
    assert owned_page_ids.ndim == 1
    assert owned_page_ids.dtype == torch.int64
    assert device_page_size > 0
    if owned_page_ids.numel() == 0:
        return owned_page_ids

    torch._assert_async(
        torch.all(owned_page_ids > 0), "HiSparse owned page IDs must be positive"
    )
    sorted_page_ids = torch.sort(owned_page_ids).values
    torch._assert_async(
        torch.all(sorted_page_ids[1:] != sorted_page_ids[:-1]),
        "HiSparse owned page IDs must be unique",
    )
    offsets = torch.arange(
        device_page_size, dtype=torch.int64, device=owned_page_ids.device
    )
    return (owned_page_ids[:, None] * device_page_size + offsets).reshape(-1)


def _release_owned_page_ids(
    child_allocator: PagedTokenToKVPoolAllocator,
    *,
    owned_page_ids: torch.Tensor,
    device_page_size: int,
) -> None:
    assert child_allocator.is_not_in_free_group

    full_page_blocks = _owned_page_ids_to_full_blocks(
        owned_page_ids,
        device_page_size=device_page_size,
    )
    assert full_page_blocks.ndim == 1
    assert full_page_blocks.dtype == torch.int64
    assert full_page_blocks.numel() % device_page_size == 0
    if full_page_blocks.numel() == 0:
        return

    page_rows = full_page_blocks.view(-1, device_page_size)
    expected_rows = page_rows[:, :1] + torch.arange(
        device_page_size, dtype=torch.int64, device=full_page_blocks.device
    )
    torch._assert_async(
        torch.all(page_rows[:, 0] % device_page_size == 0),
        "HiSparse child free blocks must be page aligned",
    )
    torch._assert_async(
        torch.all(page_rows == expected_rows),
        "HiSparse child free blocks must contain complete consecutive pages",
    )
    child_allocator.free(full_page_blocks)


def _validate_page_block_indices(
    indices: torch.Tensor,
    *,
    expected_size: int,
    page_size: int,
    device: torch.device,
) -> None:
    assert indices.ndim == 1, f"{indices.shape=}"
    assert indices.is_contiguous(), f"{indices.stride()=}"
    assert indices.dtype == torch.int64, f"{indices.dtype=}"
    assert indices.device == device, f"{indices.device=}, {device=}"
    assert indices.numel() == expected_size, f"{indices.numel()=}, {expected_size=}"
    assert indices.numel() % page_size == 0, f"{indices.numel()=}, {page_size=}"
    if indices.numel() == 0:
        return

    page_rows = indices.view(-1, page_size)
    expected_rows = page_rows[:, :1] + torch.arange(
        page_size,
        dtype=torch.int64,
        device=device,
    )
    torch._assert_async(
        torch.all(page_rows[:, 0] % page_size == 0),
        "HiSparse allocation blocks must be page aligned",
    )
    torch._assert_async(
        torch.all(page_rows == expected_rows),
        "HiSparse allocation blocks must contain complete consecutive pages",
    )


def _build_device_buffer_candidate(
    *,
    mapping: torch.Tensor,
    child_allocator: PagedTokenToKVPoolAllocator,
    ordered_real_mapping_indices: torch.Tensor,
    allocated_mapping_indices: torch.Tensor,
    need_size: int,
    device_page_size: int,
    newest_position: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    ordered_real_coordinates = mapping[ordered_real_mapping_indices]
    ordered_real_coordinates = ordered_real_coordinates[
        ordered_real_coordinates > 0
    ].to(dtype=torch.int64)
    all_allocated_coordinates = mapping[allocated_mapping_indices]
    all_owned_page_ids = _coordinates_to_owned_page_ids(
        all_allocated_coordinates,
        device_page_size=device_page_size,
    )

    if (
        newest_position is not None
        and ordered_real_coordinates.numel() > newest_position + 1
    ):
        ordered_real_coordinates = ordered_real_coordinates.clone()
        newest_coordinate = ordered_real_coordinates[-1].clone()
        displaced_coordinate = ordered_real_coordinates[newest_position].clone()
        ordered_real_coordinates[newest_position] = newest_coordinate
        ordered_real_coordinates[-1] = displaced_coordinate

    ordered_real_prefix = ordered_real_coordinates[:need_size]
    semantic_retained_page_ids = _coordinates_to_owned_page_ids(
        ordered_real_prefix,
        device_page_size=device_page_size,
    )
    semantic_full_blocks = _owned_page_ids_to_full_blocks(
        semantic_retained_page_ids,
        device_page_size=device_page_size,
    )
    semantic_unseen_coordinates = semantic_full_blocks[
        ~torch.isin(semantic_full_blocks, ordered_real_prefix)
    ]
    tail_capacity = need_size - ordered_real_prefix.numel()
    semantic_completion_tail = semantic_unseen_coordinates[:tail_capacity]

    remaining_size = (
        need_size - ordered_real_prefix.numel() - semantic_completion_tail.numel()
    )
    assert remaining_size % device_page_size == 0

    padding_only_owned_page_ids = all_owned_page_ids[
        ~torch.isin(all_owned_page_ids, semantic_retained_page_ids)
    ]
    needed_tail_pages = remaining_size // device_page_size
    tail_retained_page_ids = padding_only_owned_page_ids[:needed_tail_pages]
    padding_only_full_page_tail = _owned_page_ids_to_full_blocks(
        tail_retained_page_ids,
        device_page_size=device_page_size,
    )

    extra_size = remaining_size - padding_only_full_page_tail.numel()
    assert extra_size % device_page_size == 0
    if extra_size > 0:
        new_page_tail = child_allocator.alloc(extra_size)
        if new_page_tail is None:
            return None, all_owned_page_ids[:0]
        new_page_tail = new_page_tail.to(dtype=torch.int64)
    else:
        new_page_tail = all_owned_page_ids[:0]

    retained_page_ids = torch.cat([semantic_retained_page_ids, tail_retained_page_ids])
    pure_surplus_page_ids = all_owned_page_ids[
        ~torch.isin(all_owned_page_ids, retained_page_ids)
    ]
    buffer_indices = torch.cat(
        [
            ordered_real_prefix,
            semantic_completion_tail,
            padding_only_full_page_tail,
            new_page_tail,
        ]
    )
    assert buffer_indices.numel() == need_size
    torch._assert_async(
        torch.all(buffer_indices > 0),
        "HiSparse device buffers must contain positive coordinates",
    )
    return buffer_indices, pure_surplus_page_ids


class HiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    supports_page_aligned_alloc: bool = True
    supports_spec_page_aligned_alloc: bool = False

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

    def _get_free_page_owner_bounds(self) -> tuple[int, int]:
        return self.logical_attn_allocator._get_free_page_owner_bounds()

    def _validate_logical_domain_free(
        self,
        free_index: torch.Tensor,
    ) -> None:
        self._validate_free_index_metadata(free_index, page_size=self.page_size)
        owner_page_start, owner_page_end = self._get_free_page_owner_bounds()
        self._debug_validate_free_index(
            free_index,
            page_size=self.page_size,
            owner_page_start=owner_page_start,
            owner_page_end=owner_page_end,
        )

    def alloc(self, need_size: int):
        assert self.is_not_in_free_group
        assert need_size >= 0, f"{need_size=}"
        assert need_size % self.page_size == 0, f"{need_size=}, {self.page_size=}"
        if need_size == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        if need_size > self.logical_attn_allocator.available_size():
            return None
        if need_size > self.hisparse_attn_allocator.available_size():
            return None

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None
        _validate_page_block_indices(
            logical_indices,
            expected_size=need_size,
            page_size=self.page_size,
            device=self.full_to_hisparse_device_index_mapping.device,
        )

        hisparse_indices = self.hisparse_attn_allocator.alloc(need_size)
        if hisparse_indices is None:
            self.logical_attn_allocator.free(logical_indices)
            torch._assert_async(
                torch.all(
                    self.full_to_hisparse_device_index_mapping[logical_indices] == 0
                ),
                "HiSparse mapping must remain unpublished after allocation rollback",
            )
            return None
        _validate_page_block_indices(
            hisparse_indices,
            expected_size=need_size,
            page_size=self.hisparse_device_page_size,
            device=self.full_to_hisparse_device_index_mapping.device,
        )

        self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
        return logical_indices

    def alloc_logical_only(self, *, need_size: int):
        assert need_size >= 0, f"{need_size=}"
        assert need_size % self.page_size == 0, f"{need_size=}, {self.page_size=}"
        return self.logical_attn_allocator.alloc(need_size)

    def collect_owned_hisparse_page_ids(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coordinates = self.full_to_hisparse_device_index_mapping[mapping_indices]
        if extra_owned_coordinates is not None:
            coordinates = torch.cat([coordinates, extra_owned_coordinates])
        return _coordinates_to_owned_page_ids(
            coordinates,
            device_page_size=self.hisparse_device_page_size,
        )

    def clear_hisparse_mapping(self, *, mapping_indices: torch.Tensor) -> None:
        self.full_to_hisparse_device_index_mapping[mapping_indices] = 0
        torch._assert_async(
            torch.all(self.full_to_hisparse_device_index_mapping[mapping_indices] == 0),
            "HiSparse mapping owners must be cleared before release",
        )

    def release_owned_hisparse_pages(self, *, owned_page_ids: torch.Tensor) -> None:
        _release_owned_page_ids(
            self.hisparse_attn_allocator,
            owned_page_ids=owned_page_ids,
            device_page_size=self.hisparse_device_page_size,
        )

    def materialize_owned_hisparse_page_blocks(
        self, *, owned_page_ids: torch.Tensor
    ) -> torch.Tensor:
        return _owned_page_ids_to_full_blocks(
            owned_page_ids,
            device_page_size=self.hisparse_device_page_size,
        )

    def alloc_device_buffer(
        self,
        *,
        ordered_real_mapping_indices: torch.Tensor,
        allocated_mapping_indices: torch.Tensor,
        need_size: int,
    ) -> torch.Tensor | None:
        assert need_size % self.hisparse_device_page_size == 0
        # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
        # Filter valid (non-zero) hisparse indices.
        # In the direct-to-host path, mapping is all zeros since no hisparse
        # device indices were pre-allocated.
        # page alignment, claiming the residual space for an incomplete page
        buffer_indices, pure_surplus_page_ids = _build_device_buffer_candidate(
            mapping=self.full_to_hisparse_device_index_mapping,
            child_allocator=self.hisparse_attn_allocator,
            ordered_real_mapping_indices=ordered_real_mapping_indices,
            allocated_mapping_indices=allocated_mapping_indices,
            need_size=need_size,
            device_page_size=self.hisparse_device_page_size,
            newest_position=None,
        )
        if buffer_indices is None:
            return None

        self.clear_hisparse_mapping(mapping_indices=allocated_mapping_indices)
        self.release_owned_hisparse_pages(owned_page_ids=pure_surplus_page_ids)
        return buffer_indices

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return last_locs

    def free_hisparse(self, free_indices: torch.Tensor):
        owned_page_ids = self.collect_owned_hisparse_page_ids(
            mapping_indices=free_indices
        )
        self.clear_hisparse_mapping(mapping_indices=free_indices)
        self.release_owned_hisparse_pages(owned_page_ids=owned_page_ids)

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

    def free(self, free_index: torch.Tensor):
        self._validate_logical_domain_free(free_index)
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
    supports_page_aligned_alloc: bool = True
    supports_spec_page_aligned_alloc: bool = False

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

    def _get_free_page_owner_bounds(self) -> tuple[int, int]:
        return self.logical_attn_allocator._get_free_page_owner_bounds()

    def _validate_logical_domain_free(
        self,
        free_index: torch.Tensor,
    ) -> None:
        self._validate_free_index_metadata(free_index, page_size=self.page_size)
        owner_page_start, owner_page_end = self._get_free_page_owner_bounds()
        self._debug_validate_free_index(
            free_index,
            page_size=self.page_size,
            owner_page_start=owner_page_start,
            owner_page_end=owner_page_end,
        )

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
        self._validate_logical_domain_free(free_indices)
        self.logical_attn_allocator.free_swa(free_indices)

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def alloc(self, need_size: int):
        assert self.is_not_in_free_group
        assert self.compress_ratio == 4
        assert self.page_size % self.compress_ratio == 0
        assert self.hisparse_page_size == self.page_size // self.compress_ratio
        assert need_size >= 0, f"{need_size=}"
        assert need_size % self.page_size == 0, f"{need_size=}, {self.page_size=}"
        if need_size == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        hisparse_need_size = need_size // self.compress_ratio
        if need_size > self.logical_attn_allocator.available_size():
            return None
        if hisparse_need_size > self.hisparse_attn_allocator.available_size():
            return None

        logical_indices = self.logical_attn_allocator.alloc(need_size)
        if logical_indices is None:
            return None
        _validate_page_block_indices(
            logical_indices,
            expected_size=need_size,
            page_size=self.page_size,
            device=self.full_to_hisparse_device_index_mapping.device,
        )

        compressed_indices = (
            self.hisparse_kvcache.translate_loc_from_full_to_compressed(logical_indices)
        )
        _validate_page_block_indices(
            compressed_indices,
            expected_size=hisparse_need_size,
            page_size=self.hisparse_page_size,
            device=self.full_to_hisparse_device_index_mapping.device,
        )
        expected_compressed_indices = (
            logical_indices[(logical_indices + 1) % self.compress_ratio == 0]
            // self.compress_ratio
        )
        torch._assert_async(
            torch.all(compressed_indices == expected_compressed_indices),
            "DeepSeek V4 compressed mapping keys must match the C4 translation",
        )

        hisparse_indices = self.hisparse_attn_allocator.alloc(hisparse_need_size)
        if hisparse_indices is None:
            self.logical_attn_allocator.free(logical_indices)
            torch._assert_async(
                torch.all(
                    self.full_to_hisparse_device_index_mapping[compressed_indices] == 0
                ),
                "HiSparse mapping must remain unpublished after allocation rollback",
            )
            return None
        _validate_page_block_indices(
            hisparse_indices,
            expected_size=hisparse_need_size,
            page_size=self.hisparse_page_size,
            device=self.full_to_hisparse_device_index_mapping.device,
        )

        self.full_to_hisparse_device_index_mapping[compressed_indices] = (
            hisparse_indices
        )
        return logical_indices

    def alloc_logical_only(self, *, need_size: int):
        assert need_size >= 0, f"{need_size=}"
        assert need_size % self.page_size == 0, f"{need_size=}, {self.page_size=}"
        return self.logical_attn_allocator.alloc(need_size)

    def collect_owned_hisparse_page_ids(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coordinates = self.full_to_hisparse_device_index_mapping[mapping_indices]
        if extra_owned_coordinates is not None:
            coordinates = torch.cat([coordinates, extra_owned_coordinates])
        return _coordinates_to_owned_page_ids(
            coordinates,
            device_page_size=self.hisparse_device_page_size,
        )

    def clear_hisparse_mapping(self, *, mapping_indices: torch.Tensor) -> None:
        self.full_to_hisparse_device_index_mapping[mapping_indices] = 0
        torch._assert_async(
            torch.all(self.full_to_hisparse_device_index_mapping[mapping_indices] == 0),
            "HiSparse mapping owners must be cleared before release",
        )

    def release_owned_hisparse_pages(self, *, owned_page_ids: torch.Tensor) -> None:
        _release_owned_page_ids(
            self.hisparse_attn_allocator,
            owned_page_ids=owned_page_ids,
            device_page_size=self.hisparse_device_page_size,
        )

    def materialize_owned_hisparse_page_blocks(
        self, *, owned_page_ids: torch.Tensor
    ) -> torch.Tensor:
        return _owned_page_ids_to_full_blocks(
            owned_page_ids,
            device_page_size=self.hisparse_device_page_size,
        )

    def alloc_device_buffer(
        self,
        *,
        ordered_real_mapping_indices: torch.Tensor,
        allocated_mapping_indices: torch.Tensor,
        need_size: int,
    ) -> torch.Tensor | None:
        assert need_size % self.hisparse_device_page_size == 0
        device_buffer_size = need_size - self.hisparse_device_page_size
        buffer_indices, pure_surplus_page_ids = _build_device_buffer_candidate(
            mapping=self.full_to_hisparse_device_index_mapping,
            child_allocator=self.hisparse_attn_allocator,
            ordered_real_mapping_indices=ordered_real_mapping_indices,
            allocated_mapping_indices=allocated_mapping_indices,
            need_size=need_size,
            device_page_size=self.hisparse_device_page_size,
            newest_position=device_buffer_size,
        )
        if buffer_indices is None:
            return None

        self.clear_hisparse_mapping(mapping_indices=allocated_mapping_indices)
        self.release_owned_hisparse_pages(owned_page_ids=pure_surplus_page_ids)
        return buffer_indices

    def get_last_loc_compressed(self, last_locs: torch.Tensor):
        return (last_locs - 3) // self.compress_ratio

    def free_compressed(self, compressed_indices: torch.Tensor):
        owned_page_ids = self.collect_owned_hisparse_page_ids(
            mapping_indices=compressed_indices
        )
        self.clear_hisparse_mapping(mapping_indices=compressed_indices)
        self.release_owned_hisparse_pages(owned_page_ids=owned_page_ids)

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
        self._validate_logical_domain_free(free_index)
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.logical_attn_allocator.free(free_index)
        else:
            self.free_group.append(free_index)

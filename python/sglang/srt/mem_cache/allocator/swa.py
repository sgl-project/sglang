import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.utils import is_npu

_is_npu = is_npu()


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    supports_page_aligned_alloc: bool = True
    supports_spec_page_aligned_alloc: bool = True

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: BaseSWAKVPool,
        need_sort: bool,
    ):
        assert isinstance(kvcache, BaseSWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

        full_kv_pool = getattr(kvcache, "full_kv_pool", None)
        swa_kv_pool = getattr(kvcache, "swa_kv_pool", None)

        if page_size == 1:
            self.full_attn_allocator = TokenToKVPoolAllocator(
                size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        else:
            PagedTokenToKVPoolAllocatorClass = self._get_paged_allocator_class()
            self.full_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size,
                page_size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocatorClass(
                size_swa,
                page_size,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        # Append one more item so the -1 padding sentinel maps to itself.
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    size + self.page_size,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.need_sort = need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self._kvcache = kvcache
        self.clear()
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    def _get_paged_allocator_class(
        self,
    ) -> type[PagedTokenToKVPoolAllocator]:
        return PagedTokenToKVPoolAllocator

    def available_size(self):
        return min(
            self.full_attn_allocator.available_size(),
            self.swa_attn_allocator.available_size(),
        )

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    # Slot-conservation views for the leak invariant. On the non-shared allocator
    # the static budget IS physical (conserve == physical); the shared composite
    # overrides these with the static-cap view.
    def _conserve_full_available_size(self):
        return self.full_available_size()

    def _conserve_swa_available_size(self):
        return self.swa_available_size()

    @property
    def size(self):
        return min(self._size_full, self._size_swa)

    @property
    def size_swa(self):
        return self._size_swa

    @property
    def size_full(self):
        return self._size_full

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def _get_free_page_owner_bounds(self) -> tuple[int, int]:
        return self.full_attn_allocator._get_free_page_owner_bounds()

    def _validate_full_domain_free(
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

    def _preflight_swa_free(
        self,
        free_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not _is_npu
        self._validate_full_domain_free(free_index)
        if free_index.numel() == 0:
            return free_index, free_index

        if self.page_size == 1:
            mapping_indices = free_index
        else:
            mapping_indices = self._expand_to_full_pages(free_index)

        mapped_swa_indices = self.full_to_swa_index_mapping[mapping_indices]
        if self.page_size > 1 and envs.SGLANG_DEBUG_MEMORY_POOL.get():
            live_page_rows = (mapped_swa_indices > 0).reshape(-1, self.page_size)
            torch._assert_async(
                torch.all(live_page_rows == live_page_rows[:, :1]),
                f"{type(self).__name__}.free_swa requires pagewise-uniform "
                "SWA mappings",
            )

        swa_indices = mapped_swa_indices[mapped_swa_indices > 0]
        self.swa_attn_allocator._validate_free_index_metadata(
            swa_indices,
            page_size=self.page_size,
        )
        owner_page_start, owner_page_end = (
            self.swa_attn_allocator._get_free_page_owner_bounds()
        )
        self.swa_attn_allocator._debug_validate_free_index(
            swa_indices,
            page_size=self.page_size,
            owner_page_start=owner_page_start,
            owner_page_end=owner_page_end,
        )
        return mapping_indices, swa_indices

    def _release_preflighted_swa_free(
        self,
        mapping_indices: torch.Tensor,
        swa_indices: torch.Tensor,
    ) -> None:
        self.full_to_swa_index_mapping[mapping_indices] = 0
        self.swa_attn_allocator.free(swa_indices)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self._kvcache.full_to_swa_index_mapping is not None
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        assert need_size >= 0, f"{need_size=}"
        assert need_size % self.page_size == 0, f"{need_size=}, {self.page_size=}"
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices: torch.Tensor | None = self.full_attn_allocator.alloc(
            need_size
        )
        alloc_swa_indices: torch.Tensor | None = self.swa_attn_allocator.alloc(
            need_size
        )
        assert (
            alloc_full_indices is not None
        ), "full allocator returned None after the joint capacity pre-check passed"
        assert (
            alloc_swa_indices is not None
        ), "SWA allocator returned None after the joint capacity pre-check passed"

        expected_device: torch.device = self.full_to_swa_index_mapping.device
        allocator_outputs: tuple[tuple[str, torch.Tensor], ...] = (
            ("full", alloc_full_indices),
            ("SWA", alloc_swa_indices),
        )
        for allocator_name, allocated_indices in allocator_outputs:
            assert allocated_indices.ndim == 1, (
                f"{allocator_name} allocation must be flat: "
                f"shape={allocated_indices.shape}"
            )
            assert allocated_indices.is_contiguous(), (
                f"{allocator_name} allocation must be contiguous: "
                f"stride={allocated_indices.stride()}"
            )
            assert allocated_indices.dtype == torch.int64, (
                f"{allocator_name} allocation must use torch.int64: "
                f"dtype={allocated_indices.dtype}"
            )
            assert allocated_indices.device == expected_device, (
                f"{allocator_name} allocation is on {allocated_indices.device}, "
                f"expected {expected_device}"
            )
            assert allocated_indices.numel() == need_size, (
                f"{allocator_name} allocation returned {allocated_indices.numel()} "
                f"slots, expected {need_size}"
            )

        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)
        return alloc_full_indices

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            num_full_pages
            <= self.full_attn_allocator.available_size() // self.page_size
            and num_swa_pages
            <= self.swa_attn_allocator.available_size() // self.page_size
        )

    def alloc_extend_swa_tail(
        self,
        *,
        extend_num_tokens: int,
        swa_tail_len: int,
        swa_tail_end: int,
    ):
        assert self.page_size > 1
        assert extend_num_tokens >= 0
        assert extend_num_tokens % self.page_size == 0
        assert 0 <= swa_tail_len <= swa_tail_end <= extend_num_tokens
        win_start: int = swa_tail_end - swa_tail_len
        assert win_start % self.page_size == 0
        mapped_end: int = (
            (swa_tail_end + self.page_size - 1) // self.page_size * self.page_size
        )
        assert win_start <= swa_tail_end <= mapped_end <= extend_num_tokens
        swa_need_size: int = mapped_end - win_start
        assert swa_need_size % self.page_size == 0

        if extend_num_tokens > self.full_attn_allocator.available_size():
            return None
        if swa_need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(extend_num_tokens)
        if alloc_full_indices is None:
            return None
        assert alloc_full_indices.ndim == 1
        assert alloc_full_indices.is_contiguous()
        assert alloc_full_indices.dtype == torch.int64
        assert alloc_full_indices.device == self.full_to_swa_index_mapping.device
        assert alloc_full_indices.numel() == extend_num_tokens

        if swa_need_size == 0:
            torch._assert_async(
                torch.all(
                    self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)]
                    == 0
                ),
                "SWA tail-free allocation must not publish SWA mappings",
            )
            return alloc_full_indices

        alloc_swa_indices = self.swa_attn_allocator.alloc(swa_need_size)
        if alloc_swa_indices is None:
            self.full_attn_allocator.free(alloc_full_indices)
            torch._assert_async(
                torch.all(
                    self.full_to_swa_index_mapping[alloc_full_indices.to(torch.int64)]
                    == 0
                ),
                "SWA mapping must remain unpublished after allocation rollback",
            )
            return None
        assert alloc_swa_indices.ndim == 1
        assert alloc_swa_indices.is_contiguous()
        assert alloc_swa_indices.dtype == torch.int64
        assert alloc_swa_indices.device == self.full_to_swa_index_mapping.device
        assert alloc_swa_indices.numel() == swa_need_size

        self.set_full_to_swa_mapping(
            alloc_full_indices[win_start:mapped_end],
            alloc_swa_indices,
        )
        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if _is_npu:
            if free_index.numel() == 0:
                return

            if self.is_not_in_free_group:
                self.full_attn_allocator.free(free_index)
                self.free_swa(free_index)
            else:
                self.free_group.append(free_index)
            assert (
                self.full_attn_allocator.available_size()
                <= self.full_attn_allocator.size
            )
            assert (
                self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size
            )
            return

        mapping_indices, swa_indices = self._preflight_swa_free(free_index)
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self._release_preflighted_swa_free(mapping_indices, swa_indices)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        """Write full_to_swa_index_mapping[full_indices[i]] = swa_indices[i].

        Used by HiCache load-back path to rebuild the mapping after FULL and SWA device alloc.
        """
        if full_indices.numel() == 0:
            return
        assert full_indices.numel() == swa_indices.numel()
        full_indices = full_indices.to(torch.int64)
        swa_indices = swa_indices.to(self.full_to_swa_index_mapping.dtype)
        self.full_to_swa_index_mapping[full_indices] = swa_indices

    def free_swa(self, free_index: torch.Tensor):
        if _is_npu:
            if free_index.numel() == 0:
                return

            if self.page_size == 1:
                mapping_indices = free_index
            else:
                mapping_indices = self._expand_to_full_pages(free_index)

            swa_indices = self.full_to_swa_index_mapping[mapping_indices]
            swa_indices = swa_indices[swa_indices > 0]
            self.swa_attn_allocator.free(swa_indices)
            self.full_to_swa_index_mapping[mapping_indices] = 0
            return

        mapping_indices, swa_indices = self._preflight_swa_free(free_index)
        if free_index.numel() == 0:
            return
        self._release_preflighted_swa_free(mapping_indices, swa_indices)

    def _expand_to_full_pages(self, indices: torch.Tensor) -> torch.Tensor:
        if _is_npu:
            pages = torch.unique(indices // self.page_size)
        else:
            pages = indices[:: self.page_size].to(dtype=torch.int64) // self.page_size
        page_offsets = torch.arange(
            self.page_size, dtype=indices.dtype, device=indices.device
        )
        return (pages[:, None] * self.page_size + page_offsets[None, :]).reshape(-1)

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def resize(self, config) -> None:
        size_full = int(config.full_max_total_num_tokens)
        size_swa = int(config.swa_max_total_num_tokens)
        self._size_full = size_full
        self._size_swa = size_swa
        for alloc, sz in (
            (self.full_attn_allocator, size_full),
            (self.swa_attn_allocator, size_swa),
        ):
            alloc.size = int(sz)
            if self.page_size > 1:
                alloc.num_pages = int(sz) // self.page_size
        self.clear()

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )


class PureSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Single-pool allocator for models whose every layer is sliding-window attention."""

    supports_page_aligned_alloc: bool = False
    supports_spec_page_aligned_alloc: bool = False

    def __init__(
        self,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: BaseSWAKVPool,
        need_sort: bool,
    ):
        assert page_size == 1
        assert isinstance(kvcache, BaseSWAKVPool)

        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.need_sort = need_sort
        self._size_full = self._size_swa = size_swa

        self.swa_attn_allocator = TokenToKVPoolAllocator(
            size_swa,
            dtype,
            device,
            kvcache.swa_kv_pool,
            need_sort,
        )
        self.full_attn_allocator = self.swa_attn_allocator

        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.arange(size_swa + page_size, dtype=torch.int64, device=device),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self._kvcache = kvcache
        self.swa_attn_allocator.clear()
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    def available_size(self):
        return self.swa_attn_allocator.available_size()

    def full_available_size(self):
        return self.swa_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        avail = self.swa_attn_allocator.available_size() // self.page_size
        return num_full_pages <= avail and num_swa_pages <= avail

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return kv_indices

    def alloc(self, need_size: int):
        assert self.page_size == 1
        return self.swa_attn_allocator.alloc(need_size)

    def alloc_extend_swa_tail(self, *args, **kwargs):
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator does not support page_size > 1."
        )

    def free(self, free_index: torch.Tensor):
        self._validate_free_index_metadata(free_index, page_size=self.page_size)
        sanitized_free_index = free_index[free_index > 0]
        self._validate_full_domain_free(sanitized_free_index)
        if sanitized_free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.swa_attn_allocator.free(sanitized_free_index)
        else:
            self.free_group.append(sanitized_free_index)
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        self._validate_free_index_metadata(free_index, page_size=self.page_size)
        sanitized_free_index = free_index[free_index > 0]
        self._validate_full_domain_free(sanitized_free_index)
        if sanitized_free_index.numel() == 0:
            return
        self.swa_attn_allocator.free(sanitized_free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))
        self.free_group = []

    def backup_state(self):
        return self.swa_attn_allocator.backup_state()

    def restore_state(self, state):
        self.swa_attn_allocator.restore_state(state)

    def clear(self):
        self.swa_attn_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []

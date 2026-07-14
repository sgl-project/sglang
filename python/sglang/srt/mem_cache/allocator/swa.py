import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import get_num_new_pages

_is_npu = is_npu()

if _is_npu:
    import torch_npu

    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

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
            if _is_npu:
                PagedTokenToKVPoolAllocatorClass = NPUPagedTokenToKVPoolAllocator
            else:
                PagedTokenToKVPoolAllocatorClass = PagedTokenToKVPoolAllocator
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
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
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

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self._kvcache.full_to_swa_index_mapping is not None
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        assert self.page_size == 1
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)
        return alloc_full_indices

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            num_full_pages
            <= self.full_attn_allocator.available_size() // self.page_size
            and num_swa_pages
            <= self.swa_attn_allocator.available_size() // self.page_size
        )

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
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(alloc_full_indices, alloc_swa_indices)

        return alloc_full_indices

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
        swa_tail_len: int,
        swa_tail_end: int,
    ):
        """Allocate full KV for the whole extend and SWA KV only for the tail.

        This is used by disaggregated decode preallocation: decode receives full
        prompt KV for full-attention layers, but only the sliding-window state is
        transferred for SWA layers.
        """
        assert self.page_size > 1
        assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"
        assert len(prefix_lens_cpu) == 1
        assert 0 <= swa_tail_len <= swa_tail_end <= extend_num_tokens
        win_start: int = swa_tail_end - swa_tail_len
        assert win_start % self.page_size == 0

        num_full_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_swa_pages = (swa_tail_len + self.page_size - 1) // self.page_size
        if not self.new_pages_available(num_full_pages, num_swa_pages):
            return None

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_full_pages,
        )
        assert alloc_full_indices is not None

        if swa_tail_len == 0:
            return alloc_full_indices

        device = self.device
        swa_prefix_lens = torch.zeros((1,), dtype=torch.int64, device=device)
        swa_prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)
        swa_seq_lens = torch.tensor([swa_tail_len], dtype=torch.int64, device=device)
        swa_seq_lens_cpu = torch.tensor([swa_tail_len], dtype=torch.int64)
        swa_last_loc = torch.tensor([-1], dtype=torch.int64, device=device)

        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            swa_prefix_lens,
            swa_prefix_lens_cpu,
            swa_seq_lens,
            swa_seq_lens_cpu,
            swa_last_loc,
            swa_tail_len,
            num_new_pages=num_swa_pages,
        )
        assert alloc_swa_indices is not None

        self.set_full_to_swa_mapping(
            alloc_full_indices[win_start:swa_tail_end], alloc_swa_indices
        )
        if win_start > 0:
            self.full_to_swa_index_mapping[
                alloc_full_indices[:win_start].to(torch.int64)
            ] = 0
        if swa_tail_end < extend_num_tokens:
            self.full_to_swa_index_mapping[
                alloc_full_indices[swa_tail_end:].to(torch.int64)
            ] = 0
        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        if _is_npu:
            indices_2d = alloc_full_indices.to(torch.int64).unsqueeze(-1)
            torch_npu.npu_scatter_nd_update_(
                self.full_to_swa_index_mapping,
                indices_2d,
                alloc_swa_indices.to(torch.int64),
            )
        else:
            self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
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

    def _expand_to_full_pages(self, indices: torch.Tensor) -> torch.Tensor:
        pages = torch.unique(indices // self.page_size)
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

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator does not support page_size > 1."
        )

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator does not support page_size > 1."
        )

    def alloc_extend_swa_tail(self, *args, **kwargs):
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator does not support page_size > 1."
        )

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.swa_attn_allocator.free(free_index[free_index > 0])
        else:
            self.free_group.append(free_index)
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.swa_attn_allocator.free(free_index[free_index > 0])

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

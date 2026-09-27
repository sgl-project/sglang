import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import get_num_new_pages
from sglang.srt.utils.invariants import Bucket, Invariant, IsTrue, expect

_is_npu = is_npu()

if _is_npu:
    import torch_npu

    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )


# free_swa releases whatever the mapping points at, so an entry that reads as the
# padding slot would push slot 0 into the SWA free list and hand it out twice.
_SWA_PEER_MAPPED = Invariant("swa.peer_mapped", Bucket.FATAL_UNCONTAINABLE, IsTrue())
# free_full leaves the mapping alone, so a live entry would strand its SWA peer.
_SWA_PEER_RELEASED = Invariant("swa.peer_released", Bucket.GUARD, IsTrue())


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
        self.free_group = None
        self.swa_free_group = []
        self.full_free_group = []

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
    ):
        """Allocate full KV for the whole extend and SWA KV only for the tail.

        This is used by disaggregated decode preallocation: decode receives full
        prompt KV for full-attention layers, but only the sliding-window state is
        transferred for SWA layers.
        """
        assert self.page_size > 1
        assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"
        assert len(prefix_lens_cpu) == 1
        assert 0 <= swa_tail_len <= extend_num_tokens

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
            alloc_full_indices[-swa_tail_len:], alloc_swa_indices
        )
        if swa_tail_len < extend_num_tokens:
            self.clear_full_to_swa_mapping(alloc_full_indices[:-swa_tail_len])
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
        # SWA first: it reads the mapping, and a cache action later in this group
        # can re-point free_index at a different SWA slot.
        self.free_swa(free_index)
        self.free_full(free_index)

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

    def clear_full_to_swa_mapping(self, full_indices: torch.Tensor) -> None:
        if full_indices.numel() == 0:
            return
        full_indices = full_indices.to(torch.int64)
        if _is_npu:
            # NPU: aclnnIndexFill is unoptimized; direct assignment avoids the overhead.
            self.full_to_swa_index_mapping[full_indices] = 0
        else:
            # CUDA: index_fill_ passes the 0 as a kernel argument; mapping[idx] = 0
            # copies a host-resident scalar and blocks until the stream drains.
            self.full_to_swa_index_mapping.index_fill_(0, full_indices, 0)

    def free_swa(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.page_size == 1:
            # A filter here would make the output shape data-dependent,
            # which costs a device-to-host sync.
            mapping_indices = free_index
            swa_indices = self.full_to_swa_index_mapping[mapping_indices]
            expect(_SWA_PEER_MAPPED, swa_indices > 0, msg="caller wants free_full")
        else:
            mapping_indices = self._expand_to_full_pages(free_index)
            swa_indices = self.full_to_swa_index_mapping[mapping_indices]

        self.clear_full_to_swa_mapping(mapping_indices)

        if self.free_group is not None:
            # Resolve ownership now. A cache action later in this group may
            # install a new mapping for the same full index.
            self.swa_free_group.append(swa_indices)
            return

        self._release_swa(swa_indices)

    def _release_swa(self, swa_indices: torch.Tensor):
        if self.page_size > 1:
            # HiCache LOAD_BACK re-pairs a page-aligned full chunk with an offset
            # SWA one (commit_hicache_transfer advances by raw token count), so a
            # page can hold unmapped slots; one filter per group, not per call.
            swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_full(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # Checked at enqueue: a cache action later in this group may pair the
        # slot again, and that new peer is not this call's to judge.
        expect(
            _SWA_PEER_RELEASED,
            self.full_to_swa_index_mapping[free_index] == 0,
            msg="caller wants free",
        )
        if self.free_group is None:
            self.full_attn_allocator.free(free_index)
        else:
            self.full_free_group.append(self._copy_for_free_group(free_index))
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )

    def free_group_begin(self):
        super().free_group_begin()
        self.swa_free_group = []
        self.full_free_group = []

    def free_group_end(self):
        super().free_group_end()
        if self.swa_free_group:
            swa_free_group = self.swa_free_group
            self.swa_free_group = []
            self._release_swa(torch.cat(swa_free_group))
        if self.full_free_group:
            full_free_group = self.full_free_group
            self.full_free_group = []
            self.full_attn_allocator.free(torch.cat(full_free_group))
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def _expand_to_full_pages(self, indices: torch.Tensor) -> torch.Tensor:
        # Duplicates are kept: deduplicating would be a torch.unique whose
        # data-dependent output shape synchronizes the scheduler stream, and
        # every consumer ends in the paged free's own page dedup anyway.
        base = (indices // self.page_size) * self.page_size
        page_offsets = torch.arange(
            self.page_size, dtype=indices.dtype, device=indices.device
        )
        expanded = (base[:, None] + page_offsets[None, :]).reshape(-1)
        if self.swa_attn_allocator.debug_mode:
            # Reference unique on CPU: the expansion must cover exactly the
            # touched pages, on every caller's real input.
            got = torch.unique(expanded.cpu() // self.page_size)
            ref = torch.unique(indices.cpu() // self.page_size)
            assert torch.equal(got, ref), "expansion page set mismatch"
        return expanded

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
        self.free_group = None
        self.swa_free_group = []
        self.full_free_group = []

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
        self.free_group = None

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

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        # Registered with the KV pool and read by the attention kernels.
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator has no full->SWA mapping to rewrite"
        )

    def clear_full_to_swa_mapping(self, full_indices: torch.Tensor) -> None:
        raise NotImplementedError(
            "PureSWATokenToKVPoolAllocator has no full->SWA mapping to clear"
        )

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
        if self.free_group is None:
            self.swa_attn_allocator.free(free_index[free_index > 0])
        else:
            self.free_group.append(self._copy_for_free_group(free_index))
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.free_group is None:
            self.swa_attn_allocator.free(free_index[free_index > 0])
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

    def free_full(self, free_index: torch.Tensor):
        # All-SWA models have no full-attention pool, so there is nothing to
        # release once the SWA side is gone.
        return

    # Not inherited: the SWA parent's hooks drive swa_free_group and
    # full_free_group, which this pure-SWA variant does not have.
    def free_group_begin(self):
        BaseTokenToKVPoolAllocator.free_group_begin(self)

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(torch.cat(pending))

    def clear(self):
        self.swa_attn_allocator.clear()
        self.free_group = None

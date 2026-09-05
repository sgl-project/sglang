import torch

from sglang.srt.mem_cache.allocator.base import (
    BaseKVPool,
    BaseKVPoolSide,
    KVPoolSide,
    SinglePoolKVAllocator,
    invariant_checks_enabled,
)
from sglang.srt.mem_cache.allocator.hybrid import BaseHybridSWAKVAllocator
from sglang.srt.mem_cache.allocator.paged import PagedKVPool
from sglang.srt.mem_cache.allocator.pairing import MappingTensorPairing
from sglang.srt.mem_cache.allocator.token import TokenedKVPool
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import get_num_new_pages
from sglang.srt.utils.invariants import Bucket, Invariant, IsTrue, expect

_is_npu = is_npu()

if _is_npu:
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedKVPool,
    )


# The swa side releases whatever the pairing points at, so an entry that reads as
# the padding slot would push slot 0 into the SWA free list and hand it out twice.
_SWA_PEER_MAPPED = Invariant("swa.peer_mapped", Bucket.FATAL_UNCONTAINABLE, IsTrue())
# The full side leaves the pairing alone, so a live entry would strand its SWA peer.
_SWA_PEER_RELEASED = Invariant("swa.peer_released", Bucket.GUARD, IsTrue())


class MappedSWAKVPoolSide(BaseKVPoolSide):
    """The SWA pool of a static hybrid, addressed by full slot ids through the
    pairing table."""

    def __init__(self, pool: BaseKVPool, pairing: MappingTensorPairing):
        self.pool = pool
        self.pairing = pairing
        self.page_size = pool.page_size
        # Inside a group this holds the RESOLVED swa ids, not the full ids the
        # caller passed: ownership is settled at enqueue, see free().
        self.free_group = None

    def available_size(self):
        return self.pool.available_size()

    def conserve_available_size(self):
        return self.pool.conserve_available_size()

    def schedulable_available_size(self):
        return self.pool.schedulable_available_size()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.page_size == 1:
            mapping_indices = free_index
            swa_indices = self.pairing.translate(mapping_indices)
            if invariant_checks_enabled():
                expect(_SWA_PEER_MAPPED, swa_indices > 0, msg="caller wants full.free")
        else:
            mapping_indices = self._expand_to_full_pages(free_index)
            swa_indices = self.pairing.translate(mapping_indices)

        self.pairing.clear(mapping_indices)

        if self.free_group is not None:
            # Resolve ownership now. A cache action later in this group may
            # install a new pairing for the same full index.
            self.free_group.append(swa_indices)
            return

        self._release(swa_indices)

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self._release(torch.cat(pending))

    def _release(self, swa_indices: torch.Tensor):
        if self.page_size > 1:
            # Page shape, not liveness: the unallocated tail of a page and the
            # offset pairing of a HiCache load-back read as 0, which the paged
            # free would take for page 0.
            swa_indices = swa_indices[swa_indices > 0]
        self.pool.free(swa_indices)
        assert self.pool.available_size() <= self.pool.size

    def _expand_to_full_pages(self, indices: torch.Tensor) -> torch.Tensor:
        # Duplicates are kept: deduplicating would be a torch.unique whose
        # data-dependent output shape synchronizes the scheduler stream, and
        # every consumer ends in the paged free's own page dedup anyway.
        base = (indices // self.page_size) * self.page_size
        page_offsets = torch.arange(
            self.page_size, dtype=indices.dtype, device=indices.device
        )
        expanded = (base[:, None] + page_offsets[None, :]).reshape(-1)
        if self.pool.debug_mode:
            # Reference unique on CPU: the expansion must cover exactly the
            # touched pages, on every caller's real input.
            got = torch.unique(expanded.cpu() // self.page_size)
            ref = torch.unique(indices.cpu() // self.page_size)
            assert torch.equal(got, ref), "expansion page set mismatch"
        return expanded

    def clear(self):
        self.pool.clear()
        self.pairing.reset()
        self.free_group = None


class PairedFullKVPoolSide(KVPoolSide):
    """The full pool of a static hybrid: a slot is released only once its SWA
    peer is gone (pairing entry == 0)."""

    def __init__(self, pool: BaseKVPool, pairing: MappingTensorPairing):
        super().__init__(pool)
        self.pairing = pairing

    def _expect_peer_released(self, free_index: torch.Tensor):
        # Checked at enqueue: a cache action later in this group may pair the
        # slot again, and that new peer is not this call's to judge.
        if invariant_checks_enabled():
            expect(
                _SWA_PEER_RELEASED,
                self.pairing.mapping[free_index] == 0,
                msg="caller wants the two-sided free",
            )

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self._expect_peer_released(free_index)
        self.pool.free(free_index)
        assert self.pool.available_size() <= self.pool.size

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        if free_index.numel() == 0:
            return
        self._expect_peer_released(free_index)
        self.pool.free_segment(free_index, start_pos=start_pos)

    def free_group_end(self):
        self.pool.free_group_end()
        assert self.pool.available_size() <= self.pool.size


class HybridSWAKVAllocator(BaseHybridSWAKVAllocator):
    """Hybrid allocator with a static full -> swa pairing table."""

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
        self.need_sort = need_sort
        self._kvcache = kvcache

        full_kv_pool = getattr(kvcache, "full_kv_pool", None)
        swa_kv_pool = getattr(kvcache, "swa_kv_pool", None)

        if page_size == 1:
            full_pool = TokenedKVPool(size, dtype, device, full_kv_pool, need_sort)
            swa_pool = TokenedKVPool(size_swa, dtype, device, swa_kv_pool, need_sort)
        else:
            if _is_npu:
                PagedKVPoolClass = NPUPagedKVPool
            else:
                PagedKVPoolClass = PagedKVPool
            full_pool = PagedKVPoolClass(
                size, page_size, dtype, device, full_kv_pool, need_sort
            )
            swa_pool = PagedKVPoolClass(
                size_swa, page_size, dtype, device, swa_kv_pool, need_sort
            )

        self.pairing = MappingTensorPairing(
            size_full=size, page_size=page_size, device=device
        )
        self.sides = {
            ComponentType.SWA: MappedSWAKVPoolSide(swa_pool, self.pairing),
            ComponentType.FULL: PairedFullKVPoolSide(full_pool, self.pairing),
        }

        self.clear()
        self._kvcache.register_mapping(self.pairing.mapping)

    @property
    def full_to_swa_index_mapping(self) -> torch.Tensor:
        return self.pairing.mapping

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
        msg += f"#swa-available-size: {self.swa.available_size()}, "
        msg += f"#full-attn-available-size: {self.full.available_size()}, "
        return msg

    def alloc(self, need_size: int):
        assert self.page_size == 1
        if need_size > self.full.available_size():
            return None
        if need_size > self.swa.available_size():
            return None

        alloc_full_indices = self.full.pool.alloc(need_size)
        alloc_swa_indices = self.swa.pool.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.pairing.set(alloc_full_indices, alloc_swa_indices)
        return alloc_full_indices

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        return (
            num_full_pages <= self.full.available_size() // self.page_size
            and num_swa_pages <= self.swa.available_size() // self.page_size
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

        alloc_full_indices = self.full.pool.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        alloc_swa_indices = self.swa.pool.alloc_extend(
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

        self.pairing.set(alloc_full_indices, alloc_swa_indices)

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
        """Allocate full KV for the whole extend and SWA KV only for the tail."""
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

        alloc_full_indices = self.full.pool.alloc_extend(
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

        alloc_swa_indices = self.swa.pool.alloc_extend(
            swa_prefix_lens,
            swa_prefix_lens_cpu,
            swa_seq_lens,
            swa_seq_lens_cpu,
            swa_last_loc,
            swa_tail_len,
            num_new_pages=num_swa_pages,
        )
        assert alloc_swa_indices is not None

        self.pairing.set(alloc_full_indices[-swa_tail_len:], alloc_swa_indices)
        if swa_tail_len < extend_num_tokens:
            self.pairing.clear(alloc_full_indices[:-swa_tail_len])
        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full.pool.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa.pool.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        self.pairing.set(alloc_full_indices, alloc_swa_indices)
        return alloc_full_indices

    def resize(self, config) -> None:
        size_full = int(config.full_max_total_num_tokens)
        size_swa = int(config.swa_max_total_num_tokens)
        self._size_full = size_full
        self._size_swa = size_swa
        for pool, sz in ((self.full.pool, size_full), (self.swa.pool, size_swa)):
            pool.size = int(sz)
            if self.page_size > 1:
                pool.num_pages = int(sz) // self.page_size
        self.clear()

    def clear(self):
        self.swa.clear()
        self.full.pool.clear()

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )


class PureSWAKVAllocator(SinglePoolKVAllocator):
    """One SWA pool for models whose every layer is sliding-window attention;
    its only side is ``swa``. The attention kernels read a full -> swa mapping
    off the KV pool, so an identity mapping is registered."""

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
        super().__init__(
            TokenedKVPool(size_swa, dtype, device, kvcache.swa_kv_pool, need_sort),
            component=ComponentType.SWA,
        )
        self._kvcache = kvcache
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.arange(size_swa + page_size, dtype=torch.int64, device=device),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

    @property
    def size_swa(self):
        return self.size

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return kv_indices

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )

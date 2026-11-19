"""Paged allocator whose physical reuse is authorized by the SlotGuard owner.

CPU free-page queues mirror the GPU queues in exactly the same order. Tensor
contents are never copied back: callers must propagate explicit PageSpans.
The common TP adapter freezes Pins before budget selection and allocation,
then finishes the epoch before forward. No collective occurs in a getter.
"""

import hashlib
import threading
from collections import deque
from itertools import islice

import torch

from sglang.srt.mem_cache.allocator.page_identity import PageRowTable, PageSpan
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.l1_transfer.slot_guard import (
    EpochError,
    OwnerThreadViolation,
    UnsafeReset,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget


class SlotGuardPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def __init__(self, *args, slot_guard, active=True, **kwargs):
        # Serving starts inactive and calls activate() once warmup has finished.
        self._active = active
        self._max_pages = slot_guard.max_pages
        self.slot_guard = slot_guard
        self._owner_thread = threading.get_ident()
        self._initialized = False
        self._owned_refs = {}
        self._retired_refs = {}
        self._logical_to_ref = {}
        self._ref_to_logical = {}
        self._allocation_sequence = 0
        self._last_allocation_count = 0
        self._allocation_digest = hashlib.sha256(b"SlotGuard allocation counts v1")
        self._protected_owned = set()
        self._epoch = None
        self._epoch_installed = False
        self._common_capacity_tokens = None
        self._guarded_group = None
        self._guarded_group_refs = set()
        self.last_allocation_refs = ()
        super().__init__(*args, **kwargs)
        if self.size <= 0 or self.page_size <= 1 or self.size % self.page_size:
            raise ValueError(
                "SlotGuard requires a positive, page-aligned paged capacity"
            )
        if self.num_pages > slot_guard.max_pages:
            raise ValueError(
                "SlotGuard registration limit is smaller than physical pool"
            )
        self.page_rows = (
            PageRowTable(self.page_size, reset_preflight=self.assert_reset_safe)
            if active
            else None
        )
        self.last_allocation_spans = ()
        self._initialized = True

    def _owner(self):
        if threading.get_ident() != self._owner_thread:
            raise OwnerThreadViolation(
                "physical page operations belong to allocator owner"
            )

    def cpu_page_identity(self):
        return self if self._active else None

    def _quiescent(self, req_pool, tree_cache):
        self._owner()
        if tree_cache.tree_core.root_node.children:
            raise UnsafeReset("activation switch requires an empty tree")
        req_pool.assert_cpu_page_identity_quiescent(self.page_rows)
        if self.free_group is not None or self._epoch is not None:
            raise UnsafeReset("activation switch requires no allocation/free epoch")
        self.assert_reset_safe()
        if len(self.free_pages) + self.num_staged_pages != self.num_pages:
            raise UnsafeReset("activation switch requires all physical pages free")

    def activate(self, req_pool, tree_cache):
        """Owner-only startup after warmup. One initial device readback is legal.

        The buffers, allocator, guard and retained page-row owner never change.
        Capture both actual queues: need_sort=False can retain arbitrary warmup
        free order. No guarded free reconstructs this information from tensors.
        """
        self._quiescent(req_pool, tree_cache)
        tree_cache.assert_cpu_page_identity_supported()
        if self._active:
            raise ValueError("allocator is already active")
        if self.num_pages > self._max_pages:
            raise ValueError("final physical capacity exceeds SlotGuard cap")
        # One startup readback, even if warmup left several staged batches.
        ordered = tuple(self.get_all_free_pages().cpu().tolist())
        free, staged = ordered[: len(self.free_pages)], ordered[len(self.free_pages) :]
        if sorted((*free, *staged)) != list(range(1, self.num_pages + 1)):
            raise UnsafeReset("stock free queues do not cover the physical pool")
        rows = self.page_rows or PageRowTable(
            self.page_size, reset_preflight=self.assert_reset_safe
        )
        req_pool.assert_cpu_page_identity_quiescent(rows)
        self.page_rows = rows
        self._free_cpu, self._staged_cpu = deque(free), list(staged)
        req_pool.set_cpu_page_identity_enabled(rows, True)
        self._active = True

    def assert_allocation_owner(self):
        """Validate a controller's owner before it can bind a CPU collective."""
        self._owner()
        if not self._active:
            raise EpochError("allocation controller requires an active allocator")
        self.slot_guard.allocation_preflight()

    def assert_reset_safe(self):
        """Reject before any tree, request-row or allocator metadata is erased."""
        self._owner()
        if self._owned_refs or self._retired_refs or self._guarded_group is not None:
            raise UnsafeReset(
                "reset requires no local pages, retired pages, or free group"
            )
        self.slot_guard.assert_reset_safe()

    def clear(self):
        if not self._active:
            return super().clear()
        self._owner()
        if self._initialized:
            self.assert_reset_safe()
            # Validate before changing size, queues, or device buffers.
            self.slot_guard.reset(self.slot_guard.pool_epoch + 1)
            self._logical_to_ref.clear()
            self._ref_to_logical.clear()
            self._allocation_sequence = self._last_allocation_count = 0
            self._allocation_digest = hashlib.sha256(b"SlotGuard allocation counts v1")
            self._protected_owned.clear()
            self.page_rows.clear()
        super().clear()
        self._free_cpu = deque(range(1, self.num_pages + 1))
        self._staged_cpu = []
        self._guarded_group = None

    def resize(self, config):
        if not self._active:
            size = config.max_total_num_tokens
            if (
                size <= 0
                or size % self.page_size
                or size // self.page_size > self._max_pages
            ):
                raise ValueError("SlotGuard resize exceeds aligned physical cap")
            return super().resize(config)
        self._owner()
        size = config.max_total_num_tokens
        if size <= 0 or size % self.page_size:
            raise ValueError("SlotGuard resize must preserve whole-page capacity")
        if size // self.page_size > self.slot_guard.max_pages:
            raise ValueError("resize exceeds SlotGuard physical registration limit")
        # clear validates the complete lifetime ledger before mutating dimensions.
        self.clear()
        self.size, self.num_pages = size, size // self.page_size
        self._initialized = False
        try:
            self.clear()
        finally:
            self._initialized = True

    def merge_and_sort_free(self):
        if not self._active:
            return super().merge_and_sort_free()
        self._owner()
        if not self.staged_pages:
            return
        super().merge_and_sort_free()
        self._free_cpu = deque(sorted((*self._free_cpu, *self._staged_cpu)))
        self._staged_cpu.clear()

    def _prepare_allocation(self, count, *, merge_threshold):
        """Peek the exact proposed page batch without merging or consuming it."""
        if count > len(self._free_cpu) + len(self._staged_cpu):
            return None
        if merge_threshold > len(self._free_cpu) and self._staged_cpu:
            # Same ordering as merge_and_sort_free, only under merge pressure.
            planned = sorted((*self._free_cpu, *self._staged_cpu))
        else:
            planned = self._free_cpu
        ids = tuple(islice(planned, count))
        if ids:
            self.slot_guard.allocation_preflight(ids)
        return ids

    def _consume(self, ids):
        # The owner preflighted these FREE pages before the GPU operation.
        # Background operations cannot change their registration eligibility.
        count = len(ids)
        refs = self.slot_guard.register_allocation(ids) if ids else ()
        for _ in ids:
            self._free_cpu.popleft()
        self._owned_refs.update((ref.page_id, ref) for ref in refs)
        self._allocation_sequence += 1
        self._last_allocation_count = count
        self._allocation_digest.update(count.to_bytes(8, "big"))
        for ordinal, ref in enumerate(refs):
            logical = (self.slot_guard.pool_epoch, self._allocation_sequence, ordinal)
            self._logical_to_ref[logical] = ref
            self._ref_to_logical[ref] = logical
        self.last_allocation_refs = refs
        return ids

    def allocation_signature(self):
        """TP must validate this call sequence/count before relying on logical IDs."""
        return (
            self.slot_guard.pool_epoch,
            self._allocation_sequence,
            self._last_allocation_count,
            self._allocation_digest.hexdigest(),
        )

    def logical_pages(self, refs):
        self._owner()
        try:
            return tuple(self._ref_to_logical[ref] for ref in refs)
        except KeyError as error:
            raise ValueError(
                "page reference has no current logical allocation identity"
            ) from error

    def local_page_refs(self, logical_pages):
        self._owner()
        try:
            return tuple(
                self._logical_to_ref[tuple(logical)] for logical in logical_pages
            )
        except KeyError as error:
            raise ValueError(
                "logical page is absent or already physically reclaimed on this rank"
            ) from error

    def free_recent_allocation(self, refs):
        """Abort one allocation's ``last_allocation_refs``, which exclude the
        pre-existing partial prefix page."""
        self._owner()
        if any(self._owned_refs.get(ref.page_id) != ref for ref in refs):
            raise ValueError("aborted allocation pages are no longer locally owned")
        if refs:
            self._free_refs(refs)

    def alloc(self, need_size):
        if not self._active:
            return super().alloc(need_size)
        self._owner()
        self.slot_guard.allocation_preflight()
        if need_size < 0 or need_size % self.page_size:
            raise ValueError("guarded allocation must contain whole pages")
        count = need_size // self.page_size
        ids = self._prepare_allocation(count, merge_threshold=count)
        if ids is None:
            return None
        result = super().alloc(need_size)
        if result is not None:
            self._consume(ids)
            self.last_allocation_spans = (PageSpan(ids, self.page_size, 0, need_size),)
        return result

    def _prefixes(self, prefix_page_spans, lengths):
        if prefix_page_spans is None or len(prefix_page_spans) != len(lengths):
            raise ValueError(
                "guarded allocation needs explicit CPU prefix page identities"
            )
        for span, length in zip(prefix_page_spans, lengths):
            if (
                span.token_offset
                or span.token_length != length
                or span.page_size != self.page_size
            ):
                raise ValueError("CPU prefix page identity length mismatch")
            self.page_refs(span)

    def _allocation_spans(self, prefixes, starts, ends, ids):
        spans, cursor = [], 0
        ps = self.page_size
        for prefix, start, end in zip(prefixes, starts, ends):
            count = -(-end // ps) - -(-start // ps)
            pages = ids[cursor : cursor + count]
            cursor += count
            if start % ps and end > start:
                pages = (prefix.page_ids[-1], *pages)
            spans.append(
                PageSpan(
                    tuple(pages), ps, start % ps if end > start else 0, end - start
                )
            )
        self.last_allocation_spans = tuple(spans)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        num_new_pages=None,
        *,
        prefix_page_spans=None,
    ):
        if not self._active:
            return super().alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages,
            )
        self._owner()
        self.slot_guard.allocation_preflight()
        if prefix_lens_cpu.device.type != "cpu" or seq_lens_cpu.device.type != "cpu":
            raise ValueError("allocation length metadata must already be on CPU")
        starts, ends = prefix_lens_cpu.tolist(), seq_lens_cpu.tolist()
        if (
            len(starts) != len(ends)
            or len(prefix_lens) != len(starts)
            or len(seq_lens) != len(ends)
            or len(last_loc) != len(ends)
            or any(end < start for start, end in zip(starts, ends))
            or sum(end - start for start, end in zip(starts, ends)) != extend_num_tokens
        ):
            raise ValueError("extend CPU geometry does not match token demand")
        self._prefixes(prefix_page_spans, starts)
        count = sum(
            -(-end // self.page_size) - -(-start // self.page_size)
            for start, end in zip(starts, ends)
        )
        if num_new_pages is not None and num_new_pages != count:
            raise ValueError("new page count disagrees with CPU geometry")
        ids = self._prepare_allocation(
            count, merge_threshold=extend_num_tokens // self.page_size + len(starts) + 1
        )
        if ids is None:
            return None
        result = super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            count,
        )
        if result is not None:
            self._consume(ids)
            self._allocation_spans(prefix_page_spans, starts, ends, ids)
        return result

    def alloc_decode(
        self,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        *,
        prefix_last_page_ids=None,
    ):
        if not self._active:
            return super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self._owner()
        self.slot_guard.allocation_preflight()
        if seq_lens_cpu.device.type != "cpu":
            raise ValueError("allocation length metadata must already be on CPU")
        ends = seq_lens_cpu.tolist()
        if (
            len(seq_lens) != len(ends)
            or len(last_loc) != len(ends)
            or any(end < 1 for end in ends)
        ):
            raise ValueError("decode sequence lengths must include the new token")
        starts = [end - 1 for end in ends]
        if prefix_last_page_ids is None or len(prefix_last_page_ids) != len(starts):
            raise ValueError("decode last-page identities do not cover the batch")
        prefixes = []
        for page, start in zip(prefix_last_page_ids, starts):
            if start and page not in self._owned_refs:
                raise ValueError("decode last page has no local physical ownership")
            prefixes.append(
                PageSpan((page,), self.page_size, 0, self.page_size)
                if start
                else PageSpan((), self.page_size, 0, 0)
            )
        count = sum(start % self.page_size == 0 for start in starts)
        ids = self._prepare_allocation(count, merge_threshold=len(ends))
        if ids is None:
            return None
        result = super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if result is not None:
            self._consume(ids)
            self._allocation_spans(prefixes, starts, ends, ids)
        return result

    def page_refs(self, span):
        self._owner()
        if not isinstance(span, PageSpan) or span.page_size != self.page_size:
            raise ValueError("explicit CPU page span is required")
        try:
            return tuple(self._owned_refs[page] for page in span.page_ids)
        except KeyError as error:
            raise ValueError(
                "CPU page span references a page without local ownership"
            ) from error

    def _physical_release(self, refs):
        self._owner()
        if not refs:
            return
        ids = tuple(ref.page_id for ref in refs)
        # PyTorch's pinned-memory allocator retains the staging allocation until
        # the async copy completes. Only these selected pages cross H2D; the
        # existing GPU free list is retained, never reconstructed from CPU.
        pin = torch.device(self.device).type == "cuda"
        tensor = torch.tensor(ids, dtype=torch.int64, pin_memory=pin).to(
            self.device, non_blocking=pin
        )
        PagedTokenToKVPoolAllocator._release_page_ids(self, tensor)
        if self.need_sort:
            self._staged_cpu.extend(ids)
        else:
            self._free_cpu.extendleft(reversed(ids))
        for ref in refs:
            self._retired_refs.pop(ref.page_id, None)
            logical = self._ref_to_logical.pop(ref)
            del self._logical_to_ref[logical]

    def _retire(self, refs):
        decision = self.slot_guard.on_free(refs)
        for ref in refs:
            self._owned_refs.pop(ref.page_id)
            self._protected_owned.discard(ref.page_id)
        self._retired_refs.update(
            (ref.page_id, ref) for ref in (*decision.deferred, *decision.pending)
        )
        self._physical_release(decision.immediately_reusable)

    def _free_refs(self, refs):
        if self._guarded_group is None:
            self._retire(refs)
            return
        # Refuse at entry: a duplicate found at free_group_end wedges the group.
        batch = set(refs)
        if len(batch) != len(refs) or not batch.isdisjoint(self._guarded_group_refs):
            raise ValueError("duplicate page retirement in free group")
        self._guarded_group.append(refs)
        self._guarded_group_refs |= batch

    def free(self, free_index, *, page_span=None):
        if not self._active:
            return super().free(free_index)
        self._owner()
        if free_index.numel() == 0:
            return
        refs = self.page_refs(page_span)
        if page_span.token_length != free_index.numel():
            raise ValueError("CPU free span length disagrees with device tensor")
        self._free_refs(refs)

    def free_segment(self, free_index, *, start_pos, page_span=None):
        if not self._active:
            return super().free_segment(free_index, start_pos=start_pos)
        self._owner()
        if start_pos % self.page_size:
            raise ValueError("free segment start must be page aligned")
        if page_span is not None and page_span.token_offset:
            raise ValueError("free segment CPU span must begin at page boundary")
        self.free(free_index, page_span=page_span)

    def _release_page_ids(self, *page_ids):
        if not self._active:
            return super()._release_page_ids(*page_ids)
        raise RuntimeError("physical release must pass the SlotGuard owner decision")

    def free_group_begin(self):
        if not self._active:
            return super().free_group_begin()
        self._owner()
        if self._guarded_group is not None:
            raise RuntimeError("free groups cannot be nested")
        self._guarded_group = []
        self._guarded_group_refs = set()

    def free_group_end(self):
        if not self._active:
            return super().free_group_end()
        self._owner()
        if self._guarded_group is None:
            raise RuntimeError("no active free group")
        pending = self._guarded_group
        self._guarded_group = None
        for batch in pending:
            self._retire(batch)

    def begin_allocation_epoch(self, epoch):
        self._owner()
        snapshot = self.slot_guard.begin_epoch(epoch)
        self._epoch = epoch
        self._epoch_installed = False
        return snapshot

    def install_allocation_epoch(self, epoch, protected, *, reclaim_refs=None):
        self._owner()
        protected = tuple(protected)
        reclaimed = self.slot_guard.install_epoch_and_drain(
            epoch, protected, reclaim_refs
        )
        self._protected_owned = {
            ref.page_id for ref in protected if ref.page_id in self._owned_refs
        }
        self._physical_release(reclaimed)
        self._epoch_installed = True

    def prevalidate_finish_allocation_epoch(
        self, epoch, *, verified_headroom_pages=None
    ):
        self._owner()
        if self._guarded_group is not None:
            raise EpochError("cannot finish with an undrained free group")
        if self.verify_byte_accounting():
            raise EpochError("physical allocation page accounting mismatch")
        self.slot_guard.prevalidate_finish_epoch(
            epoch, verified_headroom_pages=verified_headroom_pages
        )

    def finish_allocation_epoch(self, epoch, *, verified_headroom_pages=None):
        self.prevalidate_finish_allocation_epoch(
            epoch, verified_headroom_pages=verified_headroom_pages
        )
        self.slot_guard.finish_epoch(
            epoch, verified_headroom_pages=verified_headroom_pages
        )
        self._epoch = None
        self._epoch_installed = False
        self._protected_owned.clear()

    def fault_allocation_epoch(self, epoch):
        self._owner()
        self.slot_guard.fault_allocation_epoch(epoch)
        self._epoch = epoch
        self._epoch_installed = False

    def retired_size(self):
        if not self._active:
            return super().retired_size()
        return len(self._retired_refs) * self.page_size

    def protected_owned_size(self):
        if not self._active:
            return 0
        return len(self._protected_owned) * self.page_size

    def set_common_capacity_tokens(self, tokens):
        """Cap admission at the capacity every rank agreed on; None lifts it."""
        self._common_capacity_tokens = tokens

    def available_and_evictable_size(self, evictable_size):
        if not self._active:
            return self.available_size() + evictable_size
        if not self._epoch_installed:
            raise EpochError(
                "guarded admission requires a frozen common allocation epoch"
            )
        capacity = self.available_size() + max(
            0, evictable_size - self.protected_owned_size()
        )
        return (
            capacity
            if self._common_capacity_tokens is None
            else min(capacity, self._common_capacity_tokens)
        )

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        if not self._active:
            return super().create_prefill_budget(
                tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
            )
        return SlotGuardPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def evict_to_free_tokens(self, tree_cache, num_tokens):
        if not self._active:
            return super().evict_to_free_tokens(tree_cache, num_tokens)
        from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

        self._owner()
        if not self._epoch_installed:
            raise EpochError("guarded eviction/allocation needs a frozen common epoch")
        while tree_cache is not None and tree_cache.supports_prefix_sharing():
            shortfall = num_tokens - self.available_size()
            if shortfall <= 0:
                return True
            # Allocation demand is total capacity needed, not the current shortfall.
            result = tree_cache.evict_for_alloc(KVDemand(full_tokens=num_tokens))
            if result.num_tokens_evicted == 0:
                break
        return self.available_size() >= num_tokens

    def conservation_counts(self):
        """Owner-local pages, actual free pages, retired pages, fixed capacity."""
        if not self._active:
            free = len(self.free_pages) + self.num_staged_pages
            return self.num_pages - free, free, 0, self.num_pages
        return (
            len(self._owned_refs),
            len(self._free_cpu) + len(self._staged_cpu),
            len(self._retired_refs),
            self.num_pages,
        )

    def verify_byte_accounting(self):
        if not self._active:
            return super().verify_byte_accounting()
        owned, free, retired, capacity = self.conservation_counts()
        errors = []
        if owned + free + retired != capacity:
            errors.append(
                f"SlotGuard page conservation: {owned=}, {free=}, {retired=}, {capacity=}"
            )
        device_free = len(self.free_pages) + self.num_staged_pages
        if device_free != free:
            errors.append(
                f"SlotGuard CPU/device free-count mismatch: {free=}, {device_free=}"
            )
        return errors


class SlotGuardPrefillBudget(PrefillBudget):
    """Common-epoch F+E-R admission, including continuing chunks."""

    def _available_and_evictable(self):
        return self.allocator.available_and_evictable_size(
            self.tree_cache.evictable_size()
        )

    def available_chunk_tokens(self, chunk_limit):
        available = min(
            chunk_limit, int(self.remaining_total), int(self.remaining_current)
        )
        available = available // self.page_size * self.page_size
        return available if available > 0 else None

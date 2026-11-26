"""Paged allocator whose physical reuse is authorized by the SlotGuard owner.

CPU free-page queues mirror the GPU queues in exactly the same order. Tensor
contents are never copied back: callers must propagate explicit PageSpans.
The common TP adapter freezes Pins before budget selection and allocation,
then finishes the epoch before forward. No collective occurs in a getter.
Every method runs on the allocator owner thread.
"""

from collections import deque
from itertools import islice

import torch

from sglang.srt.mem_cache.allocator.page_identity import PageRowTable, PageSpan
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.l1_transfer.slot_guard import EpochError, UnsafeReset


class SlotGuardPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def __init__(self, *args, slot_guard, active=True, **kwargs):
        # Serving starts inactive and calls activate() once warmup has finished.
        self._active = active
        self.slot_guard = slot_guard
        self._initialized = False
        self._owned_refs = {}
        self._retired_refs = {}
        self._guarded_group = None
        self.last_allocation_refs = ()
        super().__init__(*args, **kwargs)
        self.page_rows = (
            PageRowTable(self.page_size, reset_preflight=self.assert_reset_safe)
            if active
            else None
        )
        self.last_allocation_spans = ()
        self._initialized = True

    def cpu_page_identity(self):
        return self if self._active else None

    def activate(self, req_pool, tree_cache):
        """Startup switch after warmup; one initial device readback is legal.

        The caller activates once, with no live request rows and every physical
        page free. The buffers, allocator, guard and request pool never change.
        Capture both actual queues: need_sort=False can retain arbitrary warmup
        free order. No guarded free reconstructs this information from tensors.
        """
        tree_cache.assert_cpu_page_identity_supported()
        # One startup readback, even if warmup left several staged batches.
        ordered = tuple(self.get_all_free_pages().cpu().tolist())
        if sorted(ordered) != list(range(1, self.num_pages + 1)):
            raise UnsafeReset("activation requires every physical page to be free")
        rows = PageRowTable(self.page_size, reset_preflight=self.assert_reset_safe)
        req_pool.attach_cpu_page_identity(rows)
        self.page_rows = rows
        free_count = len(self.free_pages)
        self._free_cpu = deque(ordered[:free_count])
        self._staged_cpu = list(ordered[free_count:])
        self._active = True

    def assert_allocation_owner(self):
        """Validate a controller's owner before it can bind a CPU collective."""
        if not self._active:
            raise EpochError("allocation controller requires an active allocator")
        self.slot_guard.allocation_preflight()

    def assert_reset_safe(self):
        """Reject before any tree, request-row or allocator metadata is erased."""
        if self._owned_refs or self._retired_refs or self._guarded_group is not None:
            raise UnsafeReset(
                "reset requires no local pages, retired pages, or free group"
            )
        self.slot_guard.assert_reset_safe()

    def clear(self):
        # Construction and resize both reach here with the final page count.
        if self.num_pages > self.slot_guard.max_pages:
            raise ValueError(
                "SlotGuard registration limit is smaller than physical pool"
            )
        if not self._active:
            return super().clear()
        if self._initialized:
            # The ledger validates the lifetime state before anything changes;
            # callers end any open free group first.
            self.slot_guard.reset(self.slot_guard.pool_epoch + 1)
            self.page_rows.clear()
        super().clear()
        self._free_cpu = deque(range(1, self.num_pages + 1))
        self._staged_cpu = []
        self._guarded_group = None

    def resize(self, config):
        if not self._active:
            return super().resize(config)
        # clear validates the complete lifetime ledger before mutating dimensions.
        self.clear()
        self.size = config.max_total_num_tokens
        self.num_pages = self.size // self.page_size
        self._initialized = False
        try:
            self.clear()
        finally:
            self._initialized = True

    def merge_and_sort_free(self):
        if not self._active:
            return super().merge_and_sort_free()
        if not self.staged_pages:
            return
        super().merge_and_sort_free()
        self._free_cpu = deque(sorted((*self._free_cpu, *self._staged_cpu)))
        self._staged_cpu.clear()

    def _prepare_allocation(self, count, *, merge_threshold):
        """Peek the exact proposed page batch without merging or consuming it;
        None when the pool cannot serve ``count`` pages."""
        if merge_threshold > len(self._free_cpu) and self._staged_cpu:
            # Same ordering as merge_and_sort_free, only under merge pressure.
            planned = sorted((*self._free_cpu, *self._staged_cpu))
        else:
            planned = self._free_cpu
        ids = tuple(islice(planned, count))
        # Rejects a frozen epoch and non-FREE pages before any queue changes.
        self.slot_guard.allocation_preflight(ids)
        return ids if len(ids) == count else None

    def _consume(self, ids):
        # The owner preflighted these FREE pages before the GPU operation.
        # Background operations cannot change their registration eligibility.
        refs = self.slot_guard.register_allocation(ids) if ids else ()
        for _ in ids:
            self._free_cpu.popleft()
        self._owned_refs.update((ref.page_id, ref) for ref in refs)
        self.last_allocation_refs = refs

    def free_recent_allocation(self, refs):
        """Abort one allocation's ``last_allocation_refs``, which exclude the
        pre-existing partial prefix page."""
        if any(self._owned_refs.get(ref.page_id) != ref for ref in refs):
            raise ValueError("aborted allocation pages are no longer locally owned")
        if refs:
            self._free_refs(refs)

    def alloc(self, need_size):
        if not self._active:
            return super().alloc(need_size)
        # Callers pass whole pages; the stock allocator asserts it in debug mode.
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
        if prefix_lens_cpu.device.type != "cpu" or seq_lens_cpu.device.type != "cpu":
            raise ValueError("allocation length metadata must already be on CPU")
        starts, ends = prefix_lens_cpu.tolist(), seq_lens_cpu.tolist()
        if sum(end - start for start, end in zip(starts, ends)) != extend_num_tokens:
            raise ValueError("extend CPU geometry does not match token demand")
        self._prefixes(prefix_page_spans, starts)
        # The CPU geometry fixes the page count; ``num_new_pages`` is not needed.
        count = sum(
            -(-end // self.page_size) - -(-start // self.page_size)
            for start, end in zip(starts, ends)
        )
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
        """Every sequence length already includes the new token."""
        if not self._active:
            return super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if seq_lens_cpu.device.type != "cpu":
            raise ValueError("allocation length metadata must already be on CPU")
        ends = seq_lens_cpu.tolist()
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
        if not isinstance(span, PageSpan) or span.page_size != self.page_size:
            raise ValueError("explicit CPU page span is required")
        try:
            return tuple(self._owned_refs[page] for page in span.page_ids)
        except KeyError as error:
            raise ValueError(
                "CPU page span references a page without local ownership"
            ) from error

    def _physical_release(self, refs):
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

    def _retire(self, refs):
        reusable = self.slot_guard.on_free(refs)
        for ref in refs:
            self._owned_refs.pop(ref.page_id)
        retired = set(refs).difference(reusable)
        self._retired_refs.update((ref.page_id, ref) for ref in retired)
        self._physical_release(reusable)

    def _free_refs(self, refs):
        # A repeated free is refused by the ledger when the free group ends.
        if self._guarded_group is None:
            self._retire(refs)
        else:
            self._guarded_group.append(refs)

    def free(self, free_index, *, page_span=None):
        if not self._active:
            return super().free(free_index)
        if free_index.numel() == 0:
            return
        refs = self.page_refs(page_span)
        if page_span.token_length != free_index.numel():
            raise ValueError("CPU free span length disagrees with device tensor")
        self._free_refs(refs)

    def free_segment(self, free_index, *, start_pos, page_span=None):
        if not self._active:
            return super().free_segment(free_index, start_pos=start_pos)
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
        if self._guarded_group is not None:
            raise RuntimeError("free groups cannot be nested")
        self._guarded_group = []

    def free_group_end(self):
        if not self._active:
            return super().free_group_end()
        if self._guarded_group is None:
            raise RuntimeError("no active free group")
        pending = self._guarded_group
        self._guarded_group = None
        for batch in pending:
            self._retire(batch)

    def begin_allocation_epoch(self, epoch):
        return self.slot_guard.begin_epoch(epoch)

    def install_allocation_epoch(self, epoch, protected, *, reclaim_refs=None):
        reclaimed = self.slot_guard.install_epoch(epoch, tuple(protected), reclaim_refs)
        self._physical_release(reclaimed)

    def prevalidate_finish_allocation_epoch(self, epoch):
        if self._guarded_group is not None:
            raise EpochError("cannot finish with an undrained free group")
        if self.verify_byte_accounting():
            raise EpochError("physical allocation page accounting mismatch")
        self.slot_guard.prevalidate_finish_epoch(epoch)

    def finish_allocation_epoch(self, epoch, *, verified_headroom_pages):
        self.prevalidate_finish_allocation_epoch(epoch)
        self.slot_guard.finish_epoch(
            epoch, verified_headroom_pages=verified_headroom_pages
        )

    def fault_allocation_epoch(self, epoch):
        self.slot_guard.fault_allocation_epoch(epoch)

    def retired_size(self):
        if not self._active:
            return super().retired_size()
        return len(self._retired_refs) * self.page_size

    def conservation_counts(self):
        """Owner-local pages, actual free pages, retired pages, fixed capacity."""
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

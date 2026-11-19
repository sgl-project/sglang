"""Physical allocator and CPU page identity contracts (real torch CPU path)."""

import threading
import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.page_identity import PageRowTable, PageSpan
from sglang.srt.mem_cache.l1_transfer.slot_guard import (
    EpochError,
    OwnerThreadViolation,
    PageUnavailable,
    UnsafeReset,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.ledger import (
    allocator,
    epoch,
    lease_identity,
    owned,
    runtime_fixture,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def empty_tree():
    return SimpleNamespace(
        tree_core=SimpleNamespace(root_node=SimpleNamespace(children={})),
        assert_cpu_page_identity_supported=lambda: None,
    )


def extend_batch(cache, alloc, rows, length):
    """A one-request, no-prefix extend batch over the runtime fixture."""
    from sglang.srt.managers.schedule_batch import ReqKvInfo

    req = SimpleNamespace(kv=ReqKvInfo(), prefix_len=0, session=None)
    return req, SimpleNamespace(
        maybe_evict_swa=lambda: None,
        is_dllm=lambda: False,
        reqs=[req],
        tree_cache=cache,
        req_to_token_pool=rows,
        token_to_kv_pool_allocator=alloc,
        device="cpu",
        prefix_lens=[0],
        extend_lens=[length],
        seq_lens=torch.tensor([length]),
        seq_lens_cpu=torch.tensor([length]),
        extend_num_tokens=length,
        hisparse_coordinator=None,
        model_config=SimpleNamespace(is_encoder_decoder=False),
    )


def run_extend(batch):
    from sglang.srt.mem_cache.allocation import alloc_for_extend

    with patch(
        "sglang.srt.mem_cache.allocation.attention_backends",
        return_value=("torch_native", "torch_native"),
    ):
        _, batch.req_pool_indices, batch.req_pool_indices_cpu = alloc_for_extend(batch)


class PageIdentityTest(CustomTestCase):
    def test_slice_partial_page_and_concat_preserve_identity(self):
        span = PageSpan((8, 3, 5), 4, 0, 11)
        self.assertEqual(span.slice(3, 9), PageSpan((8, 3, 5), 4, 3, 6))
        self.assertEqual(span.slice(4, 8), PageSpan((3,), 4, 0, 4))
        self.assertEqual(PageSpan.concat((span.slice(0, 3), span.slice(3, 11))), span)
        with self.assertRaises(ValueError):
            PageSpan.concat((span.slice(0, 3), PageSpan((99,), 4, 3, 1)))

    def test_row_shadow_is_page_granular_and_rebinds_canonical_prefix(self):
        rows = PageRowTable(4)
        rows.set(1, PageSpan((8, 3), 4, 0, 7))
        rows.write(1, 4, PageSpan((9, 2), 4, 0, 6))
        self.assertEqual(rows.get(1, 0, 10), PageSpan((8, 9, 2), 4, 0, 10))
        rows.clear(1)
        with self.assertRaises(ValueError):
            rows.get(1, 0, 1)

    def test_row_write_keeps_identity_of_pages_it_partly_covers(self):
        """A write that leaves recorded tokens of a page unwritten cannot
        relabel that page, and a negative start cannot wrap to the row end."""
        rows = PageRowTable(4)
        rows.set(1, PageSpan((8, 3), 4, 0, 7))
        before = rows.get(1, 0, 7)
        writes = {
            "head_of_full_page": (0, PageSpan((9,), 4, 0, 1)),
            "head_of_tail_page": (4, PageSpan((9,), 4, 0, 2)),
            "negative_start": (-4, PageSpan((9,), 4, 0, 4)),
        }
        for name, (start, span) in writes.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    rows.write(1, start, span)
                self.assertEqual(rows.get(1, 0, 7), before)


class GuardedAllocatorTest(CustomTestCase):
    def test_inactive_warmup_resize_then_ordered_quiescent_activation(self):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        alloc = allocator(pages=4, active=False)
        rows = ReqToTokenPool(2, 32, "cpu", False)
        pool = alloc.get_kvcache()
        guard = alloc.slot_guard
        self.assertIsNone(alloc.cpu_page_identity())
        with self.assertRaises(ValueError):
            alloc.resize(SimpleNamespace(max_total_num_tokens=20))
        alloc.resize(SimpleNamespace(max_total_num_tokens=12))
        with patch.object(guard, "allocation_preflight", side_effect=AssertionError):
            first, second = alloc.alloc(4), alloc.alloc(4)
            alloc.free(first)
            alloc.free(second)
            self.assertEqual(alloc.verify_byte_accounting(), [])
            alloc.create_prefill_budget(None)
        self.assertEqual(alloc.free_pages.tolist(), [2, 1, 3])
        rows.free_rows(rows.alloc_rows(1))
        generation = rows.req_generation.clone()
        alloc.activate(rows, empty_tree())
        self.assertIs(alloc.get_kvcache(), pool)
        self.assertIs(alloc.slot_guard, guard)
        self.assertTrue(torch.equal(rows.req_generation, generation))
        self.assertEqual(tuple(alloc._free_cpu), (2, 1, 3))
        alloc.alloc(4)
        self.assertEqual(alloc.last_allocation_spans[0].page_ids, (2,))

    def test_activation_rejects_live_request_rows_before_mutation(self):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        for live in ("allocated_row", "duplicate_free_row"):
            with self.subTest(live=live):
                alloc = allocator(pages=4, active=False)
                rows = ReqToTokenPool(2, 32, "cpu", False)
                if live == "allocated_row":
                    rows.alloc_rows(1)
                else:
                    rows.free_slots = [1, 1]
                with self.assertRaises(ValueError):
                    alloc.activate(rows, empty_tree())
                self.assertIsNone(alloc.cpu_page_identity())
                self.assertIsNone(rows.cpu_page_rows)

    def test_paths_without_cpu_page_spans_are_refused(self):
        """HiCache load-back, document reuse and request-private linker loads
        put Full KV on the device without a CPU page span; page identity
        refuses them at activation, at attach time, or before allocation."""
        features = {
            "hicache": ("cache_controller", object()),
            "doc_reuse": ("uocr_doc_reuse", object()),
            "layer_wise_linker": ("linker", SimpleNamespace(load_mode="layer_wise")),
        }
        for name, (field, value) in features.items():
            with self.subTest(activate=name):
                cache, alloc, rows = runtime_fixture(active=False)
                setattr(cache, field, value)
                with self.assertRaises(ValueError):
                    alloc.activate(rows, cache)
                self.assertIsNone(alloc.cpu_page_identity())
                self.assertIsNone(rows.cpu_page_rows)

        def prefix_span(cache):
            cache.prefix_cpu_page_span(
                SimpleNamespace(
                    rid="r", last_node=cache.root_node_handle(), prefix_len=0
                )
            )

        def private_load(cache):
            cache.linker = SimpleNamespace(private_load_indices={"r": torch.arange(4)})
            prefix_span(cache)

        def doc_reuse(cache):
            # Enabling reuse stays allowed (upstream builds caches without an
            # allocator); the prefix that would take its slots is refused.
            cache.uocr_doc_reuse = object()
            try:
                prefix_span(cache)
            finally:
                cache.uocr_doc_reuse = None

        uses = {
            "hicache": lambda cache: cache.init_hicache(None, None),
            "doc_reuse": doc_reuse,
            "private_load": private_load,
        }
        for name, use in uses.items():
            with self.subTest(attached=name):
                cache, _, _ = runtime_fixture()
                with self.assertRaisesRegex(ValueError, "SlotGuard v1"):
                    use(cache)
                self.assertIsNone(cache.cache_controller)
                self.assertIsNone(cache.uocr_doc_reuse)

    def test_rejected_reclaim_batch_leaves_the_epoch_installable(self):
        """An install whose reclaim batch is rejected must change nothing,
        so the same epoch can be installed with a corrected batch."""
        alloc = allocator(pages=4)
        indices, span, refs = owned(alloc, 2)
        alloc.free(indices[:4], page_span=span.slice(0, 4))
        alloc.begin_allocation_epoch(1)
        before = alloc.slot_guard.owner_counts()
        with self.assertRaises(PageUnavailable):
            # Page 2 is still locally owned.
            alloc.install_allocation_epoch(1, (), reclaim_refs=refs[1:])
        self.assertEqual(alloc.slot_guard.owner_counts(), before)
        alloc.install_allocation_epoch(1, (), reclaim_refs=refs[:1])
        self.assertEqual(alloc.available_and_evictable_size(0), 12)

    def test_duplicate_group_free_is_refused_without_wedging_the_group(self):
        """A duplicate free inside a free group fails on its own; the group
        still ends, and a new group can begin."""
        alloc = allocator(pages=4)
        indices, span, _ = owned(alloc, 2)
        epoch(alloc)
        alloc.free_group_begin()
        alloc.free(indices[:4], page_span=span.slice(0, 4))
        with self.assertRaises(ValueError):
            alloc.free(indices[:4], page_span=span.slice(0, 4))
        alloc.free(indices[4:], page_span=span.slice(4, 8))
        alloc.free_group_end()
        self.assertEqual(alloc.conservation_counts(), (0, 4, 0, 4))
        alloc.free_group_begin()
        alloc.free_group_end()

    def test_all_free_entrypoints_intercept_and_deferred_pages_are_not_free(self):
        for entry in ("free", "free_segment", "free_group_end"):
            with self.subTest(entry=entry):
                alloc = allocator(pages=4)
                indices, span, refs = owned(alloc, 2)
                alloc.slot_guard.try_pin(lease_identity(), refs)
                epoch(alloc)
                if entry == "free_group_end":
                    alloc.free_group_begin()
                    alloc.free(indices, page_span=span)
                    alloc.free_group_end()
                elif entry == "free_segment":
                    alloc.free_segment(indices, start_pos=0, page_span=span)
                else:
                    alloc.free(indices, page_span=span)
                self.assertEqual(alloc.available_size(), 8)
                self.assertEqual(alloc.retired_size(), 8)
                self.assertEqual(set(alloc.get_all_free_pages().tolist()), {3, 4})
                self.assertEqual(alloc.conservation_counts(), (0, 2, 2, 4))

    def test_free_without_cpu_identity_fails_before_any_allocator_mutation(self):
        alloc = allocator(pages=4)
        indices, _, _ = owned(alloc, 1)
        with self.assertRaises(ValueError):
            alloc.free(indices)
        # The stock page-id entry has no CPU identity and must not bypass the guard.
        with self.assertRaises(RuntimeError):
            alloc.free_page_ids(indices[::4] // 4)
        self.assertEqual(alloc.available_size(), 12)
        self.assertEqual(alloc.conservation_counts(), (1, 3, 0, 4))

    def test_budget_charges_shared_protection_once_and_never_forces_a_chunk(self):
        alloc = allocator()
        indices, span, refs = owned(alloc, 90)
        alloc.slot_guard.try_pin(lease_identity(), refs[:40])
        alloc.slot_guard.try_pin(lease_identity(1), refs[:40])
        evictable = [90 * 4]
        budget = alloc.create_prefill_budget(
            SimpleNamespace(evictable_size=lambda: evictable[0])
        )
        snap = alloc.begin_allocation_epoch(1)
        with self.assertRaises(EpochError):
            _ = budget.remaining_total
        alloc.install_allocation_epoch(1, snap.held_pages)
        # F10 + E90 - R40, with two leases on the same 40 pages charged once.
        self.assertEqual(budget.remaining_total, 60 * 4)
        self.assertEqual(alloc.protected_owned_size(), 40 * 4)
        self.assertIsNotNone(alloc.alloc(10 * 4))
        evictable[0] = 40 * 4
        self.assertEqual(budget.remaining_total, 0)
        self.assertIsNone(budget.available_chunk_tokens(16))
        alloc.free(indices[: 40 * 4], page_span=span.slice(0, 40 * 4))
        self.assertEqual(alloc.protected_owned_size(), 0)

    def test_free_outside_epoch_only_queues_common_reclaim(self):
        alloc = allocator(pages=4)
        indices, span, refs = owned(alloc, 2)
        alloc.free(indices, page_span=span)
        self.assertEqual(alloc.available_size(), 8)
        epoch(alloc, protected=refs[:1])
        self.assertEqual(alloc.available_size(), 12)
        alloc.finish_allocation_epoch(1)
        epoch(alloc, 2)
        self.assertEqual(alloc.available_size(), 16)
        self.assertEqual(alloc.conservation_counts(), (0, 4, 0, 4))

    def test_group_union_without_local_reader_protects_and_charges_local_owner(self):
        alloc = allocator(pages=4)
        indices, span, refs = owned(alloc, 2)
        epoch(alloc, protected=refs[:1])
        self.assertEqual(alloc.available_and_evictable_size(8), 12)
        alloc.free(indices, page_span=span)
        self.assertEqual(alloc.available_size(), 12)
        self.assertEqual(alloc.retired_size(), 4)

    def test_only_owner_may_free_and_resize_rejects_retired_pages(self):
        alloc = allocator(pages=4)
        indices, span, _ = owned(alloc, 1)
        errors = []

        def wrong_owner():
            try:
                alloc.free(indices, page_span=span)
            except OwnerThreadViolation as error:
                errors.append(error)

        thread = threading.Thread(target=wrong_owner)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        alloc.free(indices, page_span=span)
        with self.assertRaises(UnsafeReset):
            alloc.resize(SimpleNamespace(max_total_num_tokens=8))
        self.assertEqual(alloc.size, 16)
        epoch(alloc)
        alloc.finish_allocation_epoch(1)
        alloc.resize(SimpleNamespace(max_total_num_tokens=8))
        self.assertEqual(alloc.available_size(), 8)

    def test_no_device_readback_for_free_or_epoch_and_need_sort_mirror(self):
        alloc = allocator(pages=4, need_sort=True)
        indices, span, _ = owned(alloc, 3)
        epoch(alloc)
        with (
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("readback")
            ),
        ):
            alloc.free(indices[:4], page_span=span.slice(0, 4))
            self.assertEqual(alloc.available_size(), 8)
            reused = alloc.alloc(8)
        self.assertEqual(reused[::4].tolist(), [4, 16])
        self.assertEqual(alloc.last_allocation_spans[0].page_ids, (1, 4))

    def test_rejected_allocation_leaves_queues_and_counts_unchanged(self):
        def extend(alloc, prefix, end):
            start = prefix.token_length
            return alloc.alloc_extend(
                torch.tensor([start]),
                torch.tensor([start]),
                torch.tensor([end]),
                torch.tensor([end]),
                torch.tensor([prefix.page_ids[-1] * 4 + (start - 1) % 4]),
                max(end - start, 0),
                prefix_page_spans=(prefix,),
            )

        def decode_in_partial_page(alloc):
            return alloc.alloc_decode(
                torch.tensor([4]),
                torch.tensor([4]),
                torch.tensor([10]),
                prefix_last_page_ids=(2,),
            )

        def queues(alloc):
            return (
                alloc.get_all_free_pages().tolist(),
                alloc.num_staged_pages,
                alloc.conservation_counts(),
                alloc.allocation_signature(),
            )

        def freeze(alloc):
            alloc.begin_allocation_epoch(2)

        def freeze_guard_only(alloc):
            alloc.slot_guard.begin_epoch(2)

        def register_staged_page_behind_allocator(alloc):
            alloc.slot_guard.register_allocation((1,))

        # (case, prepare, rejected allocation, error); zero-page cases included.
        cases = (
            ("alloc", freeze, lambda a: a.alloc(4), EpochError),
            ("alloc_zero_guard_epoch", freeze_guard_only,
                lambda a: a.alloc(0), EpochError),
            ("extend_zero", freeze,
                lambda a: extend(a, PageSpan((2,), 4, 0, 3), 4), EpochError),
            ("decode_zero", freeze, decode_in_partial_page, EpochError),
            ("extend_geometry", lambda a: None,
                lambda a: extend(a, PageSpan((2, 3), 4, 0, 7), 3), ValueError),
            ("unreclaimed_under_merge", register_staged_page_behind_allocator,
                lambda a: a.alloc(8), PageUnavailable),
        )  # fmt: skip
        for name, prepare, allocate, error in cases:
            with self.subTest(name):
                alloc = allocator(pages=4, need_sort=True)
                indices, span, _ = owned(alloc, 3)
                epoch(alloc)
                # Page 1 is staged and page 4 free, so a 2-page request must merge.
                alloc.free_segment(indices[:4], start_pos=0, page_span=span.slice(0, 4))
                alloc.finish_allocation_epoch(1)
                prepare(alloc)
                before = queues(alloc)
                with self.assertRaises(error):
                    allocate(alloc)
                self.assertEqual(queues(alloc), before)

    def test_allocation_signature_binds_complete_count_history(self):
        left, right = allocator(pages=16), allocator(pages=16)
        for count in (1, 2, 3):
            left.alloc(count * 4)
        for count in (2, 1, 3):
            right.alloc(count * 4)
        self.assertNotEqual(left.allocation_signature(), right.allocation_signature())

    def test_tp_logical_page_mapping_does_not_assume_equal_physical_ids(self):
        left, right = allocator(pages=4), allocator(pages=4)
        right._free_cpu = deque(reversed(right._free_cpu))
        right.free_pages = right.free_pages.flip(0)
        _, left_span, left_refs = owned(left, 2)
        right_indices, right_span, right_refs = owned(right, 2)
        self.assertNotEqual(left_span.page_ids, right_span.page_ids)
        logical = left.logical_pages(left_refs)
        self.assertEqual(logical, right.logical_pages(right_refs))
        self.assertEqual(right.local_page_refs(logical), right_refs)
        epoch(right, protected=right_refs[:1])
        right.free(right_indices, page_span=right_span)
        self.assertEqual(right.local_page_refs(logical[:1]), right_refs[:1])
        with self.assertRaises(ValueError):
            right.local_page_refs(logical[1:])


class GuardedRuntimeIdentityTest(CustomTestCase):
    def test_actual_tree_split_and_duplicate_insert_keep_cpu_identity(self):
        from sglang.srt.mem_cache.base_prefix_cache import (
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.radix_cache import RadixKey

        cache, alloc, _ = runtime_fixture()
        epoch(alloc)
        tokens = array("q", range(12))
        values = alloc.alloc(12)
        span = alloc.last_allocation_spans[0]
        cache.insert(InsertParams(key=RadixKey(tokens), value=values, page_span=span))
        cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens[:4])))
        duplicate = alloc.alloc(12)
        dup_span = alloc.last_allocation_spans[0]
        before = alloc.available_size()
        cache.insert(
            InsertParams(key=RadixKey(tokens), value=duplicate, page_span=dup_span)
        )
        self.assertEqual(alloc.available_size(), before + 12)
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(cache.path_cpu_page_span(match.last_device_node), span)
        self.assertEqual(alloc.conservation_counts(), (3, 13, 0, 16))

    def test_actual_row_free_uses_shadow_and_unannotated_write_fails_closed(self):
        cache, alloc, rows = runtime_fixture()
        values = alloc.alloc(7 * 4)
        span = alloc.last_allocation_spans[0]
        rows.attach_cpu_page_identity(alloc.page_rows)
        with self.assertRaises(ValueError):
            rows.write((1, slice(0, 28)), values)
        rows.write((1, slice(0, 28)), values, cpu_page_writes=((1, 0, span),))
        epoch(alloc)
        kv = SimpleNamespace(
            req_pool_idx=1,
            restore_done_event=None,
            get_evicted_seqlen=lambda _: 0,
            swa_dead_lo=lambda _: 0,
        )
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")):
            cache.free_kv_row(kv, [(4, 11), (12, 28)])
        self.assertEqual(alloc.conservation_counts(), (1, 15, 0, 16))

    def test_actual_alloc_for_extend_and_decode_update_cpu_shadow(self):
        from sglang.srt.managers.schedule_batch import ReqKvInfo
        from sglang.srt.mem_cache.allocation import (
            alloc_for_decode_default,
            alloc_for_extend,
        )

        cache, alloc, rows = runtime_fixture()
        epoch(alloc)
        req = SimpleNamespace(kv=ReqKvInfo(), prefix_len=0, session=None)
        batch = SimpleNamespace(
            maybe_evict_swa=lambda: None,
            is_dllm=lambda: False,
            reqs=[req],
            tree_cache=cache,
            req_to_token_pool=rows,
            token_to_kv_pool_allocator=alloc,
            device="cpu",
            prefix_lens=[0],
            extend_lens=[3],
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            extend_num_tokens=3,
            hisparse_coordinator=None,
            model_config=SimpleNamespace(is_encoder_decoder=False),
        )
        with patch(
            "sglang.srt.mem_cache.allocation.attention_backends",
            return_value=("torch_native", "torch_native"),
        ):
            out, batch.req_pool_indices, batch.req_pool_indices_cpu = alloc_for_extend(
                batch
            )
        row = req.kv.req_pool_idx
        self.assertEqual(alloc.page_rows.get(row, 0, 3), PageSpan((1,), 4, 0, 3))
        out = alloc_for_decode_default(batch, 1)
        self.assertEqual(out.tolist(), [7])
        self.assertEqual(alloc.page_rows.get(row, 0, 4), PageSpan((1,), 4, 0, 4))
        batch.seq_lens = batch.seq_lens_cpu = torch.tensor([4])
        out = alloc_for_decode_default(batch, 1)
        self.assertEqual(out.tolist(), [8])
        self.assertEqual(alloc.page_rows.get(row, 0, 5), PageSpan((1, 2), 4, 0, 5))

    def test_failed_row_write_or_aux_allocation_records_no_cpu_row(self):
        """When extend or decode rolls back its pages after a failed aux
        allocation, or a device row write fails, the CPU row must not cover
        the written positions: a later row free would hit returned pages."""
        from sglang.srt.mem_cache.allocation import alloc_for_decode_default

        for step in ("extend", "decode"):
            with self.subTest(step):
                cache, alloc, rows = runtime_fixture()
                epoch(alloc)
                req, batch = extend_batch(cache, alloc, rows, 4)
                if step == "decode":
                    run_extend(batch)
                with (
                    patch.object(
                        rows, "alloc_aux_to_lengths", side_effect=RuntimeError("aux")
                    ),
                    self.assertRaisesRegex(RuntimeError, "aux"),
                ):
                    if step == "extend":
                        run_extend(batch)
                    else:
                        alloc_for_decode_default(batch, 1)
                kept = 4 if step == "decode" else 0
                self.assertEqual(
                    alloc.conservation_counts(), (kept // 4, 16 - kept // 4, 0, 16)
                )
                row = req.kv.req_pool_idx
                if kept:
                    self.assertEqual(
                        alloc.page_rows.get(row, 0, kept), PageSpan((1,), 4, 0, 4)
                    )
                with self.assertRaises(ValueError):
                    alloc.page_rows.get(row, 0, kept + 1)

        with self.subTest("device_write"):
            cache, alloc, rows = runtime_fixture()
            row = rows.alloc_rows(1)[0]
            values = alloc.alloc(8)
            span = alloc.last_allocation_spans[0]
            with self.assertRaises(RuntimeError):
                rows.write(
                    (row, slice(0, 8)), values[:4], cpu_page_writes=((row, 0, span),)
                )
            with self.assertRaises(ValueError):
                alloc.page_rows.get(row, 0, 1)

    def test_encoder_decoder_rows_are_refused_before_allocation(self):
        """Encoder-decoder device rows are offset by encoder_lens while CPU
        rows use decoder positions; guarded allocation must refuse them."""
        from sglang.srt.mem_cache.allocation import alloc_for_decode_default

        for step in ("extend", "decode"):
            with self.subTest(step):
                cache, alloc, rows = runtime_fixture()
                epoch(alloc)
                _, batch = extend_batch(cache, alloc, rows, 4)
                if step == "decode":
                    run_extend(batch)
                batch.model_config = SimpleNamespace(is_encoder_decoder=True)
                batch.encoder_lens = torch.tensor([4])
                before = alloc.conservation_counts()
                with self.assertRaisesRegex(ValueError, "encoder-decoder"):
                    if step == "extend":
                        run_extend(batch)
                    else:
                        alloc_for_decode_default(batch, 1)
                self.assertEqual(alloc.conservation_counts(), before)

    def test_prompt_boundary_checkpoint_rebinds_rows_to_canonical_tree_spans(self):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.sampling.sampling_params import SamplingParams

        for guarded in (True, False):
            with self.subTest(guarded=guarded):
                cache, alloc, rows = runtime_fixture(pages=64, guarded=guarded)
                if guarded:
                    epoch(alloc)

                def new_req(name, prompt_tokens, total_tokens, last_node):
                    req = Req(
                        name,
                        "x",
                        array("q", range(prompt_tokens)),
                        SamplingParams(max_new_tokens=total_tokens - prompt_tokens),
                    )
                    req.full_untruncated_fill_ids = array("q", range(total_tokens))
                    req.last_node = last_node
                    req.kv.req_pool_idx = rows.alloc_rows(1)[0]
                    req.kv.kv_allocated_len = req.kv.kv_committed_len = total_tokens
                    return req

                def write_row(req, values, span):
                    row = req.kv.req_pool_idx
                    kwargs = {"cpu_page_writes": ((row, 0, span),)} if guarded else {}
                    rows.write((row, slice(0, len(values))), values, **kwargs)

                def assert_row_is_canonical(req, total_tokens):
                    canonical = cache.path_device_indices(req.last_node)
                    row = req.kv.req_pool_idx
                    self.assertTrue(
                        torch.equal(rows.req_to_token[row, :total_tokens], canonical)
                    )
                    if not guarded:
                        return canonical, None
                    adopted = alloc.page_rows.get(row, 0, total_tokens)
                    self.assertEqual(adopted, cache.path_cpu_page_span(req.last_node))
                    self.assertEqual(
                        adopted.page_ids, tuple((canonical[::4] // 4).tolist())
                    )
                    return canonical, adopted

                canonical_prefix = None
                for attempt, (prompt_tokens, total_tokens) in enumerate(
                    ((8, 12), (8, 12), (5, 32))
                ):
                    req = new_req(
                        f"split-{attempt}",
                        prompt_tokens,
                        total_tokens,
                        cache.root_node_handle(),
                    )
                    values = alloc.alloc(total_tokens)
                    write_row(
                        req, values, alloc.last_allocation_spans[0] if guarded else None
                    )
                    cache.checkpoint_into_tree(req, up_to=total_tokens)
                    canonical, adopted = assert_row_is_canonical(req, total_tokens)
                    if canonical_prefix is None:
                        canonical_prefix = canonical[:12].clone()
                    self.assertTrue(torch.equal(canonical[:12], canonical_prefix))
                    self.assertEqual(req.kv.cache_protected_len, total_tokens)
                    self.assertEqual(req.lock.node, req.last_node)
                    self.assertEqual(cache.protected_size(), total_tokens)
                    self.assertEqual(alloc.available_size() + total_tokens, alloc.size)
                    if guarded:
                        free_pages = set(alloc.get_all_free_pages().tolist())
                        self.assertFalse(free_pages.intersection(adopted.page_ids))
                        self.assertEqual(
                            alloc.conservation_counts(),
                            (total_tokens // 4, 64 - total_tokens // 4, 0, 64),
                        )
                    cache.unlock(req.lock)
                    self.assertEqual(cache.protected_size(), 0)
                    self.assertEqual(cache.evictable_size(), total_tokens)
                    rows.free_rows([req.kv.req_pool_idx])

                # A request locking only its prompt finds its private continuation
                # already canonical in the tree; checkpoint must swap its lock.
                prefix_node = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", range(8))))
                ).last_device_node
                req = new_req("locked-split", 8, 36, prefix_node)
                req.lock = old_lock = cache.lock(prefix_node)
                req.kv.cache_protected_len = 8
                prefix_values = cache.path_device_indices(prefix_node)
                private_values = alloc.alloc(28)
                span = None
                if guarded:
                    prefix_span = cache.path_cpu_page_span(prefix_node)
                    span = PageSpan.concat(
                        (prefix_span, alloc.last_allocation_spans[0])
                    )
                write_row(req, torch.cat((prefix_values, private_values)), span)
                self.assertEqual(cache.protected_size(), 8)
                cache.checkpoint_into_tree(req, up_to=36)
                self.assertIsNot(req.lock, old_lock)
                self.assertEqual(req.lock.node, req.last_node)
                self.assertEqual(cache.protected_size(), 36)
                assert_row_is_canonical(req, 36)
                cache.unlock(req.lock)
                self.assertEqual(cache.protected_size(), 0)
                self.assertEqual(cache.evictable_size(), 36)

    def test_actual_urc_lru_skips_no_physical_free_and_continues_same_epoch(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        cache, alloc, _ = runtime_fixture(pages=100)
        for attempt, pages in enumerate((40, 50)):
            values = alloc.alloc(pages * 4)
            span = alloc.last_allocation_spans[0]
            if attempt == 0:
                refs = alloc.page_refs(span)
                alloc.slot_guard.publish(refs)
                alloc.slot_guard.try_pin(lease_identity(), refs)
            cache.insert(
                InsertParams(
                    key=RadixKey(
                        array("q", range(attempt * 1000, attempt * 1000 + pages * 4))
                    ),
                    value=values,
                    page_span=span,
                )
            )
        epoch(alloc)
        self.assertTrue(alloc.evict_to_free_tokens(cache, 60 * 4))
        self.assertEqual(alloc.available_size(), 60 * 4)
        self.assertEqual(alloc.retired_size(), 40 * 4)
        self.assertIsNotNone(alloc.alloc(60 * 4))
        self.assertEqual(alloc.conservation_counts(), (60, 0, 40, 100))

    def test_actual_stats_and_invariant_account_retired_without_changing_capacity(self):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )

        cache, alloc, rows = runtime_fixture()
        indices = alloc.alloc(4)
        span = alloc.last_allocation_spans[0]
        alloc.free(indices, page_span=span)
        cache.kv_storage()._shape = SimpleNamespace(
            has_window=False, has_state=False, token_capacity=64
        )
        observer = SchedulerPoolStatsObserver(
            tree_cache=cache,
            storage=cache.kv_storage(),
            token_to_kv_pool_allocator=alloc,
            session_controller=SimpleNamespace(sessions={}),
            hisparse_coordinator=None,
            enable_hisparse=False,
        )
        stats = observer.get_pool_stats()
        self.assertEqual(stats.full_retired_size, 4)
        self.assertEqual(stats.full_available_size, 60)
        self.assertEqual(alloc.size, 64)
        leak, message = SchedulerInvariantChecker._check_pool_invariant(
            "full",
            stats.full_available_size,
            stats.full_evictable_size,
            0,
            0,
            64,
            retired=stats.full_retired_size,
        )
        self.assertFalse(leak, message)
        self.assertIn("retired=4", message)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_reset_entries_reject_before_destroying_tree_rows_or_linker(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        for state in ("live", "retired", "guard_epoch"):
            cache, alloc, rows = runtime_fixture()
            row = rows.alloc_rows(1)[0]
            span = None
            if state == "guard_epoch":
                alloc.slot_guard.begin_epoch(1)
            else:
                values = alloc.alloc(8)
                span = alloc.last_allocation_spans[0]
                rows.write(
                    (row, slice(0, 8)), values, cpu_page_writes=((row, 0, span),)
                )
                if state == "live":
                    refs = alloc.page_refs(span)
                    alloc.slot_guard.publish(refs)
                    alloc.slot_guard.try_pin(lease_identity(), refs)
                    key = RadixKey(array("q", range(8)))
                    cache.insert(InsertParams(key=key, value=values, page_span=span))
                else:
                    alloc.free(values, page_span=span)
            linker_calls = []
            cache.linker = SimpleNamespace(
                reset=lambda calls=linker_calls: calls.append("reset")
            )

            def observed():
                return (
                    cache.tree_core.root_node,
                    cache.evictable_size(),
                    list(rows.free_slots),
                    alloc.conservation_counts(),
                    None if span is None else alloc.page_rows.get(row, 0, 8),
                )

            expected = observed()
            entries = {
                "cache.reset": cache.reset,
                "cache._reset_full": cache._reset_full,
                "rows.clear": rows.clear,
                "page_rows.clear": alloc.page_rows.clear,
                "allocator.clear": alloc.clear,
            }
            census = AssertionError("reset preflight took a full census")
            with patch.object(alloc.slot_guard, "snapshot", side_effect=census):
                for entry, reset in entries.items():
                    with self.subTest(state=state, entry=entry):
                        with self.assertRaises(UnsafeReset):
                            reset()
                        self.assertEqual(observed(), expected)
                        self.assertEqual(linker_calls, [])


if __name__ == "__main__":
    unittest.main()

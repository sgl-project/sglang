"""Real shared-pool allocation after allocation-driven cache eviction."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_swa_cache,
    setup_allocator_context,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestUnifiedJointAllocationEviction(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def tearDown(self):
        reset_context()

    build_cache = staticmethod(build_swa_cache)

    def mark_prefixes(self, allocator, cache, occupancy, page_size=1):
        records = []
        kv = allocator.get_kvcache()
        for i in range(occupancy):
            key = RadixKey(array("q", range(i * 1000, i * 1000 + page_size)))
            slots = cache.match_prefix(
                MatchPrefixParams(key=key)
            ).device_indices.clone()
            self.assertEqual(len(slots), page_size)
            for layer, member in enumerate(
                (allocator.full_attn_allocator, allocator.swa_attn_allocator)
            ):
                ids = member.translate_kv_loc(slots)
                marker = slots.to(torch.float16).view(-1, 1, 1) + 1000 * layer
                kv.get_key_buffer(layer)[ids] = marker
                kv.get_value_buffer(layer)[ids] = marker + 100
            records.append((key, slots))
        return records

    def check_prefixes(self, allocator, cache, records, expected):
        kv = allocator.get_kvcache()
        kept = []
        for i, (key, slots) in enumerate(records):
            actual = cache.match_prefix(MatchPrefixParams(key=key)).device_indices
            if i not in expected:
                self.assertEqual(len(actual), 0, i)
                continue
            kept.append(i)
            torch.testing.assert_close(actual, slots)
            for layer, member in enumerate(
                (allocator.full_attn_allocator, allocator.swa_attn_allocator)
            ):
                ids = member.translate_kv_loc(slots)
                marker = slots.to(torch.float16).view(-1, 1, 1) + 1000 * layer
                self.assertTrue(torch.all(kv.get_key_buffer(layer)[ids] == marker))
                self.assertTrue(
                    torch.all(kv.get_value_buffer(layer)[ids] == marker + 100)
                )
        self.assertEqual(kept, list(expected))
        self.assertFalse(allocator.verify_byte_accounting())

    def test_joint_reclaim_stops_after_actual_cascade(self):
        for lazy in (False, True):
            for page_size in (1, 4, 16):
                for demand, first in ((3, 0), (4, 1), (6, 3), (90, 87)):
                    with self.subTest(lazy=lazy, page_size=page_size, demand=demand):
                        allocator, cache = self.build_cache(
                            lazy=lazy, page_size=page_size
                        )
                        records = self.mark_prefixes(allocator, cache, 96, page_size)
                        self.assertTrue(
                            evict_from_tree_cache(cache, demand * page_size)
                        )
                        self.assertIsNotNone(allocator.alloc(demand * page_size))
                        self.check_prefixes(allocator, cache, records, range(first, 96))

    def test_impossible_byte_demand_preserves_cache(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                allocator, cache = self.build_cache(lazy=lazy)
                records = self.mark_prefixes(allocator, cache, 96)
                self.assertIsNone(allocator.reclaim_plan(101, 101, empty_pool=True))
                for _ in range(3):
                    evict_from_tree_cache(cache, 101)
                    self.assertIsNone(allocator.alloc(101))
                self.check_prefixes(allocator, cache, records, range(96))

    def test_pending_groups_reclaim_without_prefix_eviction(self):
        for lazy in (False, True):
            for mode, page_size in (("ordinary", 1), ("representatives", 4)):
                with self.subTest(lazy=lazy, mode=mode):
                    allocator, cache = self.build_cache(
                        occupancy=94, lazy=lazy, page_size=page_size
                    )
                    records = self.mark_prefixes(allocator, cache, 94, page_size)
                    temporary = allocator.alloc(2 * page_size)
                    allocator.free_group_begin()
                    if mode == "ordinary":
                        allocator.free(temporary)
                    else:
                        allocator.free_segment(temporary, start_pos=0)
                    self.assertLess(allocator.available_size(), 4 * page_size)
                    self.assertTrue(evict_from_tree_cache(cache, 4 * page_size))
                    out = allocator.alloc(4 * page_size)
                    self.assertIsNotNone(out)
                    self.assertEqual(allocator.free_group, [])
                    self.assertEqual(allocator.free_page_reps_group, [])
                    self.check_prefixes(allocator, cache, records, range(94))
                    allocator.free_group_end()
                    allocator.free(out)
                    cache.evict(EvictParams(num_tokens=94 * page_size))
                    self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
                    self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)
                    self.assertFalse(allocator.verify_byte_accounting())

    def test_locked_prefixes_preserve_cache(self):
        allocator, cache = self.build_cache()
        records = self.mark_prefixes(allocator, cache, 96)
        locks = []
        for key, _ in records:
            node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
            locks.append((node, cache.inc_lock_ref(node)))
        try:
            evict_from_tree_cache(cache, 4)
            self.assertIsNone(allocator.alloc(4))
            self.check_prefixes(allocator, cache, records, range(96))
        finally:
            for node, lock in locks:
                cache.dec_lock_ref(node, lock.to_dec_params())
        self.assertTrue(evict_from_tree_cache(cache, 4))
        self.assertIsNotNone(allocator.alloc(4))
        self.check_prefixes(allocator, cache, records, range(1, 96))

    def test_explicit_and_legacy_allocation_eviction_keep_count_contract(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                allocator, cache = self.build_cache()
                records = self.mark_prefixes(allocator, cache, 96)
                method = cache.evict_for_alloc if legacy else cache.evict
                result = method(EvictParams(num_tokens=2))
                self.assertEqual(result.num_tokens_evicted, 2)
                self.check_prefixes(allocator, cache, records, range(2, 96))

    def test_schedule_batch_decode_uses_allocator_contract(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        allocator, cache = self.build_cache()
        records = self.mark_prefixes(allocator, cache, 96)
        batch = object.__new__(ScheduleBatch)
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = cache
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        batch.reqs = [
            SimpleNamespace(kv=SimpleNamespace(kv_committed_len=8), beam_group=None)
            for _ in range(6)
        ]
        self.assertTrue(batch.check_decode_mem())
        self.assertIsNotNone(allocator.alloc(6))
        self.check_prefixes(allocator, cache, records, range(3, 96))

    def test_exact_extend_demand_preserves_other_allocator_dispatch(self):
        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedMambaSWATokenToKVPoolAllocator,
            UnifiedSWAAllocatorBase,
        )

        allocator, _ = self.build_cache(occupancy=0, page_size=4)
        prefix = torch.tensor([1, 4, 7])
        seq = torch.tensor([3, 5, 9])
        kwargs = dict(conservative_num_tokens=19, shard_size=1)
        self.assertEqual(
            allocator.get_extend_allocation_demand(prefix, seq, **kwargs), 8
        )
        self.assertEqual(
            allocator.get_extend_allocation_demand(
                prefix, seq, conservative_num_tokens=31, shard_size=2
            ),
            31,
        )
        self.assertIs(
            UnifiedSWAAllocatorBase.get_extend_allocation_demand,
            BaseTokenToKVPoolAllocator.get_extend_allocation_demand,
        )
        self.assertEqual(
            UnifiedMambaSWATokenToKVPoolAllocator.get_extend_allocation_demand(
                allocator, prefix, seq, **kwargs
            ),
            8,
        )
        self.assertEqual(
            BaseTokenToKVPoolAllocator.get_extend_allocation_demand(
                allocator, prefix, seq, **kwargs
            ),
            19,
        )

    def test_page128_partial_extend_and_conservative_fallback(self):
        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedMambaSWATokenToKVPoolAllocator,
        )

        allocator, _ = self.build_cache(occupancy=0, page_size=128)
        for prefix, seq, expected in (
            ([0], [1], 128),
            ([127], [128], 0),
            ([128], [129], 128),
            ([129], [130], 0),
            ([127], [129], 128),
            ([0, 127, 128, 129], [1, 128, 129, 257], 384),
        ):
            with self.subTest(prefix=prefix, seq=seq):
                p, q = torch.tensor(prefix), torch.tensor(seq)
                for query in (
                    type(allocator).get_extend_allocation_demand,
                    UnifiedMambaSWATokenToKVPoolAllocator.get_extend_allocation_demand,
                ):
                    self.assertEqual(
                        query(
                            allocator, p, q, conservative_num_tokens=1024, shard_size=1
                        ),
                        expected,
                    )
                    self.assertEqual(
                        query(
                            allocator, p, q, conservative_num_tokens=1024, shard_size=2
                        ),
                        1024,
                    )
                self.assertEqual(
                    BaseTokenToKVPoolAllocator.get_extend_allocation_demand(
                        allocator, p, q, conservative_num_tokens=1024, shard_size=1
                    ),
                    1024,
                )

    def test_pending_full_only_reclaim_is_needed_before_walk(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                allocator, cache = self.build_cache(occupancy=94, lazy=lazy)
                records = self.mark_prefixes(allocator, cache, 94)
                extra = allocator.full_attn_allocator.alloc(4)
                self.assertIsNotNone(extra)
                allocator.free_group_begin()
                allocator.free_full(extra[:2])
                self.assertLess(allocator.available_size(), 4)
                self.assertTrue(evict_from_tree_cache(cache, 4))
                out = allocator.alloc(4)
                self.assertIsNotNone(out)
                self.check_prefixes(allocator, cache, records, range(94))
                allocator.free_group_end()
                allocator.free(out)
                allocator.free_full(extra[2:])
                cache.evict(EvictParams(num_tokens=94))
                self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
                self.assertFalse(allocator.verify_byte_accounting())

    def test_shared_virtual_id_limit_is_independent_of_swa_bytes(self):
        allocator, cache = self.build_cache(occupancy=0, ratio=(4, 1))
        fa, sa = allocator.full_attn_allocator, allocator.swa_attn_allocator
        demand = fa.num_virtual_ids - fa.min_page_index + 1
        # Asymmetric SWA restore demand fits bytes/physical SWA slots, but
        # cannot address more virtual pages than its FULL ID owner provides.
        self.assertLess(
            demand * sa.entry_bytes_per_page, allocator._empty_shared_gap_bytes
        )
        self.assertLess(demand, sa.num_pages - sa.min_page_index)
        self.assertIsNone(allocator.reclaim_plan(0, demand, empty_pool=True))
        self.assertFalse(allocator.ensure_capacity(0, demand))
        self.assertEqual(cache.full_evictable_size(), 0)
        self.assertFalse(allocator.verify_byte_accounting())

    def test_exact_extend_caller_reclaims_only_new_pages(self):
        from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend

        class CPUReferenceKernel:
            # Only the external Triton index-generation kernel is replaced;
            # the actual caller, capacity checks and page bindings run on CPU.
            def __getitem__(self, grid):
                def launch(prefix, seq, last, free, out, batch_size, page_size):
                    next_page = offset = 0
                    for i in range(len(prefix)):
                        for pos in range(int(prefix[i]), int(seq[i])):
                            if pos % page_size == 0:
                                page = int(free[next_page])
                                next_page += 1
                            elif pos == int(prefix[i]):
                                page = int(last[i]) // page_size
                            out[offset] = page * page_size + pos % page_size
                            offset += 1

                return launch

        cases = [
            (2, [0], [396], 0),
            (0, [0], [396], 0),
            (94, [1, 4, 7], [3, 5, 9], 1),
            (96, [1], [3], 0),
            (96, [4], [8], 0),
        ]
        for occupancy, prefixes, lengths, evicted in cases:
            with self.subTest(occupancy=occupancy, prefixes=prefixes, lengths=lengths):
                allocator, cache = self.build_cache(occupancy=occupancy, page_size=4)
                records = self.mark_prefixes(allocator, cache, occupancy, 4)
                prefix_slots = [
                    allocator.alloc((n + 3) // 4 * 4)
                    if n
                    else torch.empty(0, dtype=torch.int64)
                    for n in prefixes
                ]
                for slots in prefix_slots:
                    self.assertIsNotNone(slots)
                last = torch.tensor(
                    [
                        int(slots[n - 1]) if n else -1
                        for slots, n in zip(prefix_slots, prefixes)
                    ],
                    dtype=torch.int64,
                )
                prefix = torch.tensor(prefixes, dtype=torch.int64)
                seq = torch.tensor(lengths, dtype=torch.int64)
                extra_tokens = sum(b - a for a, b in zip(prefixes, lengths))
                before_pages = allocator.full_attn_allocator.allocated_count() // 4
                with patch(
                    "sglang.srt.mem_cache.allocator.unified_sub_pool.alloc_extend_kernel",
                    CPUReferenceKernel(),
                ):
                    out = alloc_paged_token_slots_extend(
                        cache, prefix, prefix, seq, seq, last, extra_tokens
                    )
                self.assertEqual(len(out), extra_tokens)
                self.assertEqual(len(set(out.tolist())), extra_tokens)
                expected_pages = sum(
                    (b + 3) // 4 - (a + 3) // 4 for a, b in zip(prefixes, lengths)
                )
                actual_evicted = occupancy if lengths == [396] else evicted
                self.assertEqual(
                    allocator.full_attn_allocator.allocated_count() // 4,
                    before_pages - actual_evicted + expected_pages,
                )
                offset = 0
                for n, end, slots in zip(prefixes, lengths, prefix_slots):
                    reused = min(end, (n + 3) // 4 * 4) - n
                    if reused:
                        torch.testing.assert_close(
                            out[offset : offset + reused], slots[n : n + reused]
                        )
                    offset += end - n
                self.check_prefixes(
                    allocator, cache, records, range(actual_evicted, occupancy)
                )

    def test_reclaim_predicate_does_not_drain_pending_groups_or_move_pages(self):
        allocator, cache = self.build_cache(occupancy=94, lazy=True)
        temporary = allocator.alloc(2)
        allocator.free_swa(temporary[:1])
        allocator.free_group_begin()
        allocator.free(temporary)

        def snapshot():
            groups = [
                [t.tolist() for t in group]
                for group in (
                    allocator.free_group,
                    allocator.free_page_reps_group,
                    allocator.full_free_group,
                )
            ]
            members = []
            for member in allocator._flush_targets():
                members.append(
                    (
                        member.virtual_to_physical.tolist(),
                        member.physical_to_virtual.tolist(),
                        member._free_phys_pages.tolist(),
                        member._byte_low_frontier(),
                        member._byte_high_frontier(),
                        member.allocated_count(),
                    )
                )
            return groups, members

        before = snapshot()
        for gate in (False, True):
            for member in allocator._flush_targets():
                member.disagg_move_gate = lambda: gate
            for _ in range(20):
                allocator.reclaim_plan(4, 4)
            self.assertEqual(snapshot(), before)
        for member in allocator._flush_targets():
            member.disagg_move_gate = None
        self.assertTrue(evict_from_tree_cache(cache, 4))
        self.assertIsNotNone(allocator.alloc(4))
        self.assertEqual(cache.full_evictable_size(), 94)
        allocator.free_group_end()
        self.assertFalse(allocator.verify_byte_accounting())

    def _check_internal_swa_reclaim(self, *, sessions):
        allocator, cache = self.build_cache(occupancy=0, sessions=sessions, window=1)
        # Each incoming branch owns its duplicate prefix until insert
        # deduplicates it: two of these 98 slots will be returned.
        slots = allocator.alloc(98)
        branch_keys = []
        for offset, first in ((0, 10), (6, 20)):
            first_key = RadixKey(array("q", [first, first + 1, first + 2]))
            second_key = RadixKey(array("q", [first, first + 3, first + 4]))
            branch_keys.extend((first_key, second_key))
            cache.insert(
                InsertParams(
                    key=first_key,
                    value=slots[offset : offset + 3],
                )
            )
            cache.insert(
                InsertParams(
                    key=second_key,
                    value=slots[offset + 3 : offset + 6],
                )
            )
        for i in range(12, 98):
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", [1000 + i])), value=slots[i : i + 1]
                )
            )
        # A short SWA window refreshes the two-token child, leaving
        # its internal ancestor as the oldest SWA reclaim candidate.
        for key in branch_keys:
            cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 96)
        self.assertEqual(allocator.available_size(), 3)
        kv = allocator.get_kvcache()
        for layer, member in enumerate(allocator._flush_targets()):
            ids = member.translate_kv_loc(slots[12:13])
            kv.get_key_buffer(layer)[ids] = 17 + layer
            kv.get_value_buffer(layer)[ids] = 117 + layer
        original = cache._evict_device_next_node
        observed = []

        def step(component, tracker):
            result = original(component, tracker)
            observed.append((result, allocator.reclaim_plan(4, 4) == (0, 0)))
            return result

        with patch.object(cache, "_evict_device_next_node", side_effect=step):
            cache.evict_for_alloc(
                EvictParams(swa_num_tokens=2),
                allocation_reclaim_satisfied=lambda: (
                    allocator.reclaim_plan(4, 4) == (0, 0)
                ),
            )
        self.assertTrue(
            any(node is None and progress for (node, progress), _ in observed),
            observed,
        )
        self.assertTrue(observed[-1][1])
        self.assertEqual(len(observed), 1)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 96)
        # The arena's half-pair slack plus one SWA page funds the
        # demand. Do not consume the second internal victim/quota.
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 95)
        self.assertIsNotNone(allocator.alloc(4))
        # The next independent leaf survives the internal tombstones.
        actual = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1012])))
        ).device_indices
        torch.testing.assert_close(actual, slots[12:13])
        for layer, member in enumerate(allocator._flush_targets()):
            ids = member.translate_kv_loc(actual)
            self.assertTrue(torch.all(kv.get_key_buffer(layer)[ids] == 17 + layer))
            self.assertTrue(torch.all(kv.get_value_buffer(layer)[ids] == 117 + layer))
        self.assertFalse(allocator.verify_byte_accounting())

    def test_internal_swa_reclaim_drains_before_joint_stop(self):
        self._check_internal_swa_reclaim(sessions=False)

    def test_session_cursor_stops_after_internal_reclaim(self):
        self._check_internal_swa_reclaim(sessions=True)

    def test_rank_consensus_ignores_callback_identity(self):
        import os
        import subprocess
        import sys

        script = r"""
import queue, runpy, sys, threading
from unittest.mock import patch
from sglang.srt.utils import rank_consensus_checker as checker
module = runpy.run_path(sys.argv[1], run_name="rank_fixture")
fixture = module["TestUnifiedJointAllocationEviction"]()
traces = []
for _ in range(2):
    fixture.setUp()
    try:
        allocator, cache = fixture.build_cache()
        records = fixture.mark_prefixes(allocator, cache, 96)
        q = queue.Queue()
        with patch.object(checker, "_q", q), patch.object(checker, "_scheduler_thread", threading.current_thread()):
            assert module["evict_from_tree_cache"](cache, 6)
            assert allocator.alloc(6) is not None
        fixture.check_prefixes(allocator, cache, records, range(3, 96))
        traces.append(list(q.queue))
    finally:
        fixture.tearDown()
assert traces[0] == traces[1], traces
assert any("num_tokens=6" in event for event in traces[0]), traces
assert any("allocation_reclaim_satisfied is not None=True" in event for event in traces[0]), traces
assert not any("<function" in event for event in traces[0]), traces
"""
        env = dict(os.environ, SGLANG_ENABLE_RANK_CONSENSUS_CHECKER="1")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, __file__],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

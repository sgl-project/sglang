"""Joint token reclaim with live SWA and Mamba state in a shared byte pool."""

import itertools
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_unified_float_move_gate import build_geometry, payload, snapshot

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import (
    init_unified_mamba_swa_pools,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_tri_cache,
    setup_allocator_context,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestTriJointReclaim(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def tearDown(self):
        reset_context()

    build = staticmethod(build_tri_cache)

    def stamp(self, b, a, slots, states, full_layers=1):
        kv = b.token_to_kv_pool
        for layer in range(kv.layer_num):
            member = (
                a.full_attn_allocator if layer < full_layers else a.swa_attn_allocator
            )
            ids = member.translate_kv_loc(slots)
            values = slots.to(torch.float16).view(-1, 1, 1) + layer * 100
            kv.get_key_buffer(layer)[ids] = values
            kv.get_value_buffer(layer)[ids] = values + 500
        mc = b.req_to_token_pool.mamba_pool.mamba_cache
        for j, view in enumerate([*mc.conv, mc.temporal]):
            if view.numel():
                mids = a.mamba_allocator.translate_kv_loc(states)
                for i, mid in enumerate(mids):
                    view[:, mid] = i + 10 * j + 1

    def check_payload(self, b, a, slots, states, original_states, full_layers=1):
        kv = b.token_to_kv_pool
        for layer in range(kv.layer_num):
            member = (
                a.full_attn_allocator if layer < full_layers else a.swa_attn_allocator
            )
            self.assertTrue(
                torch.all(member.virtual_to_physical[slots // a.page_size] > 0)
            )
            ids = member.translate_kv_loc(slots)
            values = slots.to(torch.float16).view(-1, 1, 1) + layer * 100
            torch.testing.assert_close(
                kv.get_key_buffer(layer)[ids],
                values.expand_as(kv.get_key_buffer(layer)[ids]),
            )
            torch.testing.assert_close(
                kv.get_value_buffer(layer)[ids],
                (values + 500).expand_as(kv.get_value_buffer(layer)[ids]),
            )
        mc = b.req_to_token_pool.mamba_pool.mamba_cache
        self.assertTrue(torch.all(a.mamba_allocator.virtual_to_physical[states] > 0))
        for j, view in enumerate([*mc.conv, mc.temporal]):
            if view.numel():
                mids = a.mamba_allocator.translate_kv_loc(states)
                for v, mid in zip(states.tolist(), mids):
                    i = original_states.tolist().index(v)
                    self.assertTrue(torch.all(view[:, mid] == i + 10 * j + 1))

    def insert_parts(self, c, slots, states, *, page_size=1):
        records = []
        for i, (lo, hi) in enumerate(
            ((0, page_size), (page_size, 2 * page_size), (2 * page_size, len(slots)))
        ):
            key = RadixKey(array("q", range(i * 10000, i * 10000 + hi - lo)))
            c.insert(
                InsertParams(key=key, value=slots[lo:hi], mamba_value=states[i : i + 1])
            )
            records.append((key, slots[lo:hi].clone()))
        return records

    def test_joint_reclaim_keeps_live_prefix_and_state(self):
        for lazy in (False, True):
            for occupancy, count, temporal, outside_count in (
                (96, 8, (0, 0, 0), 0),
                (80, 24, (0, 0, 0), 0),
                (96, 5, (1, 4, 8), 1),
            ):
                with self.subTest(lazy=lazy, occupancy=occupancy, states=count):
                    b, a, c = self.build(lazy=lazy, temporal=temporal)
                    slots = a.alloc(occupancy)
                    states = b.req_to_token_pool.mamba_allocator.alloc(count)
                    outside = a.alloc(outside_count)
                    self.assertIsNotNone(slots)
                    self.assertIsNotNone(states)
                    self.assertIsNotNone(outside)
                    self.stamp(b, a, torch.cat((slots, outside)), states)
                    records = self.insert_parts(c, slots, states)
                    demand = 4 if count == 8 else a.available_size() + 1
                    if count == 24:
                        demand = min(a.full_available_size(), a.swa_available_size())
                    with patch.object(
                        a, "ensure_capacity", wraps=a.ensure_capacity
                    ) as prep:
                        evict_from_tree_cache(c, demand)
                        got = a.alloc(demand)
                    self.assertIsNotNone(got)
                    self.assertLessEqual(prep.call_count, 4)
                    expected = (
                        [] if count == 24 else ([1, 2] if count == 8 else [0, 1, 2])
                    )
                    for i, (key, prior) in enumerate(records):
                        match = c.match_prefix(
                            MatchPrefixParams(key=key)
                        ).device_indices
                        if i in expected:
                            torch.testing.assert_close(match, prior)
                            self.check_payload(b, a, match, states[i : i + 1], states)
                        else:
                            self.assertEqual(len(match), 0)
                    self.check_payload(b, a, outside, states[3:], states)
                    if count == 8:
                        self.assertEqual(c.full_evictable_size(), 95)
                    self.assertFalse(a.verify_byte_accounting())

    def test_impossible_and_locked_demands_preserve_cache(self):
        for lazy in (False, True):
            for locked in (False, True):
                with self.subTest(lazy=lazy, locked=locked):
                    b, a, c = self.build(lazy=lazy)
                    slots = a.alloc(96)
                    states = b.req_to_token_pool.mamba_allocator.alloc(8)
                    self.stamp(b, a, slots, states)
                    records = self.insert_parts(c, slots, states)
                    receipts = []
                    if locked:
                        for key, _ in records:
                            node = c.match_prefix(
                                MatchPrefixParams(key=key)
                            ).last_device_node
                            receipts.append((node, c.inc_lock_ref(node)))
                    try:
                        with patch.object(
                            c, "evict_for_alloc", wraps=c.evict_for_alloc
                        ) as walks:
                            for _ in range(3):
                                evict_from_tree_cache(c, 4 if locked else 200)
                        self.assertEqual(walks.call_count, 0)
                        for key, prior in records:
                            torch.testing.assert_close(
                                c.match_prefix(
                                    MatchPrefixParams(key=key)
                                ).device_indices,
                                prior,
                            )
                        self.check_payload(b, a, slots, states, states)
                    finally:
                        for node, receipt in receipts:
                            c.dec_lock_ref(node, receipt.to_dec_params())
                    self.assertFalse(a.verify_byte_accounting())

    def test_state_phase_preserves_locked_token_bindings(self):
        b, a, c = self.build(temporal=(1, 4, 8))
        states = b.req_to_token_pool.mamba_allocator.alloc(8)
        slots = a.alloc(96)
        self.assertIsNotNone(states)
        self.assertIsNotNone(slots)
        self.stamp(b, a, slots, states)
        records = self.insert_parts(c, slots, states)
        receipts = []
        for key, _ in records:
            node = c.match_prefix(MatchPrefixParams(key=key)).last_device_node
            receipts.append(
                (
                    node,
                    c.inc_lock_ref(node, skip_lock_components=(ComponentType.MAMBA,)),
                )
            )
        demand = min(a.conserve_full_available_size(), a.conserve_swa_available_size())
        try:
            self.assertEqual(c.full_evictable_size(), 0)
            self.assertGreater(c.mamba_evictable_size(), 0)
            phases = []
            original = c.evict_for_alloc

            def walk(params, **kw):
                phases.append(params)
                return original(params, **kw)

            with patch.object(c, "evict_for_alloc", side_effect=walk):
                evict_from_tree_cache(c, demand)
                got = a.alloc(demand)
            self.assertIsNotNone(got)
            self.assertEqual(len(phases), 1)
            self.assertEqual(phases[0].num_tokens, 0)
            self.assertGreater(phases[0].mamba_num, 0)
            for (node, _), (_, prior) in zip(receipts, records):
                torch.testing.assert_close(
                    c.tree_core.get_component_device_value(node, ComponentType.FULL),
                    prior,
                )
            self.check_payload(b, a, slots, states[3:], states)
        finally:
            for node, receipt in receipts:
                c.dec_lock_ref(node, receipt.to_dec_params())
        self.assertFalse(a.verify_byte_accounting())

    def test_exact_extend_demand_avoids_impossible_probe(self):
        from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend

        class CPUReferenceKernel:
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

        for occupancy in (0, 2):
            with self.subTest(cached_pages=occupancy):
                b, a, c = self.build(page_size=4, state_cache=False)
                states = b.req_to_token_pool.mamba_allocator.alloc(8)
                self.assertIsNotNone(states)
                slots = a.alloc(occupancy * 4)
                self.assertIsNotNone(slots)
                self.stamp(b, a, slots, states)
                for i in range(occupancy):
                    c.insert(
                        InsertParams(
                            key=RadixKey(array("q", range(i * 100, i * 100 + 4))),
                            value=slots[i * 4 : (i + 1) * 4],
                        )
                    )
                self.assertGreater(
                    a._token_allocation_byte_shortfall(
                        400,
                        full_reclaim=c.full_evictable_size(),
                        swa_reclaim=c.swa_evictable_size(),
                        state_reclaim=c.mamba_evictable_size(),
                    ),
                    0,
                )
                prefix = torch.tensor([0])
                seq = torch.tensor([396])
                last = torch.tensor([-1])
                with patch(
                    "sglang.srt.mem_cache.allocator.unified_sub_pool.alloc_extend_kernel",
                    CPUReferenceKernel(),
                ):
                    got = alloc_paged_token_slots_extend(
                        c, prefix, prefix, seq, seq, last, 396
                    )
                self.assertEqual(len(got), 396)
                self.assertEqual(len(set(got.tolist())), 396)
                self.check_payload(b, a, slots[:0], states, states)
                self.assertFalse(a.verify_byte_accounting())

    def test_grouped_frees_preserve_cache(self):
        for grouped in ("ordinary", "representatives", "full_only"):
            with self.subTest(grouped=grouped):
                b, a, c = self.build()
                slots = a.alloc(96)
                states = b.req_to_token_pool.mamba_allocator.alloc(8)
                self.stamp(b, a, slots, states)
                records = self.insert_parts(c, slots, states)
                pending = a.alloc(2)
                self.assertIsNotNone(pending)
                a.free_group_begin()
                try:
                    if grouped == "ordinary":
                        a.free(pending)
                    elif grouped == "representatives":
                        a.free_segment(pending, start_pos=0)
                    else:
                        a.free_swa(pending)
                        a.free_full(pending)
                    with patch.object(
                        c, "evict_for_alloc", wraps=c.evict_for_alloc
                    ) as walks:
                        evict_from_tree_cache(c, 2)
                        got = a.alloc(2)
                    self.assertIsNotNone(got)
                    self.assertIsNotNone(a.free_group)
                    self.assertEqual(walks.call_count, 0)
                    for key, prior in records:
                        torch.testing.assert_close(
                            c.match_prefix(MatchPrefixParams(key=key)).device_indices,
                            prior,
                        )
                    self.check_payload(b, a, slots, states, states)
                    a.free(got)
                finally:
                    a.free_group_end()
                self.assertFalse(a.verify_byte_accounting())

    def test_session_cursor_joint_reclaim(self):
        b, a, c = self.build(sessions=True)
        slots = a.alloc(96)
        states = b.req_to_token_pool.mamba_allocator.alloc(8)
        self.stamp(b, a, slots, states)
        records = self.insert_parts(c, slots, states)
        evict_from_tree_cache(c, 4)
        self.assertIsNotNone(a.alloc(4))
        self.assertEqual(c.full_evictable_size(), 95)
        for key, prior in records[1:]:
            torch.testing.assert_close(
                c.match_prefix(MatchPrefixParams(key=key)).device_indices, prior
            )
        self.check_payload(b, a, slots[1:], states[1:], states)
        self.assertFalse(a.verify_byte_accounting())

    def test_ready_allocation_preserves_payload_with_closed_move_gates(self):
        for page_size, lazy, grouped in itertools.product(
            (1, 4), (False, True), (False, True)
        ):
            with self.subTest(page_size=page_size, lazy=lazy, grouped=grouped):
                b, a, c = self.build(
                    page_size=page_size, lazy=lazy, state_cache=page_size == 1
                )
                slots = a.alloc(96 * page_size)
                states = a.mamba_allocator.alloc(8)
                self.assertIsNotNone(slots)
                self.assertIsNotNone(states)
                self.stamp(b, a, slots, states)
                if page_size == 1:
                    records = self.insert_parts(c, slots, states)
                else:
                    # Paged cache owns KV; the request retains all Mamba states.
                    key = RadixKey(array("q", range(len(slots))))
                    c.insert(InsertParams(key=key, value=slots))
                    records = [(key, slots.clone())]
                for member in a._flush_targets():
                    member.disagg_move_gate = lambda: False
                if grouped:
                    a.free_group_begin()
                self.assertGreaterEqual(a.available_size(), page_size)
                before = snapshot(a)
                self.assertTrue(a.evict_to_free_tokens(c, page_size))
                self.assertEqual(snapshot(a), before)
                got = a.alloc(page_size)
                self.assertIsNotNone(got)
                self.assertEqual(len(got), page_size)
                for key, prior in records:
                    torch.testing.assert_close(
                        c.match_prefix(MatchPrefixParams(key=key)).device_indices,
                        prior,
                    )
                self.check_payload(b, a, slots, states, states)
                self.assertFalse(a.verify_byte_accounting())
                if grouped:
                    self.assertIsNotNone(a.free_group)
                    a.free_group_end()

    def test_ready_and_no_progress_skip_repeated_preparation(self):
        b, a, c = self.build()
        slots = a.alloc(96)
        states = b.req_to_token_pool.mamba_allocator.alloc(8)
        self.stamp(b, a, slots, states)
        self.insert_parts(c, slots, states)
        with (
            patch.object(a, "ensure_capacity", wraps=a.ensure_capacity) as prep,
            patch.object(c, "evict_for_alloc", wraps=c.evict_for_alloc) as walks,
        ):
            evict_from_tree_cache(c, 0)
            self.assertEqual(prep.call_count, 0)
            evict_from_tree_cache(c, 1)
            self.assertEqual(walks.call_count, 0)
        # Close every movement gate. The finite request quotas may still reclaim
        # cache bytes; actual allocation and each phase remain separately visible.
        for member in a._flush_targets():
            member.disagg_move_gate = lambda: False
        with (
            patch.object(a, "ensure_capacity", wraps=a.ensure_capacity) as prep,
            patch.object(c, "evict_for_alloc", wraps=c.evict_for_alloc) as walks,
        ):
            evict_from_tree_cache(c, 90)
            got = a.alloc(90)
        self.assertLessEqual(prep.call_count, 4)
        self.assertLessEqual(walks.call_count, 2)
        self.assertIsNotNone(got)
        self.assertFalse(a.verify_byte_accounting())

    def test_actual_decode_contract(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        b, a, c = self.build()
        slots = a.alloc(96)
        states = b.req_to_token_pool.mamba_allocator.alloc(8)
        self.stamp(b, a, slots, states)
        records = self.insert_parts(c, slots, states)
        batch = object.__new__(ScheduleBatch)
        batch.token_to_kv_pool_allocator = a
        batch.tree_cache = c
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        batch.reqs = [
            SimpleNamespace(kv=SimpleNamespace(kv_committed_len=8), beam_group=None)
            for _ in range(4)
        ]
        self.assertTrue(batch.check_decode_mem())
        self.assertIsNotNone(a.alloc(4))
        self.assertEqual(c.full_evictable_size(), 95)
        self.check_payload(b, a, slots[1:], states[1:], states)

    def test_physical_capacity_above_nominal_partition(self):
        for entry in ("direct", "helper", "decode"):
            with self.subTest(entry=entry):
                b, a, c = self.build(state_cache=False)
                demand = a.available_size()
                self.assertGreater(demand, a.conserve_full_available_size())
                self.assertTrue(a._token_reclaim_satisfied(demand))
                if entry == "direct":
                    self.assertTrue(a.ensure_capacity(demand, demand))
                elif entry == "helper":
                    self.assertTrue(evict_from_tree_cache(c, demand))
                else:
                    self.assertTrue(
                        a.check_decode_capacity(
                            num_tokens=demand,
                            tree_cache=c,
                            requests=[],
                            spec_algorithm=None,
                        )
                    )
                self.assertIsNotNone(a.alloc(demand))
                self.assertFalse(a.verify_byte_accounting())

    def test_float_recovery_preserves_prefix_when_full_band_is_short(self):
        for lazy in (False, True):
            for recovery in ("prepare", "eviction", "decode"):
                with self.subTest(lazy=lazy, recovery=recovery):
                    temporal = (0, 0, 0)
                    state_config = SimpleNamespace(
                        shape=SimpleNamespace(conv=[(3, 8)], temporal=temporal),
                        dtype=SimpleNamespace(
                            conv=torch.bfloat16, temporal=torch.float32
                        ),
                        layers=[0, 1],
                    )
                    bundle = init_unified_mamba_swa_pools(
                        device="cpu",
                        kv_cache_dtype=torch.float16,
                        head_num=1,
                        head_dim=8,
                        v_head_dim=8,
                        swa_head_num=1,
                        swa_head_dim=8,
                        swa_v_head_dim=8,
                        page_size=1,
                        start_layer=0,
                        end_layer=6,
                        swa_attention_layer_ids=[4, 5],
                        full_attention_layer_ids=[0, 1, 2, 3],
                        full_max_total_num_tokens=32,
                        swa_max_total_num_tokens=24,
                        enable_memory_saver=False,
                        need_sort=False,
                        lazy_compaction=lazy,
                        mamba_layer_ids=[0, 1],
                        mamba2_cache_params=state_config,
                        max_mamba_cache_size=8,
                        model_context_len=256,
                        extra_max_context_len=1,
                        max_num_reqs=4,
                        enable_mamba_extra_buffer=False,
                        enable_mamba_extra_buffer_lazy=False,
                        disable_overlap_schedule=True,
                        sliding_window_size=8,
                    )
                    allocator = bundle.token_to_kv_pool_allocator
                    cache = UnifiedRadixCache(
                        CacheInitParams(
                            disable=False,
                            req_to_token_pool=bundle.req_to_token_pool,
                            token_to_kv_pool_allocator=allocator,
                            page_size=1,
                            sliding_window_size=128,
                            tree_components=(ComponentType.FULL, ComponentType.SWA),
                        )
                    )
                    slots = allocator.alloc(4)
                    self.assertIsNotNone(slots)
                    kv_pool = bundle.token_to_kv_pool
                    full = allocator.full_attn_allocator
                    swa = allocator.swa_attn_allocator
                    markers = slots.to(torch.float16).view(-1, 1, 1)
                    for layer in range(6):
                        member = full if layer < 4 else swa
                        kernel_ids = member.translate_kv_loc(slots)
                        kv_pool.get_key_buffer(layer)[kernel_ids] = markers + 10 * layer
                        kv_pool.get_value_buffer(layer)[kernel_ids] = (
                            markers + 100 + 10 * layer
                        )
                    extra = full.alloc(full.available_size() - 2)
                    self.assertIsNotNone(extra)
                    key = RadixKey(array("q", [1, 2, 3, 4]))
                    cache.insert(InsertParams(key=key, value=slots))
                    self.assertEqual(allocator.full_available_size(), 2)
                    self.assertGreaterEqual(allocator.conserve_full_available_size(), 6)
                    self.assertGreaterEqual(allocator.swa_available_size(), 6)
                    self.assertLess(allocator.available_size(), 6)
                    self.assertEqual(allocator._token_allocation_byte_shortfall(6), 0)
                    if recovery == "prepare":
                        self.assertTrue(allocator.ensure_capacity(6, 6))
                    elif recovery == "decode":
                        self.assertTrue(
                            allocator.check_decode_capacity(
                                num_tokens=6, tree_cache=cache
                            )
                        )
                    else:
                        evict_from_tree_cache(cache, 6)
                    self.assertTrue(allocator.available_size() >= 6)
                    self.assertEqual(cache.full_evictable_size(), 4)
                    matched = cache.match_prefix(MatchPrefixParams(key=key))
                    torch.testing.assert_close(matched.device_indices, slots)
                    self.assertIsNotNone(allocator.alloc(6))
                    for layer in range(6):
                        member = full if layer < 4 else swa
                        kernel_ids = member.translate_kv_loc(slots)
                        self.assertTrue(
                            torch.all(
                                kv_pool.get_key_buffer(layer)[kernel_ids]
                                == markers + 10 * layer
                            )
                        )
                        self.assertTrue(
                            torch.all(
                                kv_pool.get_value_buffer(layer)[kernel_ids]
                                == markers + 100 + 10 * layer
                            )
                        )
                    self.assertFalse(allocator.verify_byte_accounting())

    def test_deep_reclaim_preserves_first_sufficient_prefix_set(self):
        for lazy, demand, first in (
            (False, 6, 3),
            (True, 6, 3),
            (False, 90, 87),
            (True, 90, 87),
        ):
            with self.subTest(lazy=lazy, demand=demand):
                b, a, c = self.build(lazy=lazy, state_cache=False)
                states = b.req_to_token_pool.mamba_allocator.alloc(8)
                slots = a.alloc(96)
                self.stamp(b, a, slots, states)
                keys = [RadixKey(array("q", [i])) for i in range(96)]
                for i, key in enumerate(keys):
                    c.insert(InsertParams(key=key, value=slots[i : i + 1]))
                with patch.object(
                    a, "ensure_capacity", wraps=a.ensure_capacity
                ) as prep:
                    evict_from_tree_cache(c, demand)
                    self.assertIsNotNone(a.alloc(demand))
                self.assertLessEqual(prep.call_count, 3)
                for i, key in enumerate(keys):
                    actual = c.match_prefix(MatchPrefixParams(key=key)).device_indices
                    if i < first:
                        self.assertEqual(len(actual), 0)
                    else:
                        torch.testing.assert_close(actual, slots[i : i + 1])
                self.check_payload(b, a, slots[first:], states, states)
                self.assertFalse(a.verify_byte_accounting())

    def test_rank_consensus_uses_stable_tri_demand(self):
        import os
        import subprocess
        import sys

        script = r"""
import queue,runpy,sys,threading
from pathlib import Path
from unittest.mock import patch
from sglang.srt.utils import rank_consensus_checker as checker
sys.path.insert(0,str(Path(sys.argv[1]).parent))
module=runpy.run_path(sys.argv[1],run_name='rank_fixture')
fixture=module['TestTriJointReclaim']()
traces=[]
for _ in range(2):
    fixture.setUp()
    try:
        b,a,c=fixture.build()
        slots=a.alloc(96);states=b.req_to_token_pool.mamba_allocator.alloc(8)
        fixture.stamp(b,a,slots,states);fixture.insert_parts(c,slots,states)
        q=queue.Queue()
        with patch.object(checker,'_q',q),patch.object(checker,'_scheduler_thread',threading.current_thread()):
            assert module['evict_from_tree_cache'](c,4)
            assert a.alloc(4) is not None
        fixture.check_payload(b,a,slots[1:],states[1:],states)
        traces.append(list(q.queue))
    finally:fixture.tearDown()
assert traces[0]==traces[1],traces
assert any('num_tokens=4'in event for event in traces[0]),traces
assert any('allocation_reclaim_satisfied is not None=True'in event for event in traces[0]),traces
assert not any('<function'in event for event in traces[0]),traces
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, __file__],
            env=dict(os.environ, SGLANG_ENABLE_RANK_CONSENSUS_CHECKER="1"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_state_quota_accounts_for_token_phase_cascades(self):
        for occupancy, demand in ((96, 10), (97, 1)):
            with self.subTest(occupancy=occupancy, demand=demand):
                group = 1
                b, a, c = self.build(temporal=(1, 4, 8))
                states = b.req_to_token_pool.mamba_allocator.alloc(8)
                slots = a.alloc(occupancy)
                self.stamp(b, a, slots, states)
                records = []
                receipts = []
                for i, (lo, hi, st) in enumerate(
                    (
                        (0, 1, states[:group]),
                        (1, 2, states[group : group + 1]),
                        (2, occupancy, states[group + 1 : group + 2]),
                    )
                ):
                    key = RadixKey(array("q", range(i * 1000, i * 1000 + hi - lo)))
                    c.insert(InsertParams(key=key, value=slots[lo:hi], mamba_value=st))
                    node = c.match_prefix(MatchPrefixParams(key=key)).last_device_node
                    records.append((node, slots[lo:hi]))
                    if i:
                        receipts.append(
                            (
                                node,
                                c.inc_lock_ref(
                                    node, skip_lock_components=(ComponentType.MAMBA,)
                                ),
                            )
                        )
                phases = []
                original = c.evict_for_alloc

                def walk(params, **kw):
                    result = original(params, **kw)
                    phases.append((params, result))
                    return result

                try:
                    with patch.object(c, "evict_for_alloc", side_effect=walk):
                        evict_from_tree_cache(c, demand)
                        self.assertIsNotNone(a.alloc(demand))
                    self.assertEqual(phases[0][1].mamba_num_evicted, group)
                    quota = -(
                        -(
                            demand
                            * (
                                a.full_attn_allocator.entry_bytes
                                + a.swa_attn_allocator.entry_bytes
                            )
                        )
                        // a.mamba_allocator.entry_bytes
                    )
                    if group < quota:
                        self.assertEqual(len(phases), 2)
                        self.assertEqual(phases[1][0].mamba_num, quota - group)
                        self.assertGreater(phases[1][1].mamba_num_evicted, 0)
                    else:
                        self.assertEqual(len(phases), 1)
                    for node, prior in records[1:]:
                        torch.testing.assert_close(
                            c.tree_core.get_component_device_value(
                                node, ComponentType.FULL
                            ),
                            prior,
                        )
                    self.check_payload(b, a, slots[1:], states[group + 2 :], states)
                finally:
                    for node, receipt in receipts:
                        c.dec_lock_ref(node, receipt.to_dec_params())
                self.assertFalse(a.verify_byte_accounting())

    def test_temporal_state_first_stops_after_one_victim(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                b, a, c = self.build(lazy=lazy, temporal=(1, 4, 8))
                states = b.req_to_token_pool.mamba_allocator.alloc(8)
                slots = a.alloc(96)
                self.stamp(b, a, slots, states)
                records = self.insert_parts(c, slots, states)
                query = a._token_reclaim_satisfied
                checkpoints, results = [], []

                def checked_query(n):
                    before = snapshot(a)
                    ready = query(n)
                    self.assertEqual(snapshot(a), before)
                    if ready:
                        checkpoints.append(c.full_evictable_size())
                    return ready

                evict = c.evict_for_alloc

                def checked_evict(*args, **kwargs):
                    result = evict(*args, **kwargs)
                    results.append(
                        (
                            result.num_tokens_evicted,
                            result.swa_num_tokens_evicted,
                            result.mamba_num_evicted,
                        )
                    )
                    return result

                sa = a.swa_attn_allocator
                move = sa.make_room
                targets = []

                def checked_move(*, side, min_bytes):
                    self.assertEqual(side, "high")
                    self.assertEqual(min_bytes, 256)
                    self.assertEqual(
                        min_bytes, a._token_float_high_target(4, flush_ends=False)
                    )
                    targets.append(min_bytes)
                    return move(side=side, min_bytes=min_bytes)

                with (
                    patch.object(
                        a, "_token_reclaim_satisfied", side_effect=checked_query
                    ),
                    patch.object(c, "evict_for_alloc", side_effect=checked_evict),
                    patch.object(sa, "make_room", side_effect=checked_move),
                    patch(
                        "sglang.srt.mem_cache.allocator.unified_hybrid_swa._relieve_for_alloc",
                        side_effect=AssertionError(
                            "certified allocation used fallback"
                        ),
                    ),
                    patch.object(
                        a, "ensure_capacity", wraps=a.ensure_capacity
                    ) as prepare,
                ):
                    self.assertTrue(evict_from_tree_cache(c, 4))
                    self.assertIsNotNone(a.alloc(4))
                self.assertEqual(results, [(1, 1, 1)])
                self.assertEqual(set(checkpoints), {95})
                self.assertEqual(targets, [256])
                self.assertLessEqual(prepare.call_count, 3)
                self.assertEqual(c.full_evictable_size(), 95)
                for key, prior in records[1:]:
                    torch.testing.assert_close(
                        c.match_prefix(MatchPrefixParams(key=key)).device_indices, prior
                    )
                self.check_payload(b, a, slots[1:], states[1:], states)
                self.assertFalse(a.verify_byte_accounting())

    def test_float_plan_rechecks_gate_changes_during_preparation(self):
        for owner, gate_attr in itertools.product(
            ("full", "mamba", "swa"), ("disagg_move_gate", "host_transfer_move_gate")
        ):
            with self.subTest(owner=owner, gate=gate_attr):
                b, a, c = self.build(lazy=True, temporal=(1, 4, 8))
                states = b.req_to_token_pool.mamba_allocator.alloc(8)
                slots = a.alloc(96)
                self.stamp(b, a, slots, states)
                self.insert_parts(c, slots, states)
                c.evict(EvictParams(num_tokens=1))
                self.assertEqual(c.full_evictable_size(), 95)
                self.assertEqual(a._token_float_high_target(4), 256)
                members = {
                    "full": a.full_attn_allocator,
                    "mamba": a.mamba_allocator,
                    "swa": a.swa_attn_allocator,
                }
                member = members[owner]
                full_flush = a.full_attn_allocator.flush_for_allocation

                def close_during_prepare():
                    setattr(member, gate_attr, lambda: False)
                    return full_flush()

                sa = a.swa_attn_allocator
                with (
                    patch.object(
                        a.full_attn_allocator,
                        "flush_for_allocation",
                        side_effect=close_during_prepare,
                    ),
                    patch.object(sa, "make_room", wraps=sa.make_room) as movement,
                    patch(
                        "sglang.srt.mem_cache.allocator.unified_hybrid_swa._relieve_for_alloc",
                        side_effect=AssertionError("stale plan used fallback"),
                    ),
                ):
                    result = a.ensure_capacity(4, 4)
                self.assertEqual(result, owner == "full")
                if owner == "full":
                    self.assertEqual(
                        movement.call_args.kwargs, {"side": "high", "min_bytes": 224}
                    )
                else:
                    self.assertEqual(movement.call_count, 0)
                    self.assertIsNone(a._token_float_high_target(4, flush_ends=False))
                self.check_payload(b, a, slots[1:], states[1:], states)
                setattr(member, gate_attr, None)
                self.assertTrue(a.ensure_capacity(4, 4))
                self.assertIsNotNone(a.alloc(4))
                self.check_payload(b, a, slots[1:], states[1:], states)
                self.assertFalse(a.verify_byte_accounting())

    def test_float_whole_move_preserves_adverse_hole_layouts(self):
        for lazy, holes in itertools.product((False, True), ((4, 7), (5, 6))):
            with self.subTest(lazy=lazy, holes=holes):
                b, a, c = self.build(lazy=lazy, ratio=(4, 2), state_cache=False)
                slots = a.alloc(8)
                states = torch.empty(0, dtype=torch.int64)
                self.stamp(b, a, slots, states, full_layers=4)
                a.free_swa(slots[list(holes)])
                fa, sa = a.full_attn_allocator, a.swa_attn_allocator
                extra = fa.alloc(fa.available_size() - 2)
                for layer in range(4):
                    ids = fa.translate_kv_loc(extra)
                    b.token_to_kv_pool.get_key_buffer(layer)[ids] = 701 + layer
                    b.token_to_kv_pool.get_value_buffer(layer)[ids] = 801 + layer
                target = a._token_float_high_target(12)
                self.assertIsNotNone(target)
                self.assertFalse(a._token_end_reclaim_satisfied(12))
                gap = sa._gap_pages()[1]
                retreat = target // sa.entry_bytes_per_page - gap
                self.assertGreaterEqual(retreat, sa._live_pages())
                with (
                    patch.object(
                        sa, "_relocate_to_positions", wraps=sa._relocate_to_positions
                    ) as whole,
                    patch(
                        "sglang.srt.mem_cache.allocator.unified_hybrid_swa._relieve_for_alloc",
                        side_effect=AssertionError(
                            "positive certificate used fallback"
                        ),
                    ),
                ):
                    self.assertTrue(a.ensure_capacity(12, 12))
                    self.assertIsNotNone(a.alloc(12))
                self.assertEqual(whole.call_count, 1)
                self.check_payload(
                    b,
                    a,
                    slots[[i for i in range(8) if i not in holes]],
                    states,
                    states,
                    full_layers=4,
                )
                for layer in range(4):
                    ids = fa.translate_kv_loc(extra)
                    self.assertTrue(
                        torch.all(
                            b.token_to_kv_pool.get_key_buffer(layer)[ids] == 701 + layer
                        )
                    )
                    self.assertTrue(
                        torch.all(
                            b.token_to_kv_pool.get_value_buffer(layer)[ids]
                            == 801 + layer
                        )
                    )
                self.assertFalse(a.verify_byte_accounting())

    def test_asymmetric_preparation_preserves_live_payload(self):
        for full, swa in ((4, 0), (4, 2), (2, 4), (0, 4)):
            with self.subTest(full=full, swa=swa):
                b, a, c = self.build(lazy=True, temporal=(1, 4, 8))
                states = b.req_to_token_pool.mamba_allocator.alloc(8)
                slots = a.alloc(96)
                self.stamp(b, a, slots, states)
                self.insert_parts(c, slots, states)
                c.evict(EvictParams(num_tokens=1))
                # Existing FULL-only pages supply any SWA demand beyond the
                # newly allocated FULL pages, as in a partial host reload.
                extra = max(0, swa - full)
                existing = slots[-extra:] if extra else slots[:0]
                if extra:
                    a.free_swa(existing)
                prepared = a.ensure_capacity(full, swa)
                self.assertTrue(prepared)
                fresh = a.full_attn_allocator.alloc(full)
                self.assertIsNotNone(fresh)
                bind = torch.cat((fresh[: min(full, swa)], existing))
                a.swa_attn_allocator.alloc_with_virtual(bind)
                kept = slots[1:-extra] if extra else slots[1:]
                self.check_payload(b, a, kept, states[1:], states)
                self.assertEqual(c.full_evictable_size(), 95)
                self.assertFalse(a.verify_byte_accounting())


class TestTriGeometry(CustomTestCase):
    def check_saved(self, a, saved):
        for member, buf, ids, values in saved:
            self.assertTrue(
                torch.all(
                    member.virtual_to_physical[ids // member.page_size]
                    >= member.min_page_index
                )
            )
            torch.testing.assert_close(buf[member.translate_kv_loc(ids)], values)
        self.assertFalse(a.verify_byte_accounting())

    def test_float_fresh_destination_reservation_boundary(self):
        for ps, extra_pages in ((1, 18), (4, 9), (16, 6)):
            for one_short in (False, True):
                with self.subTest(ps=ps, one_short=one_short):
                    a, kv, mk = build_geometry(ps, (1, 1), True, "none", False)
                    ma, fa, sa = (
                        a.mamba_allocator,
                        a.full_attn_allocator,
                        a.swa_attn_allocator,
                    )
                    live = torch.where(ma.virtual_to_physical >= ma.min_page_index)[0]
                    live = live[live >= ma.min_page_index]
                    ma.free(live[:-1])
                    fa.flush_for_allocation()
                    ma.flush_for_allocation()
                    extra = fa.alloc(extra_pages * ps)
                    self.assertIsNotNone(extra)
                    demand = 2 * ps
                    target = a._token_float_high_target(demand)
                    self.assertIsNotNone(target)
                    lo, hi = sa._region_bounds_pages()
                    retreat = target // sa.entry_bytes_per_page - (hi - sa.high_wm_page)
                    self.assertEqual(sa.low_wm_page - lo, retreat)
                    self.assertLess(retreat, sa._live_pages())
                    self.assertFalse(a._token_end_reclaim_satisfied(demand))
                    if one_short:
                        # Bind an existing FULL-only ID into the roomier LOW gap.
                        low_before = sa.low_wm_page
                        sa.alloc_with_virtual(extra[:1] // ps)
                        self.assertEqual(sa.low_wm_page, low_before - 1)
                        before = snapshot(a)
                        self.assertIsNone(a._token_float_high_target(demand))
                        self.assertEqual(snapshot(a), before)
                        self.assertFalse(a.verify_byte_accounting())
                        continue
                    saved = payload(a, kv, mk)
                    with (
                        patch.object(
                            sa,
                            "_relocate_to_positions",
                            wraps=sa._relocate_to_positions,
                        ) as whole,
                        patch(
                            "sglang.srt.mem_cache.allocator.unified_hybrid_swa._relieve_for_alloc",
                            side_effect=AssertionError(
                                "positive certificate used fallback"
                            ),
                        ),
                    ):
                        self.assertTrue(a.ensure_capacity(demand, demand))
                        self.assertIsNotNone(a.alloc(demand))
                    self.assertEqual(whole.call_count, 0)
                    self.check_saved(a, saved)

    def test_float_certificates_match_executor_across_page_grids(self):
        recognized = set()
        for ps, ratio, lazy, pattern, gated, npage in itertools.product(
            (1, 4, 16),
            ((1, 1), (1, 4), (4, 1)),
            (False, True),
            ("none", "paired", "swa_edges", "swa_internal", "empty", "mixed"),
            (False, True),
            (5, 7, 10),
        ):
            a, kv, mk = build_geometry(ps, ratio, lazy, pattern, gated)
            prior = snapshot(a)
            target = a._token_float_high_target(npage * ps)
            self.assertEqual(snapshot(a), prior)
            if gated or pattern == "empty":
                self.assertIsNone(target)
            if target is None or a._token_end_reclaim_satisfied(npage * ps):
                continue
            with self.subTest(
                ps=ps, ratio=ratio, lazy=lazy, pattern=pattern, npage=npage
            ):
                recognized.add(ps)
                saved = payload(a, kv, mk)
                with patch(
                    "sglang.srt.mem_cache.allocator.unified_hybrid_swa._relieve_for_alloc",
                    side_effect=AssertionError("FLOAT certificate used fallback"),
                ):
                    self.assertTrue(a.ensure_capacity(npage * ps, npage * ps))
                    self.assertIsNotNone(a.alloc(npage * ps))
                self.check_saved(a, saved)
        self.assertEqual(recognized, {1, 4, 16})

    def test_nontrivial_end_certificates_preserve_float(self):
        recognized = set()
        for ps, ratio, cut, npage in itertools.product(
            (1, 4, 16), ((1, 1), (1, 4), (4, 1)), (2, 3, 4), range(1, 12)
        ):
            a, kv, mk = build_geometry(ps, ratio, True, "none", False)
            ma = a.mamba_allocator
            extra = ma.alloc(ma.available_size())
            self.assertIsNotNone(extra)
            vp = torch.where(ma.virtual_to_physical >= ma.min_page_index)[0]
            vp = vp[vp >= ma.min_page_index]
            mk.buf[ma.virtual_to_physical[vp]] = vp + 1000
            ma.free(vp[1:-1:cut])
            if npage * ps <= a.available_size() or not a._token_end_reclaim_satisfied(
                npage * ps
            ):
                continue
            with self.subTest(page_size=ps, ratio=ratio, cut=cut, demand=npage):
                recognized.add(ps)
                saved = payload(a, kv, mk)
                prior = snapshot(a)
                for _ in range(20):
                    self.assertTrue(a._token_end_reclaim_satisfied(npage * ps))
                self.assertEqual(snapshot(a), prior)
                sa = a.swa_attn_allocator
                with (
                    patch.object(sa, "make_room", wraps=sa.make_room) as moves,
                    patch.object(
                        sa, "flush_for_allocation", wraps=sa.flush_for_allocation
                    ) as flush,
                ):
                    self.assertTrue(a.ensure_capacity(npage * ps, npage * ps))
                    self.assertIsNotNone(a.alloc(npage * ps))
                self.assertEqual(moves.call_count, 0)
                self.assertEqual(flush.call_count, 0)
                self.check_saved(a, saved)
        self.assertEqual(recognized, {1, 4, 16})

    def test_gate_change_invalidates_end_certificate(self):
        for pd_open, host_open in ((True, False), (False, True), (False, False)):
            with self.subTest(pd_open=pd_open, host_open=host_open):
                a, kv, mk = build_geometry(1, (1, 4), True, "swa_edges", False)
                self.assertLess(a.available_size(), 5)
                self.assertTrue(a._token_reclaim_satisfied(5))
                saved = payload(a, kv, mk)
                for member in a._flush_targets():
                    member.disagg_move_gate = lambda: pd_open
                    member.host_transfer_move_gate = lambda: host_open
                before = snapshot(a)
                self.assertFalse(a._token_reclaim_satisfied(5))
                with patch.object(
                    kv.swa_kv_pool, "move_kv_cache", wraps=kv.swa_kv_pool.move_kv_cache
                ) as copies:
                    self.assertFalse(a.ensure_capacity(5, 5))
                self.assertEqual(copies.call_count, 0)
                # Zero-copy FLOAT boundary absorption is legal even with movement gated.
                self.check_saved(a, saved)
                for member in a._flush_targets():
                    member.disagg_move_gate = None
                    member.host_transfer_move_gate = None
                self.assertTrue(a.ensure_capacity(5, 5))
                self.assertIsNotNone(a.alloc(5))
                self.check_saved(a, saved)

    def test_pending_reuse_is_not_certificate_credit(self):
        class Event:
            fired = False

            def query(self):
                return self.fired

        class Stream:
            waits = 0

            def wait_event(self, event):
                self.waits += 1

        for ps, urgent in itertools.product((1, 4, 16), (False, True)):
            with self.subTest(page_size=ps, urgent=urgent):
                a, kv, mk = build_geometry(ps, (1, 4), True, "none", False)
                fa = a.full_attn_allocator
                event = Event()
                stream = Stream()
                fa.set_latest_forward_done_event(event)
                fa.set_inflight_forward(event, None)
                vp = torch.where(fa.virtual_to_physical >= fa.min_page_index)[0]
                vp = vp[vp >= fa.min_page_index][10:11]
                tokens = (vp[:, None] * ps + torch.arange(ps)[None, :]).flatten()
                a.free(tokens)
                self.assertGreater(fa._flush(urgent=False), 0)
                self.assertGreater(len(fa._pending_reuse_pages_cpu), 0)
                saved = payload(a, kv, mk)
                for member in a._flush_targets():
                    member.disagg_move_gate = lambda: False
                before = snapshot(a)
                predictions = [a._token_reclaim_satisfied(n * ps) for n in range(1, 10)]
                for _ in range(20):
                    self.assertEqual(
                        predictions,
                        [a._token_reclaim_satisfied(n * ps) for n in range(1, 10)],
                    )
                self.assertEqual(snapshot(a), before)
                for member in a._flush_targets():
                    member.disagg_move_gate = None
                if not urgent:
                    event.fired = True
                    fa.set_latest_forward_done_event(None)
                with patch("torch.cuda.current_stream", return_value=stream):
                    fa.flush_for_allocation()
                    self.assertFalse(fa._pending_reuse)
                    n = max(
                        n for n in range(1, 10) if a._token_reclaim_satisfied(n * ps)
                    )
                    self.assertTrue(a.ensure_capacity(n * ps, n * ps))
                    self.assertIsNotNone(a.alloc(n * ps))
                self.assertEqual(stream.waits > 0, urgent)
                self.check_saved(a, saved)


if __name__ == "__main__":
    unittest.main()

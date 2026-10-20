"""HiCache load-back, admission and branching tests for UnifiedRadixCache."""

import unittest
from array import array
from dataclasses import replace
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import (
    Req,
    ReqKvInfo,
)
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    RebuildFullToSWAMapping,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.runtime_context import (
    mamba_cache_chunk_size,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
    define_config_test_classes,
    requires,
    requires_full_mamba_ps1,
    requires_full_swa_ps1_window4,
    requires_hicache,
    requires_swa,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_device_value,
    node_host_value,
    node_key_length,
    node_parent,
)
from sglang.test.kits.unified_radix_cache.load_back import (
    UnifiedRadixCacheLoadBackFixtures,
    UnifiedTreeCoreLoadBackPendingFixtures,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=11, suite="stage-b-test-1-gpu-small-amd")


class TestUnifiedTreeCoreLoadBackPending(
    UnifiedTreeCoreLoadBackPendingFixtures, CustomTestCase
):
    def test_write_through_different_anchors_track_duplicate_without_pending(self):
        core, shared, anchor_a, anchor_b = self._build_core(is_write_back=False)
        full = shared.component_data[ComponentType.FULL]
        full.value = torch.tensor([1], dtype=torch.int64)
        full.host_value = torch.tensor([2], dtype=torch.int64)

        self._commit_load_back(core, anchor_a, shared)
        self._commit_load_back(core, anchor_b, shared)
        UnifiedTreeCore.finish_load_back(core, anchor_a.id)

        self.assertIsNone(shared.load_back_pending_id)
        self.assertIn(shared.id, core.full_host_duplicates)
        core._update_duplicate_tracking.assert_has_calls(
            [mock.call(anchor_a), mock.call(shared)]
        )

    def test_write_back_pending_blocks_reclaim_until_ack(self):
        core, shared, anchor_a, anchor_b = self._build_core(is_write_back=True)
        full = shared.component_data[ComponentType.FULL]
        full.value = torch.tensor([1], dtype=torch.int64)
        full.host_value = torch.tensor([2], dtype=torch.int64)

        self._commit_load_back(core, anchor_a, shared)

        self.assertEqual(shared.load_back_pending_id, anchor_a.id)
        self.assertFalse(UnifiedTreeCore._can_reclaim_full_host_duplicate(core, shared))
        with self.assertRaisesRegex(AssertionError, "new anchor"):
            self._commit_load_back(core, anchor_b, shared)

        UnifiedTreeCore.finish_load_back(core, anchor_a.id)

        self.assertIsNone(shared.load_back_pending_id)
        self.assertTrue(UnifiedTreeCore._can_reclaim_full_host_duplicate(core, shared))
        core._update_duplicate_tracking.assert_called_once_with(shared)

    def test_auxiliary_load_does_not_reuse_full_pending_pin(self):
        core, shared, anchor_a, anchor_b = self._build_core(is_write_back=True)
        core.components_by_type[ComponentType.SWA] = mock.Mock()

        self._commit_load_back(core, anchor_a, shared)
        self.assertEqual(shared.load_back_pending_id, anchor_a.id)

        kv_transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([1], dtype=torch.int64),
            nodes_to_load=[anchor_b.id],
        )
        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([2], dtype=torch.int64),
            nodes_to_load=[shared.id],
        )
        UnifiedTreeCore.commit_load_back(
            core,
            anchor_b.id,
            torch.tensor([3], dtype=torch.int64),
            kv_transfer,
            {ComponentType.SWA: [swa_transfer]},
        )

        self.assertEqual(shared.load_back_pending_id, anchor_a.id)
        self.assertEqual(anchor_b.load_back_pending_id, anchor_b.id)


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestUnifiedTreeCoreLoadBackOwnershipBackends(CustomTestCase):
    """Run Full load-back ownership semantics through either TreeCore backend."""

    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
    )

    def test_auxiliary_load_does_not_reuse_full_pending_pin(self):
        cache, _, _ = build_fixture(self.cfg)
        core = cache.tree_core
        core.is_write_back = True

        root = cache.root_node_handle()

        def insert_host(tokens, indices, hashes):
            return core.insert_host(
                root,
                RadixKey(array("q", tokens)),
                torch.tensor(indices, dtype=torch.int64),
                hashes,
            ).inserted_host_node

        def full_transfer(node_id, host_index):
            return PoolTransfer(
                name=PoolName.KV,
                host_indices=torch.tensor([host_index], dtype=torch.int64),
                nodes_to_load=[node_id],
            )

        shared = insert_host([1], [100], ["h0"])
        anchor_b = insert_host([1, 2], [100, 101], ["h0", "h1"])
        self.assertIsNotNone(shared)
        self.assertIsNotNone(anchor_b)

        core.set_component_host_value_raw(
            shared, ComponentType.SWA, torch.tensor([200], dtype=torch.int64)
        )
        core.commit_load_back(
            shared,
            torch.tensor([10], dtype=torch.int64),
            full_transfer(shared, 100),
            {},
        )

        actions = core.commit_load_back(
            anchor_b,
            torch.tensor([11], dtype=torch.int64),
            full_transfer(anchor_b, 101),
            {
                ComponentType.SWA: [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=torch.tensor([200], dtype=torch.int64),
                        device_indices=torch.tensor([20], dtype=torch.int64),
                        nodes_to_load=[shared],
                    )
                ]
            },
        )

        self.assertEqual(
            node_device_value(cache, shared, ComponentType.SWA).tolist(), [20]
        )
        self.assertTrue(
            any(isinstance(action, RebuildFullToSWAMapping) for action in actions)
        )
        core.finish_load_back(anchor_b)
        core.finish_load_back(shared)


class UnifiedRadixCacheLoadBackSuite(UnifiedRadixCacheLoadBackFixtures):
    @requires_hicache
    def test_hicache_load_back_restores_data(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        base = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, base)

        m = match_prefix(cache, base)
        node = m.last_device_node
        original_device_indices = cache.path_device_indices(m.last_device_node).clone()
        self._fill_full_kv(allocator, original_device_indices, marker=3)
        expected_k, expected_v = self._snapshot_full_kv(
            allocator, original_device_indices
        )
        original_mamba_indices = None
        expected_temporal = None
        expected_conv = None
        if self.cfg.has_mamba:
            original_mamba_indices = node_device_value(
                cache, node, ComponentType.MAMBA
            ).clone()
            self._fill_mamba_state(req_to_token_pool, original_mamba_indices, marker=11)
            expected_temporal, expected_conv = self._snapshot_mamba_state(
                req_to_token_pool, original_mamba_indices
            )

        self._backup_node(cache, node)
        cache.evict(EvictParams(num_tokens=len(base)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(node))
        self._fill_full_kv(allocator, original_device_indices, marker=9)
        if original_mamba_indices is not None:
            self._fill_mamba_state(req_to_token_pool, original_mamba_indices, marker=21)

        self._load_back_node(cache, node)
        self.assertFalse(cache.tree_core.is_full_device_evicted(node))
        self.assertIsNotNone(node_device_value(cache, node, ComponentType.FULL))
        # Gather the whole reloaded prefix via match (a leaf may be split).
        loaded_indices = cache.path_device_indices(
            match_prefix(cache, base).last_device_node
        )
        loaded_k, loaded_v = self._snapshot_full_kv(allocator, loaded_indices)
        self.assertTrue(torch.equal(loaded_k, expected_k))
        self.assertTrue(torch.equal(loaded_v, expected_v))
        if self.cfg.has_mamba:
            loaded_mamba_indices = node_device_value(cache, node, ComponentType.MAMBA)
            loaded_temporal, loaded_conv = self._snapshot_mamba_state(
                req_to_token_pool, loaded_mamba_indices
            )
            self.assertTrue(torch.equal(loaded_temporal, expected_temporal))
            self.assertEqual(len(loaded_conv), len(expected_conv))
            for actual_conv, expected_conv_buf in zip(loaded_conv, expected_conv):
                self.assertTrue(torch.equal(actual_conv, expected_conv_buf))
        cache.sanity_check()

    @requires(
        lambda cfg: (
            cfg.has_swa
            and not cfg.has_mamba
            and cfg.page_size in (1, 4)
            and cfg.sliding_window_size == 4
        ),
        "requires Full+SWA with window_size=4",
    )
    def test_swa_branching_seqlen_uses_device_full_hit(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        window = self.cfg.sliding_window_size
        window_pages = (window + self.cfg.page_size - 1) // self.cfg.page_size
        prefix = self._make_seq(1, window_pages)
        tokens = prefix + self._make_seq(1000, 2)
        self._insert(cache, allocator, req_to_token_pool, prefix)
        self._insert(cache, allocator, req_to_token_pool, tokens)

        leaf = match_prefix(cache, tokens).last_device_node
        evicted = cache.tree_core.evict_component(
            leaf, ComponentType.SWA, EvictLayer.DEVICE
        )
        cache._free_values(evicted.device_frees, evicted.host_frees)

        result = match_prefix(cache, tokens)

        self.assertEqual(result.full_kv_hit_length, len(tokens))
        self.assertEqual(result.swa_branching_seqlen, len(tokens))
        self.assertEqual(result.swa_branching_seqlen % self.cfg.page_size, 0)

        # Simulate forward producing fresh SWA KV at the branching point.
        self._insert(
            cache,
            allocator,
            req_to_token_pool,
            tokens[: result.swa_branching_seqlen],
        )

        rematch = match_prefix(cache, tokens)
        self.assertEqual(rematch.device_prefix_len, result.swa_branching_seqlen)
        self.assertIsNone(rematch.swa_branching_seqlen)

    @requires_full_swa_ps1_window4
    def test_swa_branching_seqlen_uses_host_full_hit(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        window = self.cfg.sliding_window_size
        prefix = self._make_seq(1, window)
        tokens = prefix + self._make_seq(1000, window + 1)
        self._insert(cache, allocator, req_to_token_pool, prefix)
        self._insert(cache, allocator, req_to_token_pool, tokens)

        leaf = match_prefix(cache, tokens).last_device_node
        leaf_len = node_key_length(cache, leaf)
        parent = node_parent(cache, leaf)
        self._backup_node(cache, leaf)
        lock_result = cache.inc_lock_ref(parent)
        try:
            cache.evict(EvictParams(num_tokens=leaf_len))
        finally:
            cache.dec_lock_ref(parent, lock_result.to_dec_params())
        evicted = cache.tree_core.evict_component(
            leaf, ComponentType.SWA, EvictLayer.HOST
        )
        cache._free_values(evicted.device_frees, evicted.host_frees)
        full_host_pool = cache.cache_controller.mem_pool_host
        swa_host_pool = cache.components[ComponentType.SWA]._swa_kv_pool_host
        full_available_before = full_host_pool.available_size()
        swa_available_before = swa_host_pool.available_size()

        result = match_prefix(cache, tokens)

        self.assertEqual(result.full_kv_hit_length, len(tokens))
        self.assertEqual(result.swa_branching_seqlen, len(tokens))

        self._insert(
            cache,
            allocator,
            req_to_token_pool,
            tokens[: result.swa_branching_seqlen],
        )
        cache.writing_check(write_back=True)
        # Full was already backed up, so only the SWA window is allocated.
        self.assertEqual(full_host_pool.available_size(), full_available_before)
        self.assertEqual(
            swa_host_pool.available_size(),
            swa_available_before - leaf_len,
        )

        rematch = match_prefix(cache, tokens)
        self.assertEqual(rematch.device_prefix_len, result.swa_branching_seqlen)
        self.assertIsNone(rematch.swa_branching_seqlen)

    @requires_full_swa_ps1_window4
    def test_swa_branching_seqlen_caps_insert_after_forward(self):
        cache, _, _ = build_fixture(self.cfg)
        swa = cache.components[ComponentType.SWA]
        req = mock.Mock(
            swa_branching_seqlen=8,
            kv=ReqKvInfo(
                cache_protected_len=4,
                component_evicted_seqlens={ComponentType.SWA: 3},
            ),
        )

        for is_finished in (False, True):
            params = InsertParams(component_evicted_seqlens={ComponentType.MAMBA: 6})
            self.assertEqual(
                swa.prepare_for_caching_req(req, params, 12, is_finished), 8
            )
            self.assertEqual(
                params.component_evicted_seqlens,
                {ComponentType.SWA: 3, ComponentType.MAMBA: 6},
            )

        self.assertIsNone(swa.prepare_for_caching_req(req, InsertParams(), 7, False))
        req.kv.cache_protected_len = 8
        self.assertIsNone(swa.prepare_for_caching_req(req, InsertParams(), 12, False))

        cache.tree_core.is_eagle = True
        req.kv.cache_protected_len = 4
        params = InsertParams()
        self.assertEqual(swa.prepare_for_caching_req(req, params, 12, False), 9)
        params.key = RadixKey(array("q", range(9)), is_bigram=True)
        result = mock.Mock(swa_branch_inserted=False)
        swa.commit_insert_component_data(mock.Mock(), False, params, result, [])
        self.assertTrue(result.swa_branch_inserted)

    @requires_full_mamba_ps1
    def test_mamba_branching_seqlen_disabled_under_hicache(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        chunk_size = mamba_cache_chunk_size()
        tokens = self._make_seq(1, chunk_size + 1)
        self._insert(cache, allocator, req_to_token_pool, tokens)
        leaf = match_prefix(cache, tokens).last_device_node

        cache.tree_core.set_component_device_value_raw(leaf, ComponentType.MAMBA, None)
        no_hicache = match_prefix(cache, tokens)
        self.assertEqual(no_hicache.best_match_node, cache.root_node_handle())
        self.assertEqual(no_hicache.last_device_node, cache.root_node_handle())
        self.assertEqual(no_hicache.mamba_branching_seqlen, chunk_size)

        tree_h, allocator_h, req_to_token_pool_h = self._build_hicache_fixture()
        self._insert(tree_h, allocator_h, req_to_token_pool_h, tokens)
        leaf_h = match_prefix(tree_h, tokens).last_device_node
        self._backup_node(tree_h, leaf_h)
        tree_h.evict(EvictParams(num_tokens=len(tokens)))
        with_hicache = match_prefix(tree_h, tokens)
        self.assertEqual(with_hicache.best_match_node, leaf_h)
        self.assertEqual(with_hicache.last_device_node, tree_h.root_node_handle())
        self.assertIsNone(with_hicache.mamba_branching_seqlen)

    @requires_full_mamba_ps1
    def test_mamba_branching_seqlen_uses_device_full_hit_under_hicache(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chunk_size = mamba_cache_chunk_size()
        prefix = self._make_seq(1, chunk_size)
        tokens = prefix + self._make_seq(1000, chunk_size + 1)
        self._insert(cache, allocator, req_to_token_pool, prefix)
        self._insert(cache, allocator, req_to_token_pool, tokens)

        leaf = match_prefix(cache, tokens).last_device_node
        parent = cache.tree_core.get_parent_node_id(leaf)
        cache.tree_core.set_component_device_value_raw(leaf, ComponentType.MAMBA, None)

        result = match_prefix(cache, tokens)

        self.assertEqual(result.best_match_node, parent)
        self.assertEqual(result.last_device_node, parent)
        self.assertEqual(result.device_prefix_len, chunk_size)
        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.full_kv_hit_length, len(tokens))
        self.assertEqual(result.mamba_branching_seqlen, 2 * chunk_size)

    @requires_full_mamba_ps1
    def test_mamba_branching_from_host_full_is_reusable_after_insert(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chunk_size = mamba_cache_chunk_size()
        prefix = self._make_seq(1, chunk_size)
        tokens = prefix + self._make_seq(1000, chunk_size + 1)
        self._insert(cache, allocator, req_to_token_pool, prefix)
        self._insert(cache, allocator, req_to_token_pool, tokens)

        leaf = match_prefix(cache, tokens).last_device_node
        parent = cache.tree_core.get_parent_node_id(leaf)
        self.assertIsNotNone(parent)
        self._backup_node(cache, leaf)
        lock_result = cache.inc_lock_ref(parent)
        try:
            cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        finally:
            cache.dec_lock_ref(
                parent,
                lock_result.to_dec_params(),
            )
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))
        self.assertTrue(cache.tree_core.is_backuped(leaf))
        evict_result = cache.tree_core.evict_component(
            leaf, ComponentType.MAMBA, EvictLayer.HOST
        )
        cache._free_values(evict_result.device_frees, evict_result.host_frees)
        full_host_pool = cache.cache_controller.mem_pool_host
        mamba_host_pool = cache.components[ComponentType.MAMBA]._mamba_pool_host
        full_available_before = full_host_pool.available_size()
        mamba_available_before = mamba_host_pool.available_size()

        result = match_prefix(cache, tokens)

        self.assertEqual(result.best_match_node, parent)
        self.assertEqual(result.last_device_node, parent)
        self.assertEqual(result.device_prefix_len, chunk_size)
        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.full_kv_hit_length, len(tokens))
        branching_seqlen = 2 * chunk_size
        self.assertEqual(result.mamba_branching_seqlen, branching_seqlen)

        self._insert(
            cache,
            allocator,
            req_to_token_pool,
            tokens[:branching_seqlen],
        )
        cache.writing_check(write_back=True)
        # Full was already backed up, so only one Mamba slot is allocated.
        self.assertEqual(
            full_host_pool.available_size(),
            full_available_before,
        )
        self.assertEqual(
            mamba_host_pool.available_size(),
            mamba_available_before - 1,
        )

        second_match = match_prefix(cache, tokens)

        self.assertEqual(second_match.device_prefix_len, branching_seqlen)
        self.assertIsNone(second_match.mamba_branching_seqlen)
        self.assertIsNotNone(
            node_host_value(cache, second_match.last_device_node, ComponentType.MAMBA)
        )

    @requires_full_mamba_ps1
    def test_scheduler_hicache_full_mamba_init_load_back_appends_new_indices(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(cache, chain)

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        req = self._make_req(req_to_token_pool)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        loaded_len, new_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                req=req,
            )
        )

        self.assertEqual(new_node, leaf)
        self.assertEqual(req.prefix_len + loaded_len, len(tokens))
        self.assertIsNotNone(node_device_value(cache, leaf, ComponentType.MAMBA))
        self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)

    @requires(
        lambda cfg: cfg.page_size == 1, "page_size=1 keeps the expected suffix precise"
    )
    def test_scheduler_hicache_aux_only_load_back_appends_full_device_indices(self):
        aux = None
        if self.cfg.has_swa and not self.cfg.has_mamba:
            aux = ComponentType.SWA
        elif self.cfg.has_mamba and not self.cfg.has_swa:
            aux = ComponentType.MAMBA
        if aux is None:
            self.skipTest("requires exactly one aux component")

        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(cache, chain)
        leaf_full = node_device_value(cache, leaf, ComponentType.FULL).clone()
        self._backup_node(cache, leaf)
        self._set_aux_host_tombstone(cache, leaf, aux)

        req = self._make_req(req_to_token_pool)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        loaded_len, new_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                req=req,
            )
        )

        self.assertEqual(new_node, leaf)
        new_indices = cache.path_device_indices(new_node)[req.prefix_len :]
        self.assertEqual(new_indices.tolist(), leaf_full.tolist())
        self.assertEqual(req.prefix_len + loaded_len, len(tokens))
        self.assertEqual(
            node_device_value(cache, leaf, ComponentType.FULL).tolist(),
            leaf_full.tolist(),
        )
        self.assertIsNotNone(node_device_value(cache, leaf, aux))
        self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)

    @requires_full_mamba_ps1
    def test_scheduler_hicache_load_back_fallback_keeps_old_anchor(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(cache, chain)

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))

        req = self._make_req(req_to_token_pool)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        # Simulate a request without its own mamba slot so load-back allocates one
        # (that allocation is what a called-off load-back must free + not publish).
        req.kv.mamba_pool_idx = None
        avail_before = req_to_token_pool.mamba_allocator.available_size()
        loaded_len, new_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                req=req,
                # Pinning the resident ancestors leaves too little for the tail.
                mem_quota=len(tokens) - 1,
            )
        )

        self.assertEqual(loaded_len, 0)
        self.assertEqual(new_node, match.last_device_node)
        self.assertIsNone(node_device_value(cache, leaf, ComponentType.FULL))
        self.assertIsNone(node_device_value(cache, leaf, ComponentType.MAMBA))
        # A failed load-back must roll back the pre-allocated mamba slot.
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), avail_before
        )

    @requires_full_mamba_ps1
    def test_scheduler_hicache_load_back_rolls_back_mamba_on_load_failure(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(cache, chain)

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))

        req = self._make_req(req_to_token_pool)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        # Simulate a request without its own mamba slot so load-back allocates one.
        req.kv.mamba_pool_idx = None
        avail_before = req_to_token_pool.mamba_allocator.available_size()
        # H->D load fails after the mamba slot is pre-allocated -> must free it.
        with mock.patch.object(cache.cache_controller, "load", return_value=None):
            loaded_len, _ = cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=req.best_match_node,
                    host_hit_length=req.host_hit_length,
                    req=req,
                    mem_quota=None,
                )
            )

        self.assertEqual(loaded_len, 0)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), avail_before
        )

    @requires_full_mamba_ps1
    def test_scheduler_hicache_load_back_rolls_back_mamba_on_capacity_failure(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(cache, chain)

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))

        req = self._make_req(req_to_token_pool)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        # Simulate a request without its own mamba slot so load-back allocates one.
        req.kv.mamba_pool_idx = None
        avail_before = req_to_token_pool.mamba_allocator.available_size()
        # No device room and eviction frees nothing -> load-back bails after the
        # mamba pre-alloc, which must still be freed.
        with (
            mock.patch.object(
                cache.token_to_kv_pool_allocator, "available_size", return_value=0
            ),
            mock.patch.object(
                cache, "evict", return_value=mock.Mock(num_tokens_evicted=0)
            ),
        ):
            loaded_len, _ = cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=req.best_match_node,
                    host_hit_length=req.host_hit_length,
                    req=req,
                    mem_quota=None,
                )
            )

        self.assertEqual(loaded_len, 0)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), avail_before
        )

    @requires_full_mamba_ps1
    def test_load_back_abort_frees_unpublished_mamba_slot(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))
        self.assertIsNotNone(node_host_value(cache, leaf, ComponentType.MAMBA))

        # A request whose mamba slot was released: load_back's CoW arm allocates one.
        req = self._make_req(req_to_token_pool)
        req_to_token_pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        req.kv.mamba_pool_idx = None
        mamba_avail = req_to_token_pool.mamba_allocator.available_size()

        # Impossible quota -> load_back aborts after building the transfers.
        loaded = cache.load_back(leaf, mem_quota=-(10**9), req=req)

        self.assertFalse(loaded)
        # the aborted call must return its slot and not leave req pointing at it
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), mamba_avail
        )
        self._release_ongoing_load_back_locks(cache)

    @requires_full_mamba_ps1
    def test_load_back_abort_keeps_preexisting_mamba_slot(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        # The request already owns its slot: an aborted load-back must not free it.
        req = self._make_req(req_to_token_pool)
        preexisting_slot = req.kv.mamba_pool_idx
        self.assertIsNotNone(preexisting_slot)
        mamba_avail = req_to_token_pool.mamba_allocator.available_size()

        loaded = cache.load_back(leaf, mem_quota=-(10**9), req=req)

        self.assertFalse(loaded)
        self.assertIs(req.kv.mamba_pool_idx, preexisting_slot)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), mamba_avail
        )
        self._release_ongoing_load_back_locks(cache)

    @requires_full_mamba_ps1
    def test_load_back_success_publishes_fresh_mamba_slot(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        req = self._make_req(req_to_token_pool)
        req_to_token_pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        req.kv.mamba_pool_idx = None
        mamba_avail = req_to_token_pool.mamba_allocator.available_size()

        loaded = cache.load_back(leaf, req=req)

        self.assertTrue(loaded)
        # the successful load must keep the freshly allocated slot published
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNotNone(node_device_value(cache, leaf, ComponentType.MAMBA))
        # one slot restores the node's mamba value, one is the request's CoW slot
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), mamba_avail - 2
        )
        self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)

    @requires(
        lambda cfg: (
            cfg.has_mamba
            and not cfg.has_swa
            and cfg.page_size == 1
            and not cfg.enable_mamba_extra_buffer
            and not cfg.enable_int8_mamba_checkpoint
        ),
        "requires page_size=1 Full+Mamba without extra buffers",
    )
    def test_admission_keeps_mamba_h2d_destination_on_full_overdelivery(self):
        cache, allocator, pool = build_fixture(replace(self.cfg, kv_size=20))
        self._init_hicache(cache, write_policy="write_back")
        tokens = list(range(12))
        self._insert(cache, allocator, pool, tokens)
        leaf = match_prefix(cache, tokens).last_device_node
        self._fill_mamba_state(
            pool, node_device_value(cache, leaf, ComponentType.MAMBA), marker=11
        )
        self._backup_node(cache, leaf)
        self._set_aux_host_tombstone(cache, leaf, ComponentType.MAMBA)
        self.assertEqual(cache.evict_host(12, ComponentType.FULL), 12)

        req = Req(
            rid="mamba-overdelivery",
            origin_input_text="",
            origin_input_ids=array("q", range(16)),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.init_next_round_input(cache)
        self.assertEqual((req.prefix_len, req.host_hit_length), (0, 0))
        self.assertEqual(req.mamba_host_hit_length, 1)
        adder = PrefillAdder(
            page_size=1,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=100,
            rem_chunk_tokens=None,
        )
        adder.add_one_req(req, False, None)

        # The load pins 12 FULL slots, leaving 8 free. Rechecking the original
        # 18-token demand would reject and scheduler cleanup would free the CoW
        # destination even though the queued H2D still targets it.
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(req.prefix_len, 12)
        self.assertEqual(req.kv.cache_protected_len, 12)
        slot = req.kv.mamba_pool_idx.unsqueeze(0)
        other_slots = pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
        self.assertNotIn(slot.item(), other_slots.tolist())
        self._finish_pending_loads(cache)
        temporal, _ = self._snapshot_mamba_state(pool, slot)
        self.assertTrue(torch.all(temporal == 11))
        pool.mamba_allocator.free(other_slots)
        cache.unlock(req.lock)

    @requires_full_mamba_ps1
    def test_load_back_success_copies_mamba_state_into_request_slot(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]

        # Stamp the node's mamba state so the host backup carries it.
        node_mamba_indices = node_device_value(cache, leaf, ComponentType.MAMBA).clone()
        self._fill_mamba_state(req_to_token_pool, node_mamba_indices, marker=11)
        expected_temporal, expected_conv = self._snapshot_mamba_state(
            req_to_token_pool, node_mamba_indices
        )

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        req = self._make_req(req_to_token_pool)
        req_to_token_pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        req.kv.mamba_pool_idx = None

        loaded = cache.load_back(leaf, req=req)
        self.assertTrue(loaded)
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self._finish_pending_loads(cache)

        # The CoW slot must actually hold the backed-up mamba state, not merely exist.
        actual_temporal, actual_conv = self._snapshot_mamba_state(
            req_to_token_pool, req.kv.mamba_pool_idx.unsqueeze(0)
        )
        self.assertTrue(torch.equal(actual_temporal, expected_temporal))
        self.assertEqual(len(actual_conv), len(expected_conv))
        for actual, expected in zip(actual_conv, expected_conv):
            self.assertTrue(torch.equal(actual, expected))
        self._release_ongoing_load_back_locks(cache)

    @requires_full_mamba_ps1
    def test_prepare_load_back_mamba(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        comp = cache.components[ComponentType.MAMBA]

        self._backup_node(cache, leaf)
        req = self._make_req(req_to_token_pool)
        req_to_token_pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        req.kv.mamba_pool_idx = None

        # device value still present -> nothing to prepare even though host-backed
        self.assertIsNone(comp.prepare_load_back(leaf, req=req).allocated_mamba_slot)
        self.assertIsNone(req.kv.mamba_pool_idx)

        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        # a request that already owns a slot -> nothing to prepare
        req_owned = self._make_req(req_to_token_pool)
        self.assertIsNone(
            comp.prepare_load_back(leaf, req=req_owned).allocated_mamba_slot
        )

        # no request -> nothing to prepare
        self.assertIsNone(comp.prepare_load_back(leaf, req=None).allocated_mamba_slot)

        # fresh request + host-only mamba -> allocates and publishes onto req
        prep = comp.prepare_load_back(leaf, req=req)
        self.assertIsNotNone(prep.allocated_mamba_slot)
        self.assertEqual(int(req.kv.mamba_pool_idx), int(prep.allocated_mamba_slot[0]))

        # node without host-backed mamba -> nothing to prepare
        req2 = self._make_req(req_to_token_pool)
        req_to_token_pool.mamba_allocator.free(req2.kv.mamba_pool_idx.unsqueeze(0))
        req2.kv.mamba_pool_idx = None
        root = cache.root_node_handle()
        self.assertIsNone(node_host_value(cache, root, ComponentType.MAMBA))
        self.assertIsNone(comp.prepare_load_back(root, req=req2).allocated_mamba_slot)
        self.assertIsNone(req2.kv.mamba_pool_idx)

    @requires_full_mamba_ps1
    def test_prepare_load_back_mamba_pool_exhausted(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        comp = cache.components[ComponentType.MAMBA]

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))

        req = self._make_req(req_to_token_pool)
        req_to_token_pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        req.kv.mamba_pool_idx = None
        retry_slot = req_to_token_pool.mamba_allocator.alloc(1)

        # first alloc fails -> prepare must evict a mamba slot and retry
        with (
            mock.patch.object(
                req_to_token_pool.mamba_allocator,
                "alloc",
                side_effect=[None, retry_slot],
            ),
            mock.patch.object(
                cache, "evict_for_alloc", autospec=True
            ) as evict_for_alloc,
        ):
            prep = comp.prepare_load_back(leaf, req=req)
        evict_for_alloc.assert_called_once_with(EvictParams(num_tokens=0, mamba_num=1))
        self.assertIs(prep.allocated_mamba_slot, retry_slot)
        self.assertEqual(int(req.kv.mamba_pool_idx), int(retry_slot[0]))

    @requires_swa
    @requires(
        lambda cfg: not cfg.has_mamba,
        "SWA-only path keeps the chain construction simple",
    )
    def test_hicache_swa_load_back_min_suffix(self):
        ps = self.cfg.page_size
        sw = self.cfg.sliding_window_size
        expected_pages = (sw + ps - 1) // ps
        chain_pages = expected_pages + 2
        if chain_pages * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small for the desired chain")

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        chain = self._build_chain_pages(
            cache, allocator, req_to_token_pool, chain_pages
        )
        if len(chain) <= expected_pages:
            self.skipTest("chain collapsed below the suffix length being tested")

        self._simulate_backup_tree(cache)

        # Tombstone every chain node on the device side without going through
        # the tree-wide eviction loop. This isolates build_hicache_transfers
        # from LRU and cascade ordering.
        for n in chain:
            cache.tree_core.set_component_device_value_raw(n, ComponentType.FULL, None)
            cache.tree_core.set_component_device_value_raw(n, ComponentType.SWA, None)

        leaf = chain[-1]
        transfers = cache.tree_core.build_hicache_transfers(
            ComponentType.SWA, leaf, CacheTransferPhase.LOAD_BACK
        )
        self.assertIsNotNone(transfers)
        self.assertEqual(len(transfers), 1)
        xfer = transfers[0]
        self.assertEqual(xfer.name, PoolName.SWA)
        self.assertEqual(len(xfer.nodes_to_load), expected_pages)
        # host_indices must cover exactly the expected suffix tokens (>= sw).
        self.assertEqual(int(xfer.host_indices.numel()), expected_pages * ps)
        self.assertGreaterEqual(int(xfer.host_indices.numel()), sw)
        self.assertEqual(xfer.nodes_to_load, chain[-expected_pages:])

    @requires_swa
    @requires(
        lambda cfg: not cfg.has_mamba,
        "SWA-only path keeps the chain construction simple",
    )
    @requires(
        lambda cfg: cfg.sliding_window_size > cfg.page_size,
        "window must span past the leaf to reach the pin",
    )
    def test_hicache_swa_load_back_rejects_foreign_pinned_window(self):
        """A second load-back must not claim nodes pinned by an in-flight one.

        commit_load_back republishes Full device values before the DMA acks, so
        a later anchor can claim just the still-host-only SWA window of a
        pinned node and hit the commit pin assert. The spec build must degrade
        to an empty spec instead (the caller recomputes).
        """

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        # commit_load_back pins its source nodes only under write-back; run the
        # scenario in that mode so the foreign-pin path is reachable.
        cache.is_write_back = True
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain collapsed below the two-node suffix being tested")
        self._simulate_backup_tree(cache)

        # Host-only suffix a -> b under a device-resident ancestor.
        a, b = chain[-2], chain[-1]
        for n in (a, b):
            cache.tree_core.set_component_device_value_raw(n, ComponentType.FULL, None)
            cache.tree_core.set_component_device_value_raw(n, ComponentType.SWA, None)
            if cache.tree_core.is_node_in_device_lru(n, ComponentType.SWA):
                cache.tree_core.remove_node_from_device_lru(n, ComponentType.SWA)
            cache.tree_core.insert_node_into_host_lru(n, ComponentType.SWA)

        # Anchor `a`: a Full-only load whose SWA slice stays host-only.
        kv_xfer, _comp_xfers = cache.tree_core.build_load_back_spec(a)
        self.assertEqual(kv_xfer.nodes_to_load, [a])
        device_indices = torch.arange(
            int(kv_xfer.host_indices.numel()), dtype=torch.int64, device=cache.device
        )
        cache.tree_core.commit_load_back(a, device_indices, kv_xfer, {})

        # Anchor `b` rejects its whole spec: its SWA window claims pinned `a`.
        kv_xfer, comp_xfers = cache.tree_core.build_load_back_spec(b)
        self.assertEqual(int(kv_xfer.host_indices.numel()), 0)
        self.assertEqual(kv_xfer.nodes_to_load, [])
        self.assertEqual(comp_xfers, {})

        # After the ack unpins, the same spec builds fully.
        cache.tree_core.finish_load_back(a)
        kv_xfer, comp_xfers = cache.tree_core.build_load_back_spec(b)
        self.assertEqual(kv_xfer.nodes_to_load, [b])
        self.assertEqual(comp_xfers[ComponentType.SWA][0].nodes_to_load, [a, b])

    @requires_swa
    @requires(
        lambda cfg: not cfg.has_mamba,
        "SWA-only path keeps the chain construction simple",
    )
    def test_hicache_swa_commit_load_back_rebuilds_mapping(self):
        """LOAD_BACK commit must:
        (1) restore SWA cd.value via _restore_device_value (host LRU -> device LRU),
        (2) rewrite full_to_swa_index_mapping[full_idx] = new_swa_idx for every
            loaded chunk so subsequent SWA reads via translate_loc_from_full_to_swa
            return the freshly allocated SWA slot."""

        cache, allocator, _, chain, window_pages = self._swa_finalize_setup()

        # Tombstone every SWA node in the trailing window.
        loaded_nodes = chain[-window_pages:]
        for n in loaded_nodes:
            cache.tree_core.set_component_device_value_raw(n, ComponentType.SWA, None)
            # SWA LRU bookkeeping must reflect tombstone state for the
            # _restore_device_value path to exercise the host->device move.
            cache.tree_core.remove_node_from_device_lru(n, ComponentType.SWA)
            cache.tree_core.insert_node_into_host_lru(n, ComponentType.SWA)

        # Build the LOAD_BACK transfer the same way load_back() would.
        transfers = cache.tree_core.build_hicache_transfers(
            ComponentType.SWA, chain[-1], CacheTransferPhase.LOAD_BACK
        )
        self.assertIsNotNone(transfers)
        xfer = transfers[0]
        self.assertEqual(xfer.nodes_to_load, loaded_nodes)

        # Allocate SWA device slots from the inner allocator (mirrors how
        # _resolve_device_transfers routes via device_alloc_fn ->
        # swa_attn_allocator.alloc on the load-back path).
        n_swa = int(xfer.host_indices.numel())
        new_swa = allocator.swa_attn_allocator.alloc(n_swa)
        self.assertIsNotNone(new_swa)
        xfer.device_indices = new_swa

        # Snapshot pre-commit state for invariants checks.
        pre_evictable = cache.tree_core.component_evictable_size(ComponentType.SWA)

        load_actions = []
        cache.tree_core.commit_hicache_transfers(
            chain[-1],
            CacheTransferPhase.LOAD_BACK,
            {ComponentType.SWA: transfers},
            cache_actions=load_actions,
        )
        cache._apply_cache_actions(load_actions)

        # (1) cd.value restored, host LRU -> device LRU swap done.
        offset = 0
        for n in loaded_nodes:
            value = node_device_value(cache, n, ComponentType.SWA)
            self.assertIsNotNone(value)
            chunk_len = int(value.numel())
            self.assertEqual(
                value.tolist(),
                new_swa[offset : offset + chunk_len].tolist(),
            )
            offset += chunk_len
            self.assertTrue(cache.tree_core.is_node_in_device_lru(n, ComponentType.SWA))
            self.assertFalse(cache.tree_core.is_node_in_host_lru(n, ComponentType.SWA))
        self.assertEqual(offset, n_swa)

        # (2) full_to_swa_index_mapping rebuilt for every loaded chunk.
        for n in loaded_nodes:
            full_idx = node_device_value(cache, n, ComponentType.FULL)
            swa_idx = node_device_value(cache, n, ComponentType.SWA)
            translated = allocator.translate_loc_from_full_to_swa(full_idx)
            self.assertEqual(translated.tolist(), swa_idx.tolist())

        # Evictable size moved up by the restored token count.
        self.assertEqual(
            cache.tree_core.component_evictable_size(ComponentType.SWA) - pre_evictable,
            n_swa,
        )

    def test_hicache_swa_load_back_anchored_on_best_match_node(self):
        cache, _, _, y, x, _ = self._swa_anchor_setup()
        ps = self.cfg.page_size

        transfers = cache.tree_core.build_hicache_transfers(
            ComponentType.SWA, x, CacheTransferPhase.LOAD_BACK
        )
        self.assertEqual(len(transfers), 1)
        xfer = transfers[0]
        self.assertEqual(xfer.name, PoolName.SWA)
        self.assertEqual(xfer.nodes_to_load, [y])
        self.assertEqual(int(xfer.host_indices.numel()), ps)

        with self.assertRaises(AssertionError):
            cache.tree_core.build_hicache_transfers(
                ComponentType.SWA, y, CacheTransferPhase.LOAD_BACK
            )

    @requires_swa
    @requires(lambda cfg: not cfg.has_mamba, "SWA-only path")
    @requires(
        lambda cfg: cfg.page_size <= 1,
        "page_size==1 for direct swa_attn_allocator access",
    )
    def test_hicache_swa_load_back_uses_full_pool_capacity(self):

        cache, allocator, req_to_token_pool = self._build_hicache_fixture()

        sw = self.cfg.sliding_window_size
        kv_tokens = sw + 2
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, kv_tokens)
        if len(chain) < kv_tokens:
            self.skipTest("chain too short")
        leaf = chain[-1]

        self._backup_tree(cache)
        result = cache.evict(EvictParams(num_tokens=kv_tokens))
        self.assertGreaterEqual(result.num_tokens_evicted, kv_tokens)
        self.assertIsNone(node_device_value(cache, leaf, ComponentType.FULL))

        kv_xfer = cache.tree_core.build_hicache_transfers(
            ComponentType.FULL, leaf, CacheTransferPhase.LOAD_BACK
        )[0]
        self.assertEqual(int(kv_xfer.host_indices.numel()), kv_tokens)

        swa_xfer = cache.tree_core.build_hicache_transfers(
            ComponentType.SWA, leaf, CacheTransferPhase.LOAD_BACK
        )[0]
        self.assertEqual(int(swa_xfer.host_indices.numel()), sw)

        # Leave tree-owned SWA available for controller-side SWA eviction.
        unrelated_seq = self._make_seq(100_000, sw)
        self._insert(cache, allocator, req_to_token_pool, unrelated_seq)
        self.assertGreaterEqual(cache.swa_evictable_size(), sw)

        # Make raw SWA availability smaller than both load-back transfers.
        target_swa_avail = sw - 1
        swa_avail = allocator.swa_attn_allocator.available_size()
        self.assertGreaterEqual(swa_avail, target_swa_avail)
        if swa_avail > target_swa_avail:
            external_swa = allocator.swa_attn_allocator.alloc(
                swa_avail - target_swa_avail
            )
            self.assertIsNotNone(external_swa)

        self.assertGreaterEqual(
            allocator.full_attn_allocator.available_size(),
            int(kv_xfer.host_indices.numel()),
        )
        self.assertLess(
            allocator.swa_attn_allocator.available_size(),
            int(kv_xfer.host_indices.numel()),
        )
        self.assertLess(
            allocator.swa_attn_allocator.available_size(),
            int(swa_xfer.host_indices.numel()),
        )

        with mock.patch.object(
            cache, "evict_for_alloc", wraps=cache.evict_for_alloc
        ) as evict_for_alloc_mock:
            self.assertTrue(cache.load_back(leaf))

        # Full pre-eviction must not be triggered by SWA pool pressure.
        full_pre_evict_calls = [
            call
            for call in evict_for_alloc_mock.call_args_list
            if call.args and call.args[0].num_tokens > 0
        ]
        self.assertEqual(full_pre_evict_calls, [])

        # SWA shortage is handled by the controller through SWA-only eviction.
        self.assertTrue(
            any(
                call.args
                and call.args[0].num_tokens == 0
                and call.args[0].swa_num_tokens > 0
                for call in evict_for_alloc_mock.call_args_list
            )
        )

        self._finish_pending_loads(cache)
        self.assertIsNotNone(node_device_value(cache, leaf, ComponentType.FULL))
        self._release_ongoing_load_back_locks(cache)
        cache.sanity_check()


define_config_test_classes(UnifiedRadixCacheLoadBackSuite, globals())


if __name__ == "__main__":
    unittest.main()

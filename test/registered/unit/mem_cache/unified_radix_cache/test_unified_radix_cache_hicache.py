"""HiCache device <-> host backup, demotion and eviction tests for UnifiedRadixCache."""

import unittest
from array import array
from dataclasses import replace
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
    TreeLock,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    FreeDeviceKV,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
    define_config_test_classes,
    requires,
    requires_full_mamba_ps1,
    requires_full_swa_ps1,
    requires_full_swa_ps1_window4,
    requires_hicache,
    requires_mamba,
    requires_swa,
    requires_swa_and_mamba,
    requires_swa_only,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_children,
    node_device_lock_ref,
    node_device_value,
    node_host_lock_ref,
    node_host_value,
    node_key_length,
    node_parent,
    node_token_ids,
    selected_tree_core_test_backend,
)
from sglang.test.kits.unified_radix_cache.hicache import (
    UnifiedRadixCacheHiCacheFixtures,
)

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=12, suite="stage-b-test-1-gpu-small-amd")


class UnifiedRadixCacheHiCacheSuite(UnifiedRadixCacheHiCacheFixtures):
    @requires_swa_and_mamba
    def test_dec_swa_lock_only_hicache_child_on_host_treated_as_device_leaf(self):

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        seq_ab = seq_a + self._make_seq(7000, 2)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_ab)

        node_a = match_prefix(cache, seq_a).last_device_node
        self.assertGreater(len(node_children(cache, node_a)), 0, "A must have children")

        self.assertFalse(
            cache.tree_core.is_device_leaf(node_a),
            "A is not a device-leaf while child holds Full on device",
        )

        self._simulate_backup(cache, node_a)
        self.assertTrue(
            cache.tree_core.is_backuped(node_a),
            "node_a must be backuped (invariant a)",
        )

        def _collect_descendants(node_id):
            out = []
            for child_id in node_children(cache, node_id):
                out.extend(_collect_descendants(child_id))
                out.append(child_id)
            return out

        descendants = _collect_descendants(node_a)
        self.assertGreater(len(descendants), 0)
        for desc in descendants:
            self._simulate_backup(cache, desc)
            self.assertTrue(
                cache.tree_core.is_backuped(desc),
                "desc must be backuped before demote",
            )
            result = cache.tree_core.demote(desc)
            cache._free_values(result.device_frees, result.host_frees)
            self.assertTrue(
                cache.tree_core.is_full_device_evicted(desc),
                "desc should be D->H demoted",
            )
            self.assertIsNone(node_device_value(cache, desc, ComponentType.FULL))

        self.assertTrue(
            cache.tree_core.is_device_leaf(node_a),
            "A is a HiCache device-leaf (no child with Full on device)",
        )
        self.assertGreater(
            len(node_children(cache, node_a)), 0, "A still has tree-children"
        )
        self.assertTrue(cache.tree_core.is_device_evictable_leaf(node_a))
        cache.sanity_check()

        lock_result = cache.inc_lock_ref(node_a)
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.SWA), 1
        )
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.MAMBA), 1
        )
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL), 1
        )

        cache.release_swa(TreeLock(node_a, lock_result.to_dec_params()))
        self.assertEqual(
            node_device_lock_ref(cache, node_a, ComponentType.SWA), 0, "SWA released"
        )
        self.assertEqual(
            node_device_lock_ref(cache, node_a, ComponentType.MAMBA),
            0,
            "Mamba dropped by dec_swa_lock_only",
        )
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node_a, ComponentType.FULL),
            1,
            "Full kept by contract",
        )
        self.assertIsNotNone(
            node_device_value(cache, node_a, ComponentType.SWA),
            "SWA slot stays under contract (lazy reclaim by drive_eviction)",
        )
        self.assertTrue(
            cache.tree_core.is_node_in_device_lru(node_a, ComponentType.SWA),
            "SWA stays in LRU for drive_eviction to pick later",
        )

        cache.dec_lock_ref(node_a, lock_result.to_dec_params(), skip_swa=True)
        self.assertTrue(cache.tree_core.is_device_leaf(node_a))
        cache.sanity_check()

    @requires_hicache
    def test_hicache_write_back_evict_drops_unbacked_leaf_when_host_full(self):
        """Write-back eviction will keep freeing device KV when the host pool
        is exhausted and host eviction cannot free space to prevent OOM."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")

        # Two-node chain: the drop must cascade parent-ward across eviction
        # iterations, not just delete a single leaf.
        seq_parent = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq_parent)
        seq = seq_parent + self._make_seq(1000, 1)
        self._insert(cache, allocator, req_to_token_pool, seq)

        # Exhaust the KV host pool. The tree has no host leaves, so
        # evict_host cannot free anything and every D->H backup fails.
        host_pool = cache.cache_controller.mem_pool_host
        self.assertIsNotNone(host_pool.alloc(host_pool.available_size()))
        self.assertEqual(host_pool.available_size(), 0)

        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))

        # The chain is gone entirely: no device hit, no host hit.
        m = match_prefix(cache, seq)
        self.assertEqual(m.device_prefix_len, 0)
        self.assertEqual(m.host_hit_length, 0)
        cache.sanity_check()

    @requires_hicache
    def test_hicache_write_back_drop_respects_pins_then_frees_subtree(self):
        """The host-pressure drop fallback must decline while any node in the
        unbacked subtree is host-pinned, then reclaim the whole subtree --
        including a demoted child's host backup -- once unpinned."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        host_pool = cache.cache_controller.mem_pool_host
        baseline_host = host_pool.available_size()

        parent_seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, parent_seq)
        child_seq = parent_seq + self._make_seq(1000, 1)
        self._insert(cache, allocator, req_to_token_pool, child_seq)
        m = match_prefix(cache, child_seq)
        child = m.last_device_node

        # Evict only the child leaf -> real backup + demote, leaving it
        # host-only under a still-unbacked device parent (write-back backs up
        # single nodes leaf-first, so this is a normal intermediate state).
        child_len = node_key_length(cache, child)
        result = cache.evict(EvictParams(num_tokens=child_len))
        self.assertGreaterEqual(result.num_tokens_evicted, child_len)
        self.assertTrue(cache.tree_core.is_full_device_evicted(child))
        self.assertTrue(cache.tree_core.is_backuped(child))
        parent = node_parent(cache, child)
        self.assertFalse(cache.tree_core.is_backuped(parent))
        self.assertGreater(baseline_host - host_pool.available_size(), 0)

        # From here every backup fails (controller.write returns None), so
        # each evict() attempts the drop fallback on the parent.
        with mock.patch.object(cache.cache_controller, "write", return_value=None):
            # Pinned subtree root: drop declines, chain stays intact.
            parent_lock_params = cache.inc_host_lock_ref(parent).to_dec_params()
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
            self.assertEqual(result.num_tokens_evicted, 0)
            cache.dec_host_lock_ref(parent, parent_lock_params)

            # Pinned host-only descendant: drop declines as well.
            child_lock_params = cache.inc_host_lock_ref(child).to_dec_params()
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
            self.assertEqual(result.num_tokens_evicted, 0)
            m = match_prefix(cache, parent_seq)
            self.assertEqual(m.device_prefix_len, len(parent_seq))
            cache.dec_host_lock_ref(child, child_lock_params)

            # Unpinned: the subtree drops and the child's host slots return.
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(parent_seq))
        self.assertEqual(host_pool.available_size(), baseline_host)
        m = match_prefix(cache, child_seq)
        self.assertEqual(m.device_prefix_len, 0)
        self.assertEqual(m.host_hit_length, 0)
        cache.sanity_check()

    @requires(
        lambda cfg: cfg == CacheConfig(), "single Full page-size-1 ownership regression"
    )
    def test_hicache_split_preserves_inflight_full_host_pin(self):
        """A split prefix keeps an outstanding storage backup's host ownership."""

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        host_pool = cache.cache_controller.mem_pool_host
        baseline_host = host_pool.available_size()

        seq = [1, 2, 3, 4]
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = match_prefix(cache, seq).last_device_node
        self._backup_node(cache, leaf)

        lock_params = cache.inc_host_lock_ref(leaf).to_dec_params()
        self.assertIsNotNone(
            lock_params.get_lock_uuid(ComponentType.FULL, lock_host=True)
        )
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 9, 10])
        split_parent = node_parent(cache, leaf)
        split_host = node_host_value(cache, split_parent, ComponentType.FULL)
        self.assertEqual(node_host_lock_ref(cache, split_parent, ComponentType.FULL), 1)
        self.assertEqual(node_host_lock_ref(cache, leaf, ComponentType.FULL), 1)

        # Before the fix the prefix had host_lock_ref == 0 and these slots were
        # reclaimed while the storage thread could still be reading them.
        self.assertEqual(cache.evict_host(len(split_host)), 0)

        # A lock acquired after the split covers only the suffix and gets a new
        # boundary there. Releasing it must not consume the older segment lock
        # that was propagated to the prefix.
        suffix_lock_params = cache.inc_host_lock_ref(leaf).to_dec_params()
        self.assertEqual(node_host_lock_ref(cache, split_parent, ComponentType.FULL), 1)
        self.assertEqual(node_host_lock_ref(cache, leaf, ComponentType.FULL), 2)
        self.assertNotEqual(
            suffix_lock_params.get_lock_uuid(ComponentType.FULL, lock_host=True),
            lock_params.get_lock_uuid(ComponentType.FULL, lock_host=True),
        )
        cache.dec_host_lock_ref(leaf, suffix_lock_params)
        self.assertEqual(node_host_lock_ref(cache, split_parent, ComponentType.FULL), 1)
        self.assertEqual(node_host_lock_ref(cache, leaf, ComponentType.FULL), 1)
        self.assertEqual(cache.evict_host(len(split_host)), 0)

        cache.dec_host_lock_ref(leaf, lock_params)
        self.assertEqual(node_host_lock_ref(cache, split_parent, ComponentType.FULL), 0)
        self.assertEqual(node_host_lock_ref(cache, leaf, ComponentType.FULL), 0)
        self.assertEqual(cache.evict_host(len(seq)), len(seq))
        self.assertEqual(host_pool.available_size(), baseline_host)
        cache.sanity_check()

    def test_hicache_split_preserves_inflight_swa_host_pin(self):
        """A split SWA host segment stays pinned and releases without leaks."""
        expected_cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=4,
        )
        if self.cfg != expected_cfg:
            self.skipTest("single page-size-1 Full+SWA host-lock regression")

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        swa_host_pool = cache.swa_kv_pool_host

        seq = [1, 2, 3, 4, 5, 6, 7, 8]
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = match_prefix(cache, seq).last_device_node
        parent_before_split = node_parent(cache, leaf)
        self._backup_node(cache, leaf)
        available_after_backup = swa_host_pool.available_size()

        lock_params = cache.inc_host_lock_ref(leaf).to_dec_params()
        self.assertIsNotNone(
            lock_params.get_lock_uuid(ComponentType.FULL, lock_host=True)
        )
        self.assertIsNotNone(
            lock_params.get_lock_uuid(ComponentType.SWA, lock_host=True)
        )

        # Diverge inside the locked SWA window. The leaf handle stays on the
        # suffix while the lock boundary and copied host refs move to the new
        # prefix, so neither half may be reclaimed by component host eviction.
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4, 5, 6, 9, 10])
        split_parent = node_parent(cache, leaf)
        self.assertNotEqual(split_parent, parent_before_split)
        self.assertEqual(node_parent(cache, split_parent), parent_before_split)
        self.assertEqual(node_host_lock_ref(cache, split_parent, ComponentType.SWA), 1)
        self.assertEqual(node_host_lock_ref(cache, leaf, ComponentType.SWA), 1)

        cache.dec_host_lock_ref(leaf, lock_params)
        self.assertEqual(node_host_lock_ref(cache, split_parent, ComponentType.SWA), 0)
        self.assertEqual(node_host_lock_ref(cache, leaf, ComponentType.SWA), 0)
        self.assertEqual(swa_host_pool.available_size(), available_after_backup)
        cache.sanity_check()

    @requires_hicache
    def test_hicache_evict_device_leaf_aborts_demote_when_backup_fails(self):
        """The tree op defers a write-back victim's demote: it returns a BackupKV
        without freeing device, and _demote requires a completed backup, so a
        failed backup leaves the node recoverable on device."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        ct = ComponentType.FULL

        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)
        m = match_prefix(cache, seq)
        node = m.last_device_node
        self.assertNotEqual(node, cache.root_node_handle())
        self.assertFalse(cache.tree_core.is_backuped(node))
        self.assertFalse(cache.tree_core.is_full_device_evicted(node))

        # Tree op defers: returns a BackupKV and leaves the node on device (no demote).
        leaf_result = cache.tree_core.evict_device_leaf(node, is_write_back=True)
        self.assertIsNotNone(leaf_result.backup_kv)
        cache._free_values(leaf_result.device_frees, leaf_result.host_frees)

        self.assertFalse(cache.tree_core.is_full_device_evicted(node))
        self.assertIsNotNone(node_device_value(cache, node, ct))
        self.assertIsNone(node_host_value(cache, node, ct))

        # Demote needs a completed backup: _demote asserts on the un-backed node.
        with self.assertRaises(AssertionError):
            cache.tree_core.demote(node)

        cache.sanity_check()

    @requires_hicache
    def test_hicache_evict_to_host(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)

        m = match_prefix(cache, seq)
        node = m.last_device_node

        self._backup_node(cache, node)
        self.assertTrue(cache.tree_core.is_backuped(node))

        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))

        self.assertTrue(cache.tree_core.is_full_device_evicted(node))
        self.assertTrue(cache.tree_core.is_backuped(node))
        self.assertIsNone(node_device_value(cache, node, ComponentType.FULL))
        self.assertIsNotNone(node_host_value(cache, node, ComponentType.FULL))

        self.assertFalse(cache.tree_core.is_device_evictable_leaf(node))
        self.assertTrue(cache.tree_core.is_host_evictable_leaf(node))
        cache.sanity_check()

    @requires_hicache
    def test_hicache_partial_match_splits_evicted_backed_up_node(self):
        """Partial matches on host-only nodes must keep the host prefix usable."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        ps = self.cfg.page_size
        seq = self._make_seq(1, 4)
        expected_prefix = seq[: 2 * ps]
        expected_suffix = seq[len(expected_prefix) :]
        query = expected_prefix + self._make_seq(9000, 1)

        self._insert(cache, allocator, req_to_token_pool, seq)
        m = match_prefix(cache, seq)
        node = m.last_device_node
        self._backup_node(cache, node)

        cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(node))
        self.assertTrue(cache.tree_core.is_backuped(node))

        m = match_prefix(cache, query)

        self.assertEqual(m.device_prefix_len, 0)
        self.assertEqual(m.last_device_node, cache.root_node_handle())

        # Locate the host prefix via last_host_node and rebuild prefix/suffix
        # from path keys (a leaf may span several nodes).
        if self.cfg.has_mamba:
            self.assertEqual(m.host_hit_length, 0)
            self.assertEqual(m.last_host_node, cache.root_node_handle())
        else:
            self.assertEqual(m.host_hit_length, len(expected_prefix))
            split_parent = m.last_host_node
            self.assertNotEqual(split_parent, cache.root_node_handle())
            self.assertTrue(cache.tree_core.is_full_device_evicted(split_parent))
            self.assertTrue(cache.tree_core.is_backuped(split_parent))
            # root -> split_parent keys reconstruct expected_prefix
            prefix_tokens: list[int] = []
            chain = []
            cur = split_parent
            while cur != cache.root_node_handle():
                chain.append(cur)
                cur = node_parent(cache, cur)
            for n in reversed(chain):
                prefix_tokens.extend(node_token_ids(cache, n))
            self.assertEqual(prefix_tokens, expected_prefix)
            # the diverged suffix stays as evicted+backuped descendant(s)
            suffix_tokens: list[int] = []
            cur = split_parent
            while children := node_children(cache, cur):
                self.assertEqual(len(children), 1)
                cur = children[0]
                suffix_tokens.extend(node_token_ids(cache, cur))
            self.assertEqual(suffix_tokens, expected_suffix)
            self.assertTrue(cache.tree_core.is_full_device_evicted(cur))
            self.assertTrue(cache.tree_core.is_backuped(cur))
        cache.sanity_check()

    @requires_hicache
    def test_hicache_host_leaf_eviction(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)

        m = match_prefix(cache, seq)
        node = m.last_device_node

        self._backup_node(cache, node)
        cache.evict(EvictParams(num_tokens=len(seq)))

        self.assertTrue(cache.tree_core.is_full_device_evicted(node))
        self.assertTrue(cache.tree_core.is_host_evictable_leaf(node))

        cache.evict_host(len(seq))

        self.assertFalse(cache.tree_core.is_host_evictable_leaf(node))
        self.assertEqual(len(node_children(cache, cache.root_node_handle())), 0)
        cache.sanity_check()

    @requires_hicache
    def test_hicache_evict_drops_unlocked_node_when_backup_fails(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)
        m = match_prefix(cache, seq)
        node = m.last_device_node
        self.assertNotEqual(node, cache.root_node_handle())
        self.assertFalse(cache.tree_core.is_backuped(node))
        self.assertFalse(cache.tree_core.is_full_device_evicted(node))

        # Backup failure falls back to dropping the unlocked subtree.
        with mock.patch.object(cache, "_execute_and_commit_kv_backup", return_value=0):
            result = cache.evict(EvictParams(num_tokens=len(seq)))

        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))
        self.assertFalse(cache.tree_core.contains_node(node))
        self.assertEqual(len(node_children(cache, cache.root_node_handle())), 0)
        cache.sanity_check()

    @requires_swa
    @requires(
        lambda cfg: not cfg.has_mamba, "SWA-only path keeps the split setup simple"
    )
    def test_hicache_write_through_offloads_swa_split_leaf(self):

        ps = self.cfg.page_size
        cache, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.write_through_threshold = 1

        seq = self._make_seq(1, 2)
        value = self._alloc(allocator, len(seq))
        result = cache.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=value,
                component_evicted_seqlens={ComponentType.SWA: ps},
            )
        )
        self.assertEqual(result.prefix_len, 0)

        (split_parent,) = node_children(cache, cache.root_node_handle())
        (split_leaf,) = node_children(cache, split_parent)

        cache.writing_check(write_back=True)
        cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(split_leaf))
        self.assertTrue(cache.tree_core.is_backuped(split_leaf))
        self.assertTrue(cache.tree_core.is_host_evictable_leaf(split_leaf))
        cache.sanity_check()

    def test_hicache_evict_to_host_updates_aux_lru(self):
        """Aux components (MAMBA / SWA) move from device LRU to host LRU on D->H eviction."""
        aux_types = [
            ct
            for ct in (ComponentType.MAMBA, ComponentType.SWA)
            if ct in self.cfg.components
        ]
        if not aux_types:
            self.skipTest("requires at least one aux component")

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)

        m = match_prefix(cache, seq)
        node = m.last_device_node

        for aux in aux_types:
            self.assertTrue(cache.tree_core.is_node_in_device_lru(node, aux))
            self.assertFalse(cache.tree_core.is_node_in_host_lru(node, aux))

        self._simulate_backup(cache, node)
        cache.evict(EvictParams(num_tokens=len(seq)))

        for aux in aux_types:
            self.assertFalse(cache.tree_core.is_node_in_device_lru(node, aux))
            if node_host_value(cache, node, aux) is not None:
                self.assertTrue(cache.tree_core.is_node_in_host_lru(node, aux))
        cache.sanity_check()

    @requires_mamba
    def test_hicache_write_back_internal_mamba_evict_demotes_state(self):
        """write_back: an internal node's tombstoned mamba state is demoted
        to host, keeping the node a valid match boundary so the KV beneath
        it stays servable."""
        cache, req_to_token_pool, seq_a = self._build_internal_mamba_fixture(
            "write_back"
        )

        result = cache.evict(EvictParams(num_tokens=0, mamba_num=10))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)

        m = match_prefix(cache, seq_a)
        node = m.best_match_node
        self.assertNotEqual(
            node,
            cache.root_node_handle(),
            "host-demoted mamba must keep the node a valid match boundary",
        )
        self.assertIsNone(
            node_device_value(cache, node, ComponentType.MAMBA),
            "device state was tombstoned",
        )
        self.assertIsNotNone(
            node_host_value(cache, node, ComponentType.MAMBA),
            "state demoted to host, not dropped",
        )
        # SWA can split off device-only Full ancestors without Mamba states.
        # They count toward the Full hit but not the host-hit budget; the
        # scheduler collects their resident KV after reviving the state below.
        self.assertEqual(
            m.full_kv_hit_length,
            len(seq_a),
            "the full prefix remains cached",
        )
        self.assertGreaterEqual(
            m.mamba_host_hit_length, 1, "load-back armed for the host state"
        )

        # Scheduler-side serve path: init_load_back revives the state and
        # returns the node's still-device-resident KV.
        req = self._make_req(req_to_token_pool)
        self._apply_match_to_req(req, m)
        loaded_len, last_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=m.best_match_node,
                host_hit_length=m.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(last_node, node)
        self.assertEqual(m.device_prefix_len + loaded_len, len(seq_a))
        self.assertIsNotNone(
            node_device_value(cache, node, ComponentType.MAMBA),
            "mamba state revived on device",
        )
        self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)
        cache.sanity_check()

    @requires_swa_and_mamba
    def test_hicache_internal_mamba_backup_waits_for_pending_swa(self):
        page = self.cfg.page_size
        cache, allocator, req_pool = build_fixture(
            replace(self.cfg, sliding_window_size=3 * page)
        )
        self.assertEqual(cache._tree_core_backend, selected_tree_core_test_backend())

        def insert(length):
            return self._insert(
                cache, allocator, req_pool, list(range(length))
            ).last_device_node

        nodes = [insert(size) for size in range(page, 5 * page, page)]
        grandparent, _, anchor, leaf = nodes
        self._init_hicache(cache, write_policy="write_back")
        cache.evict(EvictParams(mamba_num=3))
        cache.evict_host(page, ComponentType.FULL)
        cache.evict(EvictParams(swa_num_tokens=3 * page))
        cache.evict_host(3 * page, ComponentType.SWA)

        # The backed parent stops the Full backup chain, while the SWA window
        # reaches the unbacked grandparent and marks it under the anchor's ack.
        insert(3 * page)
        # Queue the asynchronous window backup explicitly: write-back insertion
        # no longer publishes it, but Mamba eviction must still respect its ack.
        cache._execute_and_commit_kv_backup(BackupKV(node_ids=[anchor]))
        self.assertEqual(
            cache.ongoing_write_through[anchor].publish_node_ids, nodes[:3]
        )
        insert(page)
        self.assertFalse(cache.tree_core.is_backuped(grandparent))
        self.assertIn(
            ComponentType.MAMBA, cache.tree_core.build_backup_spec(grandparent)[1]
        )
        leaf_lock = cache.inc_lock_ref(leaf).to_dec_params()
        try:
            result = cache.evict(EvictParams(mamba_num=1))
            self.assertEqual(result.mamba_num_evicted, 1)
            self.assertIsNone(
                node_device_value(cache, grandparent, ComponentType.MAMBA)
            )
            self.assertIsNotNone(
                node_host_value(cache, grandparent, ComponentType.MAMBA)
            )
            self.assertFalse(cache.ongoing_write_through)
        finally:
            cache.dec_lock_ref(leaf, leaf_lock)
        cache.sanity_check()

    @requires_mamba
    def test_hicache_write_through_internal_mamba_evict_keeps_drop(self):
        """Non-write_back policies keep the legacy tombstone-and-drop."""
        cache, _, seq_a = self._build_internal_mamba_fixture("write_through")

        cache.evict(EvictParams(num_tokens=0, mamba_num=10))

        m = match_prefix(cache, seq_a)
        self.assertEqual(
            m.device_prefix_len,
            0,
            "write_through keeps the legacy drop: frontier capped at root",
        )
        cache.sanity_check()

    @requires_swa
    def test_hicache_write_back_internal_swa_evict_demotes_kv(self):
        """write_back: an internal node's tombstoned SWA KV is demoted to
        host, keeping the node a valid match boundary so the KV beneath it
        stays servable instead of losing a window of matchable prefix."""
        cache, req_to_token_pool, seq_a, seq_b = self._build_internal_swa_fixture(
            "write_back"
        )
        self.assertEqual(cache._tree_core_backend, selected_tree_core_test_backend())

        result = cache.evict(EvictParams(num_tokens=0, swa_num_tokens=len(seq_b)))
        self.assertGreaterEqual(result.swa_num_tokens_evicted, len(seq_a))

        m = match_prefix(cache, seq_a)
        node = m.best_match_node
        self.assertNotEqual(
            node,
            cache.root_node_handle(),
            "host-demoted SWA must keep the node a valid match boundary",
        )
        self.assertIsNone(
            node_device_value(cache, node, ComponentType.SWA),
            "device SWA was tombstoned",
        )
        self.assertIsNotNone(
            node_host_value(cache, node, ComponentType.SWA),
            "SWA KV demoted to host, not dropped",
        )
        self.assertEqual(
            m.device_prefix_len + m.host_hit_length,
            len(seq_a),
            "the full prefix stays servable",
        )
        self.assertGreaterEqual(
            m.swa_host_hit_length, 1, "load-back armed for the host SWA KV"
        )

        # Scheduler-side serve path: init_load_back revives the SWA KV and
        # the node's still-resident full KV serves the prefix.
        req = self._make_req(req_to_token_pool)
        self._apply_match_to_req(req, m)
        loaded_len, last_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=m.best_match_node,
                host_hit_length=m.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(last_node, node)
        self.assertEqual(m.device_prefix_len + loaded_len, len(seq_a))
        self.assertIsNotNone(
            node_device_value(cache, node, ComponentType.SWA),
            "SWA KV revived on device",
        )
        self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)
        cache.sanity_check()

    def test_match_prefix_best_and_device_node_without_hicache(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size
        min_tokens = 2 * ps
        if self.cfg.has_swa:
            min_tokens = max(min_tokens, self.cfg.sliding_window_size + ps)
        seq = self._make_seq(1, (min_tokens + ps - 1) // ps)
        self._insert(cache, allocator, req_to_token_pool, seq)

        result = match_prefix(cache, seq)

        self.assertEqual(result.device_prefix_len, len(seq))
        self.assertEqual(result.best_match_node, result.last_device_node)
        self.assertEqual(result.last_host_node, result.last_device_node)
        self.assertEqual(result.host_hit_length, 0)

    @requires_swa_only
    @requires(
        lambda cfg: cfg.sliding_window_size > cfg.page_size,
        "the window must reach past the leaf's own page",
    )
    def test_swa_backup_collector_is_shared_by_both_call_sites(self):
        """needs_incremental_backup and the BACKUP_HOST transfer read one
        collector: cache mode walks the window past a host-backed target to
        its device-only ancestor, buffer mode stages the target alone."""
        # This test exercises the Python component's collector directly.
        cache, allocator, req_to_token_pool = self._build_hicache_fixture(
            tree_core_backend="python"
        )
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 2)
        if len(chain) < 2:
            self.skipTest("chain too short")
        parent, leaf = chain[-2], chain[-1]
        swa = cache.components[ComponentType.SWA]
        leaf_node = cache.tree_core.node_by_id(leaf)
        cache.tree_core.set_component_host_value_raw(
            leaf,
            ComponentType.SWA,
            node_device_value(cache, leaf, ComponentType.SWA).clone(),
        )

        self.assertTrue(swa.needs_incremental_backup(leaf_node))
        xfer = cache.tree_core.build_hicache_transfers(
            ComponentType.SWA, leaf, CacheTransferPhase.BACKUP_HOST
        )[0]
        self.assertEqual(xfer.nodes_to_load, [parent])

        cache.tree_core.set_host_memory_buffer_only()
        self.assertTrue(swa.needs_incremental_backup(leaf_node))
        xfer = cache.tree_core.build_hicache_transfers(
            ComponentType.SWA, leaf, CacheTransferPhase.BACKUP_HOST
        )[0]
        self.assertEqual(xfer.nodes_to_load, [leaf])

    def test_evict_host_drains_freed_host_values_to_the_pools(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        host_pool_attr = {
            ComponentType.FULL: "full_kv_pool_host",
            ComponentType.SWA: "swa_kv_pool_host",
            ComponentType.MAMBA: "mamba_pool_host",
        }
        for ct in cache.tree_components:
            # set by the hicache assembler for every enabled component
            pool = getattr(cache, host_pool_attr[ct])
            if pool is None:
                continue
            available_before = pool.available_size()
            evicted = cache.evict_host(pool.size, ct)
            if evicted == 0:
                continue
            # the drain must return every freed host value to the pool
            self.assertGreater(pool.available_size(), available_before)

    @requires_full_mamba_ps1
    def test_hicache_mamba_host_best_match_keeps_device_anchor(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        parent = chain[-2]
        tokens = self._match_tokens_for_chain(cache, chain)

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        result = match_prefix(cache, tokens)

        self.assertEqual(result.best_match_node, leaf)
        self.assertEqual(result.last_device_node, parent)
        self.assertEqual(
            result.device_prefix_len, len(tokens) - node_key_length(cache, leaf)
        )
        self.assertEqual(result.host_hit_length, node_key_length(cache, leaf))

    @requires_full_mamba_ps1
    def test_mamba_has_host_value_only_predicate(self):
        """Needs a slot only when mamba is host-only (device evicted, host backed up)."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if not chain:
            self.skipTest("chain too short")
        node = chain[-1]
        dev = torch.tensor([0], dtype=torch.int64)
        host = torch.tensor([0], dtype=torch.int64)

        # host-only: device evicted, host backup present -> restore needs a slot
        cache.tree_core.set_component_device_value_raw(node, ComponentType.MAMBA, None)
        cache.tree_core.set_component_host_value_raw(node, ComponentType.MAMBA, host)
        self.assertTrue(
            cache.tree_core.component_has_host_value_only(node, ComponentType.MAMBA)
        )
        # device + host: device value present -> no restore, no slot (the D+H bug case)
        cache.tree_core.set_component_device_value_raw(node, ComponentType.MAMBA, dev)
        self.assertFalse(
            cache.tree_core.component_has_host_value_only(node, ComponentType.MAMBA)
        )
        # device-only / neither: nothing to restore
        cache.tree_core.set_component_host_value_raw(node, ComponentType.MAMBA, None)
        self.assertFalse(
            cache.tree_core.component_has_host_value_only(node, ComponentType.MAMBA)
        )
        cache.tree_core.set_component_device_value_raw(node, ComponentType.MAMBA, None)
        self.assertFalse(
            cache.tree_core.component_has_host_value_only(node, ComponentType.MAMBA)
        )

    @requires_full_swa_ps1
    def test_hicache_swa_host_best_match_keeps_device_anchor(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        parent = chain[-2]
        tokens = self._match_tokens_for_chain(cache, chain)

        self._backup_node(cache, leaf)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, leaf)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(leaf))

        result = match_prefix(cache, tokens)

        self.assertEqual(result.best_match_node, leaf)
        self.assertEqual(result.last_device_node, parent)
        self.assertEqual(
            result.device_prefix_len, len(tokens) - node_key_length(cache, leaf)
        )
        self.assertEqual(result.host_hit_length, node_key_length(cache, leaf))
        self.assertEqual(result.swa_host_hit_length, node_key_length(cache, leaf))

        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        parent = chain[-2]
        tokens = self._match_tokens_for_chain(cache, chain)

        self._set_aux_host_tombstone(cache, leaf, ComponentType.SWA)

        result = match_prefix(cache, tokens)

        self.assertEqual(result.best_match_node, leaf)
        self.assertEqual(result.last_device_node, parent)
        self.assertEqual(
            result.device_prefix_len, len(tokens) - node_key_length(cache, leaf)
        )
        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.swa_host_hit_length, node_key_length(cache, leaf))

    @requires_full_swa_ps1_window4
    def test_hicache_swa_backup_window_stops_at_pending_ancestor(self):

        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain collapsed below the pending ancestor test")

        c = chain[-1]
        c_swa = node_device_value(cache, c, ComponentType.SWA).clone()

        # First transfer: publish Full for C only. C's SWA is a device tombstone
        # (decode-evicted, never backed up), so the write-through ack stays
        # pending on a node the SWA backup window has nothing to send for.
        cache.tree_core.set_component_device_value_raw(c, ComponentType.SWA, None)
        if cache.tree_core.is_node_in_device_lru(c, ComponentType.SWA):
            cache.tree_core.remove_node_from_device_lru(c, ComponentType.SWA)
        cache.tree_core.set_component_evictable_size(
            ComponentType.SWA,
            cache.tree_core.component_evictable_size(ComponentType.SWA) - len(c_swa),
        )
        self.assertGreater(
            cache._execute_and_commit_kv_backup(BackupKV(node_ids=[c])),
            0,
        )
        self.assertEqual(cache.tree_core.get_write_through_pending_id(c), c)
        self.assertIsNotNone(node_host_value(cache, c, ComponentType.FULL))
        self.assertIsNone(node_host_value(cache, c, ComponentType.SWA))

        # SWA is reconstructed on device before the first ack, the way a
        # load-back commit stores it: under the pending segment lock the value
        # counts as protected until the ack releases it. The next incremental
        # SWA backup must treat C as the boundary and back up only the newly
        # inserted descendant.
        cache.tree_core.set_component_device_value(c, ComponentType.SWA, c_swa)
        tokens = self._match_tokens_for_chain(cache, chain)
        next_tokens = tokens + self._make_seq(9000, 1)
        cache.write_through_threshold = 1

        self._insert(cache, allocator, req_to_token_pool, next_tokens)

        d = match_prefix(cache, next_tokens).last_device_node
        self.assertEqual(cache.tree_core.get_write_through_pending_id(c), c)
        self.assertEqual(cache.tree_core.get_write_through_pending_id(d), d)
        self.assertIsNone(node_host_value(cache, c, ComponentType.SWA))
        self.assertIsNotNone(node_host_value(cache, d, ComponentType.SWA))

        cache.writing_check(write_back=True)
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(c))
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(d))
        cache.sanity_check()

    @requires_swa
    @requires(
        lambda cfg: not cfg.has_mamba,
        "SWA-only path keeps the chain construction simple",
    )
    def test_hicache_swa_finalize_match_result(self):
        """finalize_match_result accumulates host_value lengths of SWA tombstones
        within the trailing sliding window into ``swa_host_hit_length``. Out-of-window
        tombstones and chains fully on device must leave ``swa_host_hit_length`` at 0.
        ``host_hit_length`` is Full-KV only and is never written by SWA.
        """

        cache, _, _, chain, window_pages = self._swa_finalize_setup()
        leaf = chain[-1]
        ps = self.cfg.page_size

        cases = [
            ("all_on_device", None, 0),
            ("tombstone_in_window", chain[-window_pages], ps),
            ("tombstone_outside_window", chain[-(window_pages + 1)], 0),
        ]
        for name, victim, expected in cases:
            with self.subTest(name):
                # Reset SWA state for each subcase.
                for n in chain:
                    device_value = node_device_value(cache, n, ComponentType.SWA)
                    host_value = node_host_value(cache, n, ComponentType.SWA)
                    if device_value is None and host_value is not None:
                        cache.tree_core.set_component_device_value_raw(
                            n, ComponentType.SWA, host_value.clone()
                        )
                if victim is not None:
                    cache.tree_core.set_component_device_value_raw(
                        victim, ComponentType.SWA, None
                    )

                result = MatchResult(
                    device_prefix_len=0,
                    last_device_node=leaf,
                    last_host_node=leaf,
                    best_match_node=leaf,
                    host_hit_length=0,
                    cache_protected_len=7,
                    cache_actions=(FreeDeviceKV(indices=[]),),
                )
                result = cache.tree_core.finalize_component_match_result(
                    ComponentType.SWA,
                    result=result,
                    params=MatchPrefixParams(
                        key=RadixKey(array("q", self._make_seq(1, 1)))
                    ),
                    value_chunks=[],
                    best_value_len=0,
                )
                self.assertEqual(result.host_hit_length, 0)
                self.assertEqual(result.swa_host_hit_length, expected)
                self.assertEqual(result.cache_protected_len, 7)
                self.assertEqual(result.cache_actions, (FreeDeviceKV(indices=[]),))

    def test_hicache_swa_finalize_anchored_on_best_match_node(self):
        cache, _, _, y, x, _ = self._swa_anchor_setup()
        ps = self.cfg.page_size

        base = MatchResult(
            device_prefix_len=0,
            last_device_node=x,
            last_host_node=y,
            best_match_node=x,
            host_hit_length=0,
        )
        result = cache.tree_core.finalize_component_match_result(
            ComponentType.SWA,
            result=base,
            params=MatchPrefixParams(key=RadixKey(array("q", self._make_seq(1, 1)))),
            value_chunks=[],
            best_value_len=0,
        )
        # SWA host hit goes to swa_host_hit_length; host_hit_length stays at 0.
        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.swa_host_hit_length, ps)

    @requires_swa
    @requires(
        lambda cfg: not cfg.has_mamba,
        "SWA-only path keeps the chain construction simple",
    )
    def test_hicache_swa_temp_lock_does_not_release_restored_tombstone(self):
        """A temporary scheduler lock that skipped a SWA tombstone must not
        release later load-back/request locks after the tombstone is restored.
        """

        cache, allocator, _, chain, _ = self._swa_finalize_setup()
        leaf = chain[-1]
        tombstone = leaf
        old_swa = node_device_value(cache, tombstone, ComponentType.SWA)
        self.assertIsNotNone(old_swa)

        cache.tree_core.set_component_device_value_raw(
            tombstone, ComponentType.SWA, None
        )
        cache.tree_core.remove_node_from_device_lru(tombstone, ComponentType.SWA)
        cache.tree_core.insert_node_into_host_lru(tombstone, ComponentType.SWA)
        cache.tree_core.set_component_evictable_size(
            ComponentType.SWA,
            cache.tree_core.component_evictable_size(ComponentType.SWA) - len(old_swa),
        )

        temp_lock = cache.inc_lock_ref(leaf)
        self.assertEqual(node_device_lock_ref(cache, tombstone, ComponentType.SWA), 1)

        xfer = cache.tree_core.build_hicache_transfers(
            ComponentType.SWA, leaf, CacheTransferPhase.LOAD_BACK
        )[0]
        new_swa = allocator.swa_attn_allocator.alloc(int(xfer.host_indices.numel()))
        self.assertIsNotNone(new_swa)
        xfer.device_indices = new_swa
        load_actions = []
        cache.tree_core.commit_hicache_transfers(
            leaf,
            CacheTransferPhase.LOAD_BACK,
            {ComponentType.SWA: [xfer]},
            cache_actions=load_actions,
        )
        cache._apply_cache_actions(load_actions)

        load_back_lock = cache.inc_lock_ref(leaf)
        request_lock = cache.inc_lock_ref(leaf)
        self.assertEqual(node_device_lock_ref(cache, tombstone, ComponentType.SWA), 3)

        cache.dec_lock_ref(leaf, temp_lock.to_dec_params())
        self.assertEqual(node_device_lock_ref(cache, tombstone, ComponentType.SWA), 2)

        cache.dec_lock_ref(leaf, load_back_lock.to_dec_params())
        cache.dec_lock_ref(leaf, request_lock.to_dec_params())
        self.assertEqual(node_device_lock_ref(cache, tombstone, ComponentType.SWA), 0)

    @requires_hicache
    def test_hicache_full_temp_lock_covers_evicted_anchor_and_mirrors_on_release(
        self,
    ):
        """Segment locks count the evicted anchor too (no skip receipts), so
        a value restored mid-hold stays correctly attributed: each release
        takes back exactly its own ref regardless of interleaved holders.
        """
        ps = self.cfg.page_size
        if 3 * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small")
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        a, y, anchor = chain
        self._simulate_backup_tree(cache)

        anchor_value = node_device_value(cache, anchor, ComponentType.FULL)
        # Simulate the anchor's FULL device eviction: drop the value and take
        # its tokens out of the evictable ledger, as a real evict would.
        cache.tree_core.set_component_device_value_raw(anchor, ComponentType.FULL, None)
        cache.tree_core.set_component_evictable_size(
            ComponentType.FULL,
            cache.tree_core.component_evictable_size(ComponentType.FULL)
            - len(anchor_value),
        )

        self.assertEqual(node_device_lock_ref(cache, anchor, ComponentType.FULL), 0)
        self.assertEqual(node_device_lock_ref(cache, y, ComponentType.FULL), 0)
        self.assertEqual(node_device_lock_ref(cache, a, ComponentType.FULL), 0)

        temp_lock = cache.inc_lock_ref(anchor)
        self.assertEqual(node_device_lock_ref(cache, anchor, ComponentType.FULL), 1)
        self.assertEqual(node_device_lock_ref(cache, y, ComponentType.FULL), 1)
        self.assertEqual(node_device_lock_ref(cache, a, ComponentType.FULL), 1)

        # Restore the value mid-hold: a value materialized under lock is
        # protected until the last release, exactly as a load-back credits it.
        cache.tree_core.set_component_device_value_raw(
            anchor, ComponentType.FULL, anchor_value
        )
        cache.tree_core.set_component_protected_size(
            ComponentType.FULL,
            cache.tree_core.component_protected_size(ComponentType.FULL)
            + len(anchor_value),
        )

        second_lock = cache.inc_lock_ref(anchor)
        self.assertEqual(node_device_lock_ref(cache, anchor, ComponentType.FULL), 2)
        self.assertEqual(node_device_lock_ref(cache, y, ComponentType.FULL), 2)
        self.assertEqual(node_device_lock_ref(cache, a, ComponentType.FULL), 2)

        cache.dec_lock_ref(anchor, temp_lock.to_dec_params())
        self.assertEqual(node_device_lock_ref(cache, anchor, ComponentType.FULL), 1)
        self.assertEqual(node_device_lock_ref(cache, y, ComponentType.FULL), 1)
        self.assertEqual(node_device_lock_ref(cache, a, ComponentType.FULL), 1)

        cache.dec_lock_ref(anchor, second_lock.to_dec_params())
        self.assertEqual(node_device_lock_ref(cache, anchor, ComponentType.FULL), 0)
        self.assertEqual(node_device_lock_ref(cache, y, ComponentType.FULL), 0)
        self.assertEqual(node_device_lock_ref(cache, a, ComponentType.FULL), 0)

    @requires_mamba
    @requires(
        lambda cfg: not cfg.has_swa,
        "Mamba-only path keeps the chain construction simple",
    )
    def test_hicache_mamba_temp_lock_does_not_release_restored_tombstone(self):
        """A temporary scheduler lock that skipped a Mamba tombstone must not
        release later load-back/request locks after the tombstone is restored.
        """

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)
        m = match_prefix(cache, seq)
        node = m.last_device_node
        old_mamba = node_device_value(cache, node, ComponentType.MAMBA)
        self.assertIsNotNone(old_mamba)
        self._simulate_backup(cache, node)

        cache.tree_core.set_component_device_value_raw(node, ComponentType.MAMBA, None)
        cache.tree_core.remove_node_from_device_lru(node, ComponentType.MAMBA)
        cache.tree_core.insert_node_into_host_lru(node, ComponentType.MAMBA)
        cache.tree_core.set_component_evictable_size(
            ComponentType.MAMBA,
            cache.tree_core.component_evictable_size(ComponentType.MAMBA)
            - len(old_mamba),
        )

        temp_lock = cache.inc_lock_ref(node)
        self.assertEqual(node_device_lock_ref(cache, node, ComponentType.MAMBA), 1)

        xfer = cache.tree_core.build_hicache_transfers(
            ComponentType.MAMBA, node, CacheTransferPhase.LOAD_BACK
        )[0]
        new_mamba = req_to_token_pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(new_mamba)
        xfer.device_indices = new_mamba
        cache.tree_core.commit_hicache_transfers(
            node,
            CacheTransferPhase.LOAD_BACK,
            {ComponentType.MAMBA: [xfer]},
            cache_actions=[],
        )

        load_back_lock = cache.inc_lock_ref(node)
        request_lock = cache.inc_lock_ref(node)
        self.assertEqual(node_device_lock_ref(cache, node, ComponentType.MAMBA), 3)

        cache.dec_lock_ref(node, temp_lock.to_dec_params())
        self.assertEqual(node_device_lock_ref(cache, node, ComponentType.MAMBA), 2)

        cache.dec_lock_ref(node, load_back_lock.to_dec_params())
        cache.dec_lock_ref(node, request_lock.to_dec_params())
        self.assertEqual(node_device_lock_ref(cache, node, ComponentType.MAMBA), 0)
        # The commit ran under held locks: the restored value must have been
        # credited to protected, or the ledger drifts on the final release.
        cache.sanity_check()

    @requires_hicache
    def test_hicache_write_back_leaf_backup(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")

        base = self._make_seq(1, 2)
        leaf_seq = base + self._make_seq(500, 2)
        self._insert(cache, allocator, req_to_token_pool, base)
        self._insert(cache, allocator, req_to_token_pool, leaf_seq)

        m = match_prefix(cache, leaf_seq)
        leaf = m.last_device_node
        parent = cache.tree_core.get_parent_node_id(leaf)
        self.assertNotEqual(parent, cache.root_node_handle())

        self.assertFalse(cache.tree_core.is_backuped(leaf))
        self.assertFalse(cache.tree_core.is_backuped(parent))

        lr = cache.inc_lock_ref(parent)
        try:
            evict_tokens = len(leaf_seq) - len(base)
            cache.evict(EvictParams(num_tokens=evict_tokens))
        finally:
            cache.dec_lock_ref(
                parent,
                lr.to_dec_params(),
            )

        self.assertTrue(
            cache.tree_core.is_full_device_evicted(leaf),
            "leaf should be demoted to host",
        )
        self.assertTrue(
            cache.tree_core.is_backuped(leaf),
            "write_back must back up the leaf on eviction",
        )
        self.assertFalse(
            cache.tree_core.is_backuped(parent),
            "parent must NOT be backed up under write_back",
        )

        cache.sanity_check()

    @requires(
        lambda cfg: not cfg.has_swa,
        "SWA boundary splits deepen the chain non-deterministically",
    )
    def test_build_backup_kv_action_orders_ancestors_first(self):
        """write-through backs up unbacked ancestors parent-first; write-back only the leaf."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()

        top_seq = self._make_seq(1, 1)
        mid_seq = top_seq + self._make_seq(1000, 1)
        leaf_seq = mid_seq + self._make_seq(2000, 1)
        self._insert(cache, allocator, req_to_token_pool, top_seq)
        self._insert(cache, allocator, req_to_token_pool, mid_seq)
        self._insert(cache, allocator, req_to_token_pool, leaf_seq)

        m = match_prefix(cache, leaf_seq)
        leaf = m.last_device_node
        mid = cache.tree_core.get_parent_node_id(leaf)
        top = cache.tree_core.get_parent_node_id(mid)
        self.assertNotEqual(top, cache.root_node_handle())
        self.assertFalse(cache.tree_core.is_backuped(leaf))
        self.assertFalse(cache.tree_core.is_backuped(mid))
        self.assertFalse(cache.tree_core.is_backuped(top))

        # write-through: every unbacked ancestor, top-most first, child last
        write_through = cache.tree_core.build_backup_node_ids(leaf, write_back=False)
        self.assertEqual(write_through, [top, mid, leaf])

        # write-back: only the eviction victim, even with unbacked ancestors
        write_back = cache.tree_core.build_backup_node_ids(leaf, write_back=True)
        self.assertEqual(write_back, [leaf])

        cache.sanity_check()

    @requires_swa
    @requires_hicache
    def test_write_back_defers_swa_window_backup_until_eviction(self):
        """Reinserting backed Full nodes must not publish freshly recovered SWA."""
        ps = self.cfg.page_size
        if self.cfg.sliding_window_size <= 2 * ps:
            self.skipTest("window must span past the leaf and its parent to reach c")

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")

        # c(Full unbacked) -> b(Full backed) -> t1(Full backed)
        #                  -> d(Full unbacked) -> t2(Full backed)
        # Every node holds a device-only SWA value inside the window.
        c_seq = self._make_seq(1, 2)
        t1_seq = c_seq + self._make_seq(1000, 1) + self._make_seq(2000, 1)
        t2_seq = c_seq + self._make_seq(3000, 1) + self._make_seq(4000, 1)
        for seq in (t1_seq, t2_seq, t1_seq[:-ps], t2_seq[:-ps]):
            self._insert(cache, allocator, req_to_token_pool, seq)

        def leaf(seq):
            return match_prefix(cache, seq).last_device_node

        c, b, t1 = self._path_chain(cache, leaf(t1_seq))
        _, d, t2 = self._path_chain(cache, leaf(t2_seq))
        for node in (b, t1, t2):
            cache.tree_core.set_component_host_value_raw(
                node,
                ComponentType.FULL,
                node_device_value(cache, node, ComponentType.FULL).clone(),
            )
        for node in (c, b, t1, d, t2):
            self.assertIsNotNone(node_device_value(cache, node, ComponentType.SWA))
            self.assertIsNone(node_host_value(cache, node, ComponentType.SWA))

        # Two requests in one batch finish at t1 and t2; neither ack drains.
        with mock.patch.object(
            cache,
            "_execute_and_commit_kv_backup",
            wraps=cache._execute_and_commit_kv_backup,
        ) as backups:
            for seq in (t1_seq, t2_seq):
                self._insert(cache, allocator, req_to_token_pool, seq)
            backups.assert_not_called()

        for node in (c, b, t1, d, t2):
            self.assertIsNone(cache.tree_core.get_write_through_pending_id(node))
            self.assertIsNone(node_host_value(cache, node, ComponentType.SWA))
        self.assertEqual(cache.cache_controller.write_queue, [])
        # Write-back defers the unbacked Full prefix to eviction.
        self.assertFalse(cache.tree_core.is_backuped(c))
        self.assertFalse(cache.tree_core.is_backuped(d))


define_config_test_classes(UnifiedRadixCacheHiCacheSuite, globals())


if __name__ == "__main__":
    unittest.main()

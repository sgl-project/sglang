import unittest
from array import array
from dataclasses import replace
from unittest import mock

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
)
from sglang.srt.mem_cache.unified_cache.components.base import (
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
        """Write-back eviction keeps freeing device KV when the host pool is full."""
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

        m = match_prefix(cache, seq)
        self.assertEqual(m.device_prefix_len, 0)
        self.assertEqual(m.host_hit_length, 0)
        cache.sanity_check()

    @requires_hicache
    def test_hicache_write_back_drop_respects_pins_then_frees_subtree(self):
        """The host-pressure drop fallback declines while any node in the unbacked
        subtree is host-pinned, then reclaims the whole subtree once unpinned."""
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
            parent_lock_params = cache.inc_host_lock_ref(parent).to_dec_params()
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
            self.assertEqual(result.num_tokens_evicted, 0)
            cache.dec_host_lock_ref(parent, parent_lock_params)

            child_lock_params = cache.inc_host_lock_ref(child).to_dec_params()
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
            self.assertEqual(result.num_tokens_evicted, 0)
            m = match_prefix(cache, parent_seq)
            self.assertEqual(m.device_prefix_len, len(parent_seq))
            cache.dec_host_lock_ref(child, child_lock_params)

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

        # The prefix must hold a host lock, else these slots are reclaimed while
        # the storage thread still reads them.
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

        # Diverging inside the locked SWA window moves the lock boundary and copied
        # host refs to the new prefix; neither half may be reclaimed.
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
        """The tree op defers a write-back victim's demote, so a failed backup
        leaves the node recoverable on device."""
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
    def test_hicache_partial_match_splits_evicted_backed_up_node(self):
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
            prefix_tokens: list[int] = []
            chain = []
            cur = split_parent
            while cur != cache.root_node_handle():
                chain.append(cur)
                cur = node_parent(cache, cur)
            for n in reversed(chain):
                prefix_tokens.extend(node_token_ids(cache, n))
            self.assertEqual(prefix_tokens, expected_prefix)
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

        self.assertTrue(cache.tree_core.is_backuped(node))
        self.assertIsNone(node_device_value(cache, node, ComponentType.FULL))
        self.assertIsNotNone(node_host_value(cache, node, ComponentType.FULL))
        self.assertFalse(cache.tree_core.is_device_evictable_leaf(node))
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
        """write_back: an internal node's tombstoned mamba state is demoted to host,
        keeping the node a valid match boundary."""
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
        # Queue the asynchronous window backup explicitly; Mamba eviction must
        # respect its ack.
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
        """Non-write_back policies tombstone and drop."""
        cache, _, seq_a = self._build_internal_mamba_fixture("write_through")

        cache.evict(EvictParams(num_tokens=0, mamba_num=10))

        m = match_prefix(cache, seq_a)
        self.assertEqual(
            m.device_prefix_len,
            0,
            "write_through keeps the drop: frontier capped at root",
        )
        cache.sanity_check()

    @requires_swa
    def test_hicache_write_back_internal_swa_evict_demotes_kv(self):
        """write_back: an internal node's tombstoned SWA KV is demoted to host,
        keeping the node a valid match boundary."""
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
            pool = getattr(cache, host_pool_attr[ct])
            if pool is None:
                continue
            available_before = pool.available_size()
            evicted = cache.evict_host(pool.size, ct)
            if evicted == 0:
                continue
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
        tokens = self._match_tokens_for_chain(cache, chain)
        # C's SWA is decode-evicted before it is ever backed up, so the
        # write-through of C publishes Full only and its ack stays pending on a
        # node the SWA backup window has nothing to send for.
        self._evict_device_component(cache, c, ComponentType.SWA)
        self.assertGreater(
            cache._execute_and_commit_kv_backup(BackupKV(node_ids=[c])),
            0,
        )
        self.assertEqual(cache.tree_core.get_write_through_pending_id(c), c)
        self.assertIsNotNone(node_host_value(cache, c, ComponentType.FULL))
        self.assertIsNone(node_host_value(cache, c, ComponentType.SWA))
        # A request re-inserting C's tokens recovers its SWA on device before
        # the first ack. The next incremental SWA backup must treat C as the
        # boundary and back up only the newly inserted descendant.
        self._insert(cache, allocator, req_to_token_pool, tokens)
        self.assertIsNotNone(node_device_value(cache, c, ComponentType.SWA))
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
    def test_hicache_swa_host_hit_counts_tombstones_inside_the_window(self):
        """A host-backed SWA tombstone inside the trailing sliding window stops
        the device match before it: the Full-backed tail from the tombstone on
        is a host hit, and the tombstone's own SWA counts in
        ``swa_host_hit_length``."""
        ps = self.cfg.page_size
        window_pages = (self.cfg.sliding_window_size + ps - 1) // ps
        for name, victim in (
            ("all_on_device", None),
            ("tombstone_in_window", -window_pages),
            ("tombstone_outside_window", -(window_pages + 1)),
        ):
            with self.subTest(name):
                cache, allocator, req_to_token_pool = self._build_hicache_fixture()
                chain = self._build_chain_pages(
                    cache, allocator, req_to_token_pool, window_pages + 2
                )
                if len(chain) <= window_pages:
                    self.skipTest("chain collapsed below the window length")
                tokens = self._match_tokens_for_chain(cache, chain)
                self._backup_node(cache, chain[-1])
                host_tail, swa_host = 0, 0
                if victim is not None:
                    self._evict_device_component(
                        cache, chain[victim], ComponentType.SWA
                    )
                if name == "tombstone_in_window":
                    host_tail = sum(node_key_length(cache, n) for n in chain[victim:])
                    swa_host = node_key_length(cache, chain[victim])
                result = match_prefix(cache, tokens)
                self.assertEqual(result.device_prefix_len, len(tokens) - host_tail)
                self.assertEqual(result.host_hit_length, host_tail)
                self.assertEqual(result.swa_host_hit_length, swa_host)

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
        release later load-back/request locks after the tombstone is restored."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(cache, chain)
        self._backup_node(cache, leaf)
        self.assertIsNotNone(node_host_value(cache, leaf, ComponentType.SWA))
        self._evict_device_component(cache, leaf, ComponentType.SWA)
        self.assertIsNone(node_device_value(cache, leaf, ComponentType.SWA))

        def swa_ref():
            return node_device_lock_ref(cache, leaf, ComponentType.SWA)

        temp_lock = cache.inc_lock_ref(leaf)
        self.assertEqual(swa_ref(), 1)
        self._load_back_through_admission(cache, req_to_token_pool, tokens)
        self.assertIsNotNone(node_device_value(cache, leaf, ComponentType.SWA))
        request_lock = cache.inc_lock_ref(leaf)
        self.assertEqual(swa_ref(), 3)
        cache.dec_lock_ref(leaf, temp_lock.to_dec_params())
        self.assertEqual(swa_ref(), 2)
        self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)
        cache.dec_lock_ref(leaf, request_lock.to_dec_params())
        self.assertEqual(swa_ref(), 0)
        cache.sanity_check()

    @requires_hicache
    def test_hicache_full_temp_lock_covers_evicted_anchor_and_mirrors_on_release(
        self,
    ):
        """Segment locks count the evicted anchor too (no skip receipts), so a value
        restored mid-hold stays correctly attributed."""
        ps = self.cfg.page_size
        if 3 * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small")
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        anchor = chain[-1]
        self._backup_node(cache, anchor)
        cache.evict(EvictParams(num_tokens=node_key_length(cache, anchor)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(anchor))

        def full_refs():
            return [node_device_lock_ref(cache, n, ComponentType.FULL) for n in chain]

        self.assertEqual(full_refs(), [0, 0, 0])
        temp_lock = cache.inc_lock_ref(anchor)
        self.assertEqual(full_refs(), [1, 1, 1])
        # Restore the anchor mid-hold.
        self._load_back_node(cache, anchor)
        second_lock = cache.inc_lock_ref(anchor)
        self.assertEqual(full_refs(), [2, 2, 2])
        cache.dec_lock_ref(anchor, temp_lock.to_dec_params())
        self.assertEqual(full_refs(), [1, 1, 1])
        cache.dec_lock_ref(anchor, second_lock.to_dec_params())
        self.assertEqual(full_refs(), [0, 0, 0])
        cache.sanity_check()

    @requires_mamba
    @requires(
        lambda cfg: not cfg.has_swa,
        "Mamba-only path keeps the chain construction simple",
    )
    def test_hicache_mamba_temp_lock_does_not_release_restored_tombstone(self):
        """A temporary scheduler lock that skipped a Mamba tombstone must not
        release later load-back/request locks after the tombstone is restored."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq)
        node = match_prefix(cache, seq).last_device_node
        self._backup_node(cache, node)
        self.assertIsNotNone(node_host_value(cache, node, ComponentType.MAMBA))
        self._evict_device_component(cache, node, ComponentType.MAMBA)
        self.assertIsNone(node_device_value(cache, node, ComponentType.MAMBA))

        def mamba_ref():
            return node_device_lock_ref(cache, node, ComponentType.MAMBA)

        temp_lock = cache.inc_lock_ref(node)
        self.assertEqual(mamba_ref(), 1)
        self._load_back_through_admission(cache, req_to_token_pool, seq)
        self.assertIsNotNone(node_device_value(cache, node, ComponentType.MAMBA))
        request_lock = cache.inc_lock_ref(node)
        self.assertEqual(mamba_ref(), 3)
        cache.dec_lock_ref(node, temp_lock.to_dec_params())
        self.assertEqual(mamba_ref(), 2)
        self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)
        cache.dec_lock_ref(node, request_lock.to_dec_params())
        self.assertEqual(mamba_ref(), 0)
        # The restore ran under held locks: the restored value must have been
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

        write_through = cache.tree_core.build_backup_node_ids(leaf, write_back=False)
        self.assertEqual(write_through, [top, mid, leaf])

        write_back = cache.tree_core.build_backup_node_ids(leaf, write_back=True)
        self.assertEqual(write_back, [leaf])

        cache.sanity_check()

    @requires_swa
    @requires_hicache
    def test_write_back_defers_swa_window_backup_until_eviction(self):
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
        self.assertFalse(cache.tree_core.is_backuped(c))
        self.assertFalse(cache.tree_core.is_backuped(d))


define_config_test_classes(UnifiedRadixCacheHiCacheSuite, globals())


if __name__ == "__main__":
    unittest.main()

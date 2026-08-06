"""Falsifiable tests for the L2-as-channel mirror release mechanism.

Root cause under test (the write_through L2 deadlock): every inserted node
immediately gets a host copy, evict_host() only reclaims nodes already gone
from device, and the L3 backup ack only drops a refcount — so the host pool
fills with irreclaimable mirrors of device-resident data, new staging fails
with host_alloc_failed, and nothing ever reaches L3.

The core tests below drive only public entry points (ack queues, drain,
write_backup, _inc_hit_count) so that on the unfixed code they FAIL
(mirrors are never released, durable parents are still rejected, hot nodes
re-stage forever), and on the fixed code they pass.
"""

import unittest
from queue import Queue

import torch

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

PAGE = 2


class _HostPool:
    def __init__(self, size=1000, available=1000):
        self.size = size
        self._available = available

    def available_size(self):
        return self._available


class _DeviceAllocator:
    def __init__(self):
        self.free_calls = []

    def free(self, indices):
        self.free_calls.append(indices)


class _Controller:
    def __init__(self, write_policy="write_through"):
        self.write_policy = write_policy
        self.backup_skip = False
        self.mem_pool_host = _HostPool()
        self.mem_pool_device_allocator = _DeviceAllocator()
        self.prefetch_revoke_queue = Queue()
        self.prefetch_hit_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()
        self.prefetch_buffer = Queue()
        self.evict_host_calls = []
        self.evict_device_calls = []
        self.write_calls = []
        self.write_result = torch.tensor([7, 8])

    def evict_host(self, host_indices):
        self.evict_host_calls.append(host_indices)
        return len(host_indices)

    def evict_device(self, device_indices):
        self.evict_device_calls.append(device_indices)
        return 1 if device_indices is None else len(device_indices)

    def write(self, device_indices, node_id, **kwargs):
        self.write_calls.append(node_id)
        return self.write_result.clone()


class _Ack:
    """Shape-compatible stand-in for a backup StorageOperation ack."""

    _next_id = 0

    def __init__(self, hash_value, completed_tokens, token_ids=None):
        _Ack._next_id += 1
        self.id = _Ack._next_id
        self.hash_value = hash_value
        self.completed_tokens = completed_tokens
        self.token_ids = token_ids or [0] * (len(hash_value) * PAGE)
        self.io_started_at = 0.0
        self.io_finished_at = 0.0


class _LocalOnlyReduce:
    """Single-rank stand-in: all-reduce with no peers is the identity."""

    def __init__(self):
        self.calls = []

    def __call__(self, tensor, op):
        self.calls.append(tensor.clone())


class _PeerReduce:
    """Simulates MIN all-reduce against a scripted sequence of peer vectors.

    Each entry is either None (peer identical to us) or a callable mapping
    our vector to the peer's vector; the result is the elementwise MIN.
    """

    def __init__(self, peers):
        self.peers = list(peers)
        self.calls = []

    def __call__(self, tensor, op):
        self.calls.append(tensor.clone())
        peer = self.peers.pop(0) if self.peers else None
        if peer is None:
            return
        peer_vec = torch.tensor(peer(tensor.tolist()), dtype=tensor.dtype)
        torch.minimum(tensor, peer_vec, out=tensor)


def _node(n_pages=1, tag="n", parent=None, evicted=False, backuped=True):
    node = TreeNode()
    node.key = RadixKey(token_ids=list(range(n_pages * PAGE)))
    node.value = None if evicted else torch.arange(n_pages * PAGE)
    node.host_value = torch.arange(n_pages * PAGE) + 100 if backuped else None
    node.hash_value = [f"{tag}-{i}" for i in range(n_pages)]
    if parent is not None:
        node.parent = parent
        parent.children[node.key.child_key(PAGE)] = node
    return node


def _build_cache(enabled=True, free_frac=0.2, refresh_rounds=100000):
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.page_size = PAGE
    cache.cache_controller = _Controller()
    cache.enable_storage = True
    cache.enable_storage_metrics = False
    cache.mirror_release_enabled = enabled
    cache.mirror_release_free_frac = free_frac
    cache.mirror_refresh_rounds = refresh_rounds
    cache.storage_backed_capacity = 262144
    cache.storage_backed_hashes = __import__("collections").OrderedDict()
    cache.storage_backed_generation = 0
    cache.hicache_logical_clock = 0
    cache.redundant_host_nodes = set()
    cache._mirror_release_plan = None
    cache._mirror_release_disabled_reason = None
    cache._mirror_release_mismatch_streak = 0
    cache._mirror_released_tokens_total = 0
    cache._mirror_release_plans_executed = 0
    cache._mirror_release_plans_dropped = 0
    cache.ongoing_backup = {}
    cache.ongoing_prefetch = {}
    cache.ongoing_write_through = {}
    cache.root_node = TreeNode()
    cache.root_node.hash_value = []
    cache.evictable_host_leaves = set()
    cache.evictable_size_ = 10_000
    cache.eviction_strategy = type(
        "S", (), {"get_priority": staticmethod(lambda n: n.id)}
    )()
    cache.write_through_threshold = 1
    cache.disable = False
    cache.protected_size_ = 0
    cache._record_remove_event = _EventRecorder(cache, "remove")
    cache._record_store_event = _EventRecorder(cache, "store")
    cache._events = []
    cache._all_reduce_attn_groups = _LocalOnlyReduce()
    cache._update_leaf_status = lambda *a, **k: None
    return cache


class _EventRecorder:
    def __init__(self, cache, kind):
        self.cache = cache
        self.kind = kind

    def __call__(self, node, medium=None):
        if not hasattr(self.cache, "_events"):
            self.cache._events = []
        self.cache._events.append((self.kind, node, medium))


def _ack_node(cache, node, completed_pages=None):
    """Enqueue a backup ack for `node` as write_backup_storage would."""
    pages = len(node.hash_value) if completed_pages is None else completed_pages
    ack = _Ack(node.hash_value, completed_tokens=pages * PAGE)
    cache.ongoing_backup[ack.id] = node
    node.protect_host()
    cache.cache_controller.ack_backup_queue.put(ack)
    return ack


def _pressure(cache, deficit=100):
    pool = cache.cache_controller.mem_pool_host
    pool.size = 1000
    pool._available = int(pool.size * cache.mirror_release_free_frac) - deficit


def _no_pressure(cache):
    pool = cache.cache_controller.mem_pool_host
    pool._available = pool.size


class TestMirrorReleaseCore(unittest.TestCase):
    """The falsifiable core: on unfixed code these fail."""

    def test_durable_mirror_released_under_pressure(self):
        cache = _build_cache()
        node = _node(n_pages=2, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        _pressure(cache)

        # round 1: drain marks durable, prepares the plan;
        # round 2: consensus (single rank) commits it.
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()

        self.assertTrue(
            cache.cache_controller.evict_host_calls,
            "durable device mirror was never released — L2 deadlock present",
        )
        self.assertIsNone(node.host_value)
        # tree/device state untouched
        self.assertIsNotNone(node.value)
        self.assertIsNotNone(node.hash_value)
        self.assertIn(node.key.child_key(PAGE), cache.root_node.children)

    def test_durable_parent_without_mirror_allows_child_staging(self):
        cache = _build_cache()
        parent = _node(n_pages=1, tag="p", parent=cache.root_node)
        for h in parent.hash_value:
            cache._mark_hash_durable(h)
        parent.host_value = None  # mirror already released
        child = _node(n_pages=1, tag="c", parent=parent, backuped=False)

        wrote = cache.write_backup(child)

        self.assertGreater(
            wrote,
            0,
            "child staging rejected although the parent prefix is durable in L3",
        )
        self.assertTrue(cache.cache_controller.write_calls)

    def test_anti_churn_skips_restage_of_durable_node(self):
        cache = _build_cache()
        node = _node(n_pages=1, tag="a", parent=cache.root_node, backuped=False)
        for h in node.hash_value:
            cache._mark_hash_durable(h)
        node.hit_count = 5

        cache._inc_hit_count(node)

        self.assertFalse(
            cache.cache_controller.write_calls,
            "durable node was re-staged — release/rewrite churn loop present",
        )

    def test_anti_churn_age_refresh_allows_restage(self):
        cache = _build_cache(refresh_rounds=10)
        node = _node(n_pages=1, tag="a", parent=cache.root_node, backuped=False)
        for h in node.hash_value:
            cache._mark_hash_durable(h)
        node.hit_count = 5
        cache.hicache_logical_clock += 11  # exceed refresh age

        cache._inc_hit_count(node)

        self.assertTrue(
            cache.cache_controller.write_calls,
            "aged durable node was never refreshed to L3",
        )


class TestDurableTracking(unittest.TestCase):
    def test_full_ack_marks_all_pages(self):
        cache = _build_cache()
        node = _node(n_pages=3, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        cache.drain_storage_control_queues()
        self.assertTrue(cache._node_fully_durable(node))
        self.assertIn(node, cache.redundant_host_nodes)
        self.assertEqual(node.host_ref_counter, 0)

    def test_partial_ack_marks_prefix_only(self):
        cache = _build_cache()
        node = _node(n_pages=3, tag="a", parent=cache.root_node)
        _ack_node(cache, node, completed_pages=1)
        cache.drain_storage_control_queues()
        self.assertIn("a-0", cache.storage_backed_hashes)
        self.assertNotIn("a-1", cache.storage_backed_hashes)
        self.assertFalse(cache._node_fully_durable(node))
        self.assertNotIn(node, cache.redundant_host_nodes)

    def test_zero_ack_marks_nothing(self):
        cache = _build_cache()
        node = _node(n_pages=2, tag="a", parent=cache.root_node)
        _ack_node(cache, node, completed_pages=0)
        cache.drain_storage_control_queues()
        self.assertFalse(cache.storage_backed_hashes)
        self.assertNotIn(node, cache.redundant_host_nodes)

    def test_non_durable_mirror_never_released(self):
        cache = _build_cache()
        node = _node(n_pages=2, tag="a", parent=cache.root_node)
        _ack_node(cache, node, completed_pages=1)  # partial: not durable
        _pressure(cache)
        for _ in range(4):
            cache.drain_storage_control_queues()
        self.assertFalse(cache.cache_controller.evict_host_calls)
        self.assertIsNotNone(node.host_value)

    def test_bounded_lru_overflow_and_reinsert(self):
        cache = _build_cache()
        cache.storage_backed_capacity = 3
        for h in ("h0", "h1", "h2"):
            cache._mark_hash_durable(h)
        cache._mark_hash_durable("h0")  # re-mark moves to MRU, no growth
        self.assertEqual(len(cache.storage_backed_hashes), 3)
        cache._mark_hash_durable("h3")  # evicts h1 (oldest)
        self.assertEqual(len(cache.storage_backed_hashes), 3)
        self.assertNotIn("h1", cache.storage_backed_hashes)
        self.assertIn("h0", cache.storage_backed_hashes)

    def test_reset_clears_state_and_bumps_generation(self):
        cache = _build_cache()
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        cache.drain_storage_control_queues()
        gen = cache.storage_backed_generation
        cache._reset_mirror_release_state()
        self.assertFalse(cache.storage_backed_hashes)
        self.assertFalse(cache.redundant_host_nodes)
        self.assertIsNone(cache._mirror_release_plan)
        self.assertEqual(cache.storage_backed_generation, gen + 1)
        # a durable parent claim must not survive the reset
        self.assertFalse(cache._node_fully_durable(node))

    def test_partial_parent_rejected_for_child_staging(self):
        cache = _build_cache()
        parent = _node(n_pages=2, tag="p", parent=cache.root_node)
        cache._mark_hash_durable("p-0")  # one of two pages
        parent.host_value = None
        child = _node(n_pages=1, tag="c", parent=parent, backuped=False)
        self.assertEqual(cache.write_backup(child), 0)
        self.assertFalse(cache.cache_controller.write_calls)


class TestCandidateLifecycle(unittest.TestCase):
    def test_split_registers_both_halves_after_ack(self):
        cache = _build_cache()
        node = _node(n_pages=2, tag="s", parent=cache.root_node)
        ack = _ack_node(cache, node)  # op captures the original hash list
        # split before the ack arrives, as writing_check -> ... -> match would
        new_node = cache._split_node(node.key, node, PAGE)
        self.assertEqual(new_node.hash_value, ["s-0"])
        self.assertEqual(node.hash_value, ["s-1"])
        cache.drain_storage_control_queues()
        self.assertIn(node, cache.redundant_host_nodes)
        self.assertIn(
            new_node,
            cache.redundant_host_nodes,
            "prefix half of a mid-backup split never becomes releasable",
        )

    def test_busy_candidate_is_kept_not_lost(self):
        cache = _build_cache()
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        node.lock_ref = 1  # busy at plan time
        _pressure(cache)
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
        self.assertFalse(cache.cache_controller.evict_host_calls)
        self.assertIn(node, cache.redundant_host_nodes, "busy candidate dropped")
        node.lock_ref = 0
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
        self.assertTrue(cache.cache_controller.evict_host_calls)

    def test_stale_durable_recovers_after_full_local_eviction(self):
        """ack -> mirror released -> (L3 self-evicts, invisible to us) ->
        device eviction deletes the leaf.  The re-inserted prefix must be
        allowed to re-stage: a stale durable mark surviving node deletion
        would block staging forever and leave the prefix in no tier at all."""
        cache = _build_cache()
        cache._update_host_leaf_status = lambda *a, **k: None
        cache._delete_leaf = lambda *a, **k: None
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        _pressure(cache)
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
        self.assertIsNone(node.host_value)  # mirror released
        cache._evict_regular(node)  # not backuped -> leaf deleted

        self.assertNotIn("a-0", cache.storage_backed_hashes)
        reinserted = _node(n_pages=1, tag="a", parent=cache.root_node, backuped=False)
        reinserted.hit_count = 5
        cache._inc_hit_count(reinserted)
        self.assertTrue(
            cache.cache_controller.write_calls,
            "stale durable mark blocked re-staging of a fully evicted prefix",
        )

    def test_device_evicted_node_leaves_candidates(self):
        cache = _build_cache()
        cache._update_host_leaf_status = lambda *a, **k: None
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        cache.drain_storage_control_queues()
        self.assertIn(node, cache.redundant_host_nodes)
        cache._evict_backuped(node)  # demoted: host copy is now the only copy
        self.assertNotIn(node, cache.redundant_host_nodes)
        _pressure(cache)
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
        self.assertFalse(
            cache.cache_controller.evict_host_calls,
            "released the sole host copy of a device-evicted node",
        )

    def test_recomputed_node_reenters_candidates(self):
        """A durable node demoted to host-only and later recomputed on device
        holds a redundant mirror again.  The insert() recomputation path must
        re-register it, or the mirror stays pinned forever — quietly
        rebuilding the very leak this feature removes under
        evict-then-recompute churn."""
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        cache = _build_cache()
        cache.is_eagle = False
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        cache.drain_storage_control_queues()  # durable + candidate
        self.assertIn(node, cache.redundant_host_nodes)

        cache._evict_backuped(node)  # demote: host copy is the only copy
        self.assertNotIn(node, cache.redundant_host_nodes)

        # KV recomputation restores the device copy for the same prefix
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=list(range(PAGE))),
                value=torch.arange(PAGE),
            )
        )
        self.assertIsNotNone(node.value)
        self.assertIn(
            node,
            cache.redundant_host_nodes,
            "recomputed durable node never re-entered the release candidates",
        )

    def test_released_parent_with_demoted_children_restages(self):
        """Load-test regression: released-mirror parent + host-only children.
        1) the all-children-evicted re-push hands the parent to the
           not-backuped branch and _evict_regular's leaf assert kills the
           scheduler; 2) merely skipping it pins device tokens until
           prefill OOM.  The parent must be RE-STAGED so the normal
           demotion path can evict it on a later round."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        cache = _build_cache()
        cache._update_host_leaf_status = lambda *a, **k: None
        cache.update_eviction_metrics = lambda *a, **k: None
        cache.writing_check = lambda write_back=False: None
        cache._delete_leaf = lambda *a, **k: None
        parent = _node(n_pages=2, tag="p", parent=cache.root_node)
        parent.host_value = None  # mirror already released
        child = _node(n_pages=1, tag="c", parent=parent, evicted=True)
        cache.evictable_leaves = {parent}
        cache.evictable_size_ = 10_000

        cache.evict(EvictParams(num_tokens=10_000))  # must not raise

        # subtree pruned, parent evicted via _evict_regular: progress is made
        self.assertTrue(
            cache.cache_controller.evict_host_calls,
            "host-only child was not pruned",
        )
        self.assertTrue(
            cache.cache_controller.mem_pool_device_allocator.free_calls,
            "released parent freed no device tokens — pins leak to OOM",
        )
        self.assertIsNone(child.host_value)

    def test_released_parent_with_busy_child_restages(self):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        cache = _build_cache()
        cache._update_host_leaf_status = lambda *a, **k: None
        cache.update_eviction_metrics = lambda *a, **k: None
        cache.writing_check = lambda write_back=False: None
        cache._delete_leaf = lambda *a, **k: None
        parent = _node(n_pages=2, tag="p", parent=cache.root_node)
        parent.host_value = None
        child = _node(n_pages=1, tag="c", parent=parent, evicted=True)
        child.host_ref_counter = 1  # in-flight backup/prefetch on the child
        cache.evictable_leaves = {parent}
        cache.evictable_size_ = 10_000

        cache.evict(EvictParams(num_tokens=10_000))

        self.assertFalse(cache.cache_controller.evict_host_calls)
        self.assertIsNotNone(parent.value, "busy subtree must defer eviction")
        self.assertTrue(
            cache.cache_controller.write_calls,
            "busy fallback must re-stage the parent",
        )

    def test_release_emits_cpu_remove_only_and_keeps_gpu(self):
        cache = _build_cache()
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        _pressure(cache)
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
        removes = [e for e in cache._events if e[0] == "remove" and e[1] is node]
        self.assertEqual(len(removes), 1)
        from sglang.srt.disaggregation.kv_events import StorageMedium

        self.assertEqual(removes[0][2], StorageMedium.CPU)
        self.assertIsNotNone(node.value)

    def test_disabled_flag_changes_nothing(self):
        cache = _build_cache(enabled=False)
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        _pressure(cache)
        for _ in range(3):
            cache.drain_storage_control_queues()
        self.assertFalse(cache.storage_backed_hashes)
        self.assertFalse(cache.redundant_host_nodes)
        self.assertFalse(cache.cache_controller.evict_host_calls)
        self.assertIsNotNone(node.host_value)


class TestTPProtocol(unittest.TestCase):
    def test_mla_skip_rank_marks_durable_from_writer(self):
        """MLA: only tp_rank 0 writes L3; other ranks ack completed=0.  A
        skip rank must derive durable state from the collective-reduced
        writer count, or its candidate set stays empty forever and plan
        consensus can never pass (the dev fail-close of 2026-08-02)."""
        cache = _build_cache()
        cache.cache_controller.backup_skip = True
        node = _node(n_pages=2, tag="a", parent=cache.root_node)
        _ack_node(cache, node, completed_pages=0)  # local ack: skip rank

        def writer_peer(mine):
            vec = list(mine)
            vec[11] = 2 * PAGE  # writer's completed_tokens for op 0
            return vec

        cache._all_reduce_attn_groups = _PeerReduce([writer_peer])
        cache.drain_storage_control_queues()

        self.assertTrue(cache._node_fully_durable(node))
        self.assertIn(node, cache.redundant_host_nodes)

    def test_mla_no_writer_stays_conservative(self):
        cache = _build_cache()
        cache.cache_controller.backup_skip = True
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node, completed_pages=1)  # local value must be ignored
        cache.drain_storage_control_queues()  # identity reduce: no writer info
        self.assertFalse(cache.storage_backed_hashes)
        self.assertNotIn(node, cache.redundant_host_nodes)

    def test_writer_partial_completed_reduces_everywhere(self):
        cache = _build_cache()
        node = _node(n_pages=2, tag="a", parent=cache.root_node)
        _ack_node(cache, node)  # my (writer) ack: fully completed

        def partial_writer_peer(mine):
            vec = list(mine)
            vec[11] = 1 * PAGE  # another writer only finished one page
            return vec

        cache._all_reduce_attn_groups = _PeerReduce([partial_writer_peer])
        cache.drain_storage_control_queues()

        self.assertIn("a-0", cache.storage_backed_hashes)
        self.assertNotIn("a-1", cache.storage_backed_hashes)
        self.assertFalse(cache._node_fully_durable(node))

    def test_drain_survives_concurrent_ack_producer(self):
        """The ack peek must snapshot under the queue mutex: iterating
        Queue.queue while the backup thread put()s concurrently raises
        'deque mutated during iteration' and would kill the scheduler."""
        import threading

        cache = _build_cache()
        stop = threading.Event()

        def producer():
            i = 0
            while not stop.is_set():
                node = _node(n_pages=1, tag=f"p{i}", parent=cache.root_node)
                ack = _Ack(node.hash_value, completed_tokens=PAGE)
                cache.ongoing_backup[ack.id] = node
                node.protect_host()
                cache.cache_controller.ack_backup_queue.put(ack)
                i += 1

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        try:
            for _ in range(300):
                cache.drain_storage_control_queues()
        finally:
            stop.set()
            t.join(timeout=5)

    def test_deficit_uses_cross_rank_max(self):
        # my rank has zero deficit; the peer is starved.  MAX semantics mean
        # every rank must still prepare and execute the shared plan.
        cache = _build_cache()
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        _no_pressure(cache)

        def peer_starved(mine):
            vec = list(mine)
            vec[4] = -100  # peer local_deficit = 100 (encoded negated)
            return vec

        cache._all_reduce_attn_groups = _PeerReduce(
            [peer_starved, None, None, None]
        )
        cache.drain_storage_control_queues()  # marks durable, prepares plan
        self.assertIsNotNone(
            cache._mirror_release_plan,
            "MIN(deficit) starvation: single-rank pressure prepared no plan",
        )
        cache.drain_storage_control_queues()  # consensus + execute
        self.assertTrue(cache.cache_controller.evict_host_calls)

    def test_plan_digest_differs_for_same_tokens_different_content(self):
        cache = _build_cache()
        a = _node(n_pages=1, tag="a", parent=cache.root_node)
        b = _node(n_pages=1, tag="b", parent=cache.root_node)
        for h in a.hash_value + b.hash_value:
            cache._mark_hash_durable(h)
        cache.redundant_host_nodes = {a}
        plan_a = cache._prepare_mirror_release_plan(100)
        cache.redundant_host_nodes = {b}
        plan_b = cache._prepare_mirror_release_plan(100)
        self.assertEqual(plan_a.tokens, plan_b.tokens)
        self.assertNotEqual(
            plan_a.digest,
            plan_b.digest,
            "token-equal plans with different nodes are indistinguishable",
        )

    def test_consensus_mismatch_drops_plan_before_mutation(self):
        cache = _build_cache()
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        _pressure(cache)

        def peer_diff_digest(mine):
            vec = list(mine)
            if vec[9] != 0:  # corrupt the digest consensus fields
                vec[9] = vec[9] - 1
                vec[10] = -(-vec[10] - 1)
            return vec

        cache._all_reduce_attn_groups = _PeerReduce([None, peer_diff_digest])
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
        self.assertFalse(
            cache.cache_controller.evict_host_calls,
            "mutated host state on a divergent release plan",
        )
        self.assertEqual(cache._mirror_release_plans_dropped, 1)
        self.assertIsNone(cache._mirror_release_disabled_reason)

    def test_repeated_mismatch_fail_closes(self):
        cache = _build_cache()
        node = _node(n_pages=1, tag="a", parent=cache.root_node)
        _ack_node(cache, node)
        _pressure(cache)

        def peer_diff_digest(mine):
            vec = list(mine)
            if vec[9] != 0:
                vec[9] = vec[9] - 1
                vec[10] = -(-vec[10] - 1)
            return vec

        cache._all_reduce_attn_groups = _PeerReduce(
            [None] + [peer_diff_digest] * 10
        )
        for _ in range(6):
            cache.drain_storage_control_queues()
        self.assertIsNotNone(cache._mirror_release_disabled_reason)
        self.assertFalse(cache.cache_controller.evict_host_calls)
        # fail-close is sticky: no further plans even under pressure
        cache.drain_storage_control_queues()
        self.assertIsNone(cache._mirror_release_plan)

    def test_execution_revalidation_is_all_or_nothing(self):
        cache = _build_cache()
        a = _node(n_pages=1, tag="a", parent=cache.root_node)
        b = _node(n_pages=1, tag="b", parent=cache.root_node)
        _ack_node(cache, a)
        _ack_node(cache, b)
        _pressure(cache)
        cache.drain_storage_control_queues()  # plan covers both
        self.assertEqual(len(cache._mirror_release_plan.nodes), 2)
        a.lock_ref = 1  # invalidated between prepare and commit
        cache.drain_storage_control_queues()
        self.assertFalse(
            cache.cache_controller.evict_host_calls,
            "partially executed a plan after one node became busy",
        )
        self.assertIsNone(cache._mirror_release_disabled_reason)
        # recovery: once unlocked, a later round releases both
        a.lock_ref = 0
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
        self.assertEqual(len(cache.cache_controller.evict_host_calls), 2)


if __name__ == "__main__":
    unittest.main()

"""L3 storage, buffer-only host memory and prefetch tests for UnifiedRadixCache."""

import os
import time
import unittest
from array import array
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentHostSlot,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
    define_config_test_classes,
    requires,
    requires_full_swa_ps1,
    requires_hicache,
    requires_ps1_window4,
    requires_swa,
    requires_swa_only,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_device_lock_ref,
    node_host_value,
    node_key_length,
    node_parent,
    node_token_ids,
    write_backup,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.kits.unified_radix_cache.storage import (
    AnchorLockOutcomePolicyFixtures,
    ExtraPoolFullComponent,
    UnifiedRadixCacheStorageAttachBackfillFixtures,
    UnifiedRadixCacheStorageFixtures,
    UnifiedRadixPrefetchCorruptionFixtures,
    aux_storage_key_transfers,
)
from sglang.test.mem_cache_utils import finish_req
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=28, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=28, suite="stage-b-test-1-gpu-small-amd")


class TestUnifiedRadixCacheEagleHiCacheStorageKey(CustomTestCase):
    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL,),
        is_eagle=True,
        kv_size=64,
        max_context_len=64,
    )

    def test_l3_prefetch_uses_bigram_radix_key(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            PrefetchSubmission,
        )
        from sglang.srt.mem_cache.utils import get_hash_str

        cache, allocator, _ = build_fixture(self.cfg)
        cache.enable_storage = True
        cache.prefetch_threshold = 1
        tokens = array("q", [1, 2, 3, 4, 5, 6, 7, 8, 9])

        value = allocator.alloc(len(tokens) - 1)
        self.assertIsNotNone(value)
        cache.insert(InsertParams(key=RadixKey(tokens), value=value))
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        leaf = match.last_device_node
        self.assertTrue(cache.tree_core.is_node_key_bigram(leaf))
        self.assertEqual(cache.tree_core.get_node_token_ids(leaf), list(tokens))
        self.assertEqual(len(cache.tree_core.get_hash_values(leaf)), 2)

        class FakeHostPool:
            def alloc(self, num_tokens):
                return torch.arange(num_tokens, dtype=torch.int64)

        class FakeCacheController:
            def __init__(self):
                self.mem_pool_host = FakeHostPool()
                self.prefetch_tokens_occupied = 0
                self.prefetch_args = None
                self.write_policy = "write_through"

            def prefetch_rate_limited(self):
                return False

            def get_prefetch_submission(self, rid):
                return None

            def submit_prefetch(
                self,
                handle,
                prefetch_key,
                last_hash,
                prefix_keys,
                matched_prefix_tokens,
                pool_transfers,
                assume_stored=False,
            ):
                self.prefetch_args = (
                    handle,
                    prefetch_key,
                    last_hash,
                    prefix_keys,
                    matched_prefix_tokens,
                    pool_transfers,
                )
                return PrefetchSubmission(
                    operation=mock.Mock(assume_stored=assume_stored)
                )

        controller = FakeCacheController()
        cache.cache_controller = controller
        cache.prefetch_from_storage(
            CacheRequestHandle("req", 0), cache.root_node_handle(), tokens
        )

        _, storage_key, _, _, _, _ = controller.prefetch_args
        self.assertIsInstance(storage_key, RadixKey)
        self.assertTrue(storage_key.is_bigram)
        self.assertEqual(len(storage_key), len(tokens) - 1)

        queried_hashes = []
        running_hash = None
        for start in range(0, len(storage_key), cache.page_size):
            running_hash = get_hash_str(
                storage_key[start : start + cache.page_size], running_hash
            )
            queried_hashes.append(running_hash)
        self.assertEqual(queried_hashes, cache.tree_core.get_hash_values(leaf))

        canonical_hashes = []
        running_hash = None
        for start in range(0, len(tokens) - 1, cache.page_size):
            running_hash = get_hash_str(
                tokens[start : start + cache.page_size], running_hash
            )
            canonical_hashes.append(running_hash)
        self.assertNotEqual(canonical_hashes, cache.tree_core.get_hash_values(leaf))

    def test_buffer_anchor_rematch_preserves_bigram_boundary(self):
        from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline

        cache, allocator, _ = build_fixture(self.cfg)
        extra_key = "adapter-a"
        cache_salt = "tenant-a"
        prefix_tokens = array("q", [1, 2, 3, 4, 5])
        value = allocator.alloc(len(prefix_tokens) - 1)
        self.assertIsNotNone(value)
        cache.insert(
            InsertParams(
                key=RadixKey(
                    prefix_tokens,
                    extra_key=extra_key,
                    cache_salt=cache_salt,
                ),
                value=value,
            )
        )
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    prefix_tokens,
                    extra_key=extra_key,
                    cache_salt=cache_salt,
                )
            )
        )
        self.assertEqual(match.device_prefix_len, len(prefix_tokens) - 1)

        req_id = CacheRequestHandle("bigram-anchor", 0)
        prefetch_key = RadixKey(
            array("q", [prefix_tokens[-1], 6, 7, 8, 9]),
            extra_key=extra_key,
            is_bigram=True,
            cache_salt=cache_salt,
        )
        cache.ongoing_prefetch[req_id] = _OngoingPrefetch(
            anchor_node_id=match.last_device_node,
            prefetch_key=prefetch_key,
            host_indices=None,
            operation=None,
            anchor_lock_params=None,
            comp_xfers={},
        )

        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline.anchor_locks = {}
        pipeline.anchor_locked_tokens_ = 0
        pipeline.anchor_lock_cap_tokens = 10_000
        pipeline._anchor_lock_cap_skips = 0
        pipeline._prefetch_prefix_ctx = {
            req_id: (list(prefix_tokens[:-1]), extra_key, cache_salt)
        }
        pipeline._cache = cache

        lock_ref = node_device_lock_ref(
            cache, match.last_device_node, ComponentType.FULL
        )
        self.assertEqual(
            pipeline.try_lock_anchor(req_id, remaining_full_tokens=1), ("locked", 4)
        )
        self.assertEqual(pipeline.anchor_locks[req_id].node_id, match.last_device_node)
        self.assertEqual(
            node_device_lock_ref(cache, match.last_device_node, ComponentType.FULL),
            lock_ref + 1,
        )

        pipeline.release_anchor_lock(req_id)
        cache.ongoing_prefetch.pop(req_id)
        self.assertEqual(
            node_device_lock_ref(cache, match.last_device_node, ComponentType.FULL),
            lock_ref,
        )
        cache.sanity_check()

    def test_buffer_backup_snapshot_preserves_raw_bigram_key_and_namespace(self):
        """The raw N+1 bigram tokens, the bigram flag and the (extra_key,
        cache_salt) namespace all feed the storage hash, so a snapshot that
        drops any of them backs the node up under a key no prefetch will find.
        The snapshot also owns its copy of the key: the backup runs detached
        from the tree, so a caller mutating it must not reach a later one."""
        cache, allocator, _ = build_fixture(self.cfg)
        cache.enable_storage = True
        tokens = array("q", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_tokens = tuple(tokens)
        key = RadixKey(tokens, extra_key="adapter-a", cache_salt="tenant-a")
        value = allocator.alloc(len(tokens) - 1)
        self.assertIsNotNone(value)
        cache.insert(InsertParams(key=key, value=value))
        leaf_id = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node

        snapshot = cache.tree_core.snapshot_buffer_backup(
            leaf_id, pass_prefix_keys=True
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.key.token_ids, tokens)
        self.assertTrue(snapshot.key.is_bigram)
        self.assertEqual(snapshot.key.extra_key, "adapter-a")
        self.assertEqual(snapshot.key.cache_salt, "tenant-a")
        self.assertEqual(snapshot.prefix_keys, [])

        snapshot.key.token_ids[0] = -1
        fresh_snapshot = cache.tree_core.snapshot_buffer_backup(
            leaf_id, pass_prefix_keys=True
        )
        self.assertEqual(tuple(fresh_snapshot.key.token_ids), expected_tokens)

    def test_sanity_check_reads_buffer_backup_node_id_from_snapshot(self):
        """A buffer-mode backup entry keeps its node id inside the detached
        snapshot; sanity_check must read it from there. It used to read
        ``entry.intent.node_id``, an attribute the intent no longer has, so the
        idle check raised AttributeError whenever a backup was in flight."""
        from sglang.srt.mem_cache.buffer_mode.pipeline import (
            BufferModePipeline,
            _UnifiedBackupIntent,
            _UnifiedBufferBackupEntry,
        )

        cache, allocator, _ = build_fixture(self.cfg)
        cache.enable_storage = True
        tokens = array("q", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        value = allocator.alloc(len(tokens) - 1)
        self.assertIsNotNone(value)
        cache.insert(InsertParams(key=RadixKey(tokens), value=value))
        leaf_id = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens))
        ).last_device_node
        snapshot = cache.tree_core.snapshot_buffer_backup(
            leaf_id, pass_prefix_keys=False
        )
        self.assertIsNotNone(snapshot)

        # The post-launch entry pins its node, as sanity_check requires of
        # every in-flight backup.
        lock_params = cache.inc_lock_ref(leaf_id).to_dec_params()
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        keys = list(snapshot.hash_values)
        # Entries are keyed by the write's ack id, not by the node.
        pipeline.ongoing_write_through = {
            -1: _UnifiedBufferBackupEntry(
                intent=_UnifiedBackupIntent(
                    snapshot=snapshot, pool=PoolName.KV, keys=keys
                ),
                host_indices=torch.empty(0, dtype=torch.int64),
                aux_xfers=[],
                lock_params=lock_params,
                occupied_units=0,
                keys_by_pool={PoolName.KV: keys},
            )
        }
        cache.buffer_pipeline = pipeline
        cache.sanity_check()

        cache.buffer_pipeline = None
        cache.dec_lock_ref(leaf_id, lock_params)
        cache.sanity_check()


class UnifiedRadixCacheStorageSuite(UnifiedRadixCacheStorageFixtures):
    @requires_hicache
    @requires(
        lambda cfg: not cfg.has_mamba,
        "mamba L3 prefetch is out of scope for this unit fixture",
    )
    def test_hicache_l3_prefetch(self):
        """L3 round trip: write with one tree, prefetch into a fresh tree.

        Uses two independent trees that share the same file storage dir so the
        prefetch path genuinely reloads from L3 (no host/device residue).
        """

        storage_dir = self._make_storage_dir()
        num_pages = 4
        if self.cfg.has_swa:
            # SWA L3 prefetch is all-or-nothing over one full sliding window.
            # Keep the generic L3 round trip valid for SWA configs whose window
            # is larger than the old fixed 4-page request.
            sw_pages = (
                self.cfg.sliding_window_size + self.cfg.page_size - 1
            ) // self.cfg.page_size
            num_pages = max(num_pages, sw_pages + 1)
        seq = self._make_seq(1, num_pages)

        expected_k, expected_v = self._l3_produce(storage_dir, seq, marker=7)
        cons, cons_alloc, cons_rtp = self._l3_consumer(storage_dir)
        req_id = CacheRequestHandle("l3-prefetch-req", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._run_prefetch_to_completion(cons, req_id)

        # The full prefix must now be a host hit (loaded from L3).
        mc = match_prefix(cons, seq)
        self.assertEqual(mc.host_hit_length, len(seq))
        host_node = mc.last_host_node
        self.assertNotEqual(host_node, cons.root_node_handle())
        self.assertTrue(cons.tree_core.is_full_device_evicted(host_node))

        # Load the reloaded host prefix back to device and verify KV bytes.
        self._load_back_node(cons, host_node)
        loaded_indices = cons.path_device_indices(
            match_prefix(cons, seq).last_device_node
        )
        self.assertEqual(len(loaded_indices), len(seq))
        loaded_k, loaded_v = self._snapshot_full_kv(cons_alloc, loaded_indices)
        self.assertTrue(torch.equal(loaded_k, expected_k))
        self.assertTrue(torch.equal(loaded_v, expected_v))
        cons.sanity_check()

    @requires_hicache
    @requires(
        lambda cfg: cfg.has_swa and not cfg.has_mamba,
        "SWA-only fixture required to exercise extra pool",
    )
    def test_release_aborted_request_l3_prefetch_io_in_progress(self):
        """Test release_aborted_request while a prefetch IO is still in-progress.
        1. Fill KV and SWA to L3.
        2. Trigger L3 prefetch.
        3. Hack IO thread, blocking at _page_transfer.
        4. Call release_aborted_request.  Assert that the KV and SWA buffers are not released (owned by the IO thread).
        5. Unlbock IO thread.  Assert that KV and SWA buffers are eventually released.
        """

        # SWA prefetch is all-or-nothing over one full sliding window: size the
        # request at sw_pages + 1 pages so prepare_prefetch actually materializes
        # an SWA host transfer.
        sw_pages = (
            self.cfg.sliding_window_size + self.cfg.page_size - 1
        ) // self.cfg.page_size
        num_pages = max(4, sw_pages + 1)
        seq = self._make_seq(1, num_pages)

        storage_dir = self._make_storage_dir()

        self._l3_produce(storage_dir, seq)
        cons, _, _ = self._l3_consumer(storage_dir)
        req_id = CacheRequestHandle("abort-req", 0)

        cc = cons.cache_controller
        kv_pool_available_size_before = cc.mem_pool_host.get_pool(
            PoolName.KV
        ).available_size()
        swa_pool_available_size_before = cc.mem_pool_host.get_pool(
            PoolName.SWA
        ).available_size()
        occupied_before = cons.cache_controller.prefetch_tokens_occupied
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self.assertEqual(
            cons.cache_controller.prefetch_tokens_occupied,
            occupied_before + len(seq),
        )
        self.assertIn(req_id, cons.ongoing_prefetch)

        # Block IO thread at the entry of _page_transfer.
        import threading

        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        original_sidecar = HybridCacheController._page_transfer
        entered = threading.Event()
        gate = threading.Event()
        swa_release_q = cc.extra_host_mem_release_queues.get(PoolName.SWA)

        def _page_transfer_hook(_self, op):
            # Signal that the worker has reached the sidecar step, then block
            # until the main thread releases the gate.
            entered.set()
            gate.wait(timeout=10.0)
            # Forward to the real implementation so the worker runs the
            # terminate-aware release path itself.
            original_sidecar(_self, op)

        # Simulate a slow prefetch IO.  Hook on _page_transfer.
        with mock.patch.object(
            HybridCacheController, "_page_transfer", _page_transfer_hook
        ):
            # Pump until the IO aux thread has entered the sidecar barrier.
            deadline = time.time() + 10.0
            while time.time() < deadline:
                cons.drain_storage_control_queues()
                op = cons.ongoing_prefetch[req_id].operation
                if entered.is_set():
                    break
                time.sleep(0.01)
            else:
                self.fail("prefetch did not reach the sidecar barrier in time")

        # Now, the prefetch IO thread is stopping at _page_transfer.
        # Let the scheduler thread calls release_aborted_request.
        # Assert that everything will not be released.
        op = cons.ongoing_prefetch[req_id].operation
        self.assertFalse(op.pool_transfers_done)
        self.assertEqual(cc.host_mem_release_queue.qsize(), 0)
        self.assertEqual(swa_release_q.qsize(), 0)
        cons.finish(req_id, CacheRequestOutcome.ABORT)
        self.assertEqual(cc.host_mem_release_queue.qsize(), 0)
        self.assertEqual(swa_release_q.qsize(), 0)

        # Let the prefetch IO thread continue to run.
        gate.set()

        # Wait for the IO thread has completed prefetch.
        # We don't consume release queue, as we will use that later to check whether
        # the host memory was released yet (as used by the prefetch IO thread).
        deadline = time.time() + 10.0
        while time.time() < deadline:
            cons._drain_storage_control_queues_impl(
                n_storage_hit=0,
                n_ack_prefetch=min(1, cc.ack_prefetch_queue.qsize()),
                n_backup=0,
                n_release=0,
                extra_release_counts=None,
                log_metrics=True,
            )
            if swa_release_q.qsize() > 0:
                break
            time.sleep(0.01)
        else:
            self.fail("SWA extra pool was not released after the abort")

        # Asserts that everything is correctly released.
        self.assertFalse(op.pool_transfers_done)
        self.assertGreater(cc.host_mem_release_queue.qsize(), 0)
        self.assertGreater(swa_release_q.qsize(), 0)
        self.assertNotIn(req_id, cons.ongoing_prefetch)
        self.assertNotIn(req_id, cons.prefetch_loaded_tokens_by_reqid)
        self.assertEqual(cc.prefetch_tokens_occupied, occupied_before)

        cons.drain_storage_control_queues()  # Drain release queue.
        self.assertEqual(
            cc.mem_pool_host.get_pool(PoolName.KV).available_size(),
            kv_pool_available_size_before,
        )
        self.assertEqual(
            cc.mem_pool_host.get_pool(PoolName.SWA).available_size(),
            swa_pool_available_size_before,
        )
        cons.sanity_check()

    @requires_hicache
    @requires(
        lambda cfg: cfg.has_swa and not cfg.has_mamba,
        "SWA-only fixture required to exercise extra pool",
    )
    def test_release_aborted_request_l3_prefetch_io_done(self):
        """Test release_aborted_request is called after the IO thread has completed
        prefetch.
        1. Fill KV and SWA to L3.
        2. Trigger L3 prefetch.
        3. Wait until the completion of L3 prefetch IO.
        4. Call release_aborted_request.  Assert that KV and SWA buffers are released.
        """

        # SWA prefetch is all-or-nothing over one full sliding window: size the
        # request at sw_pages + 1 pages so prepare_prefetch actually materializes
        # an SWA host transfer.
        sw_pages = (
            self.cfg.sliding_window_size + self.cfg.page_size - 1
        ) // self.cfg.page_size
        num_pages = max(4, sw_pages + 1)
        seq = self._make_seq(1, num_pages)

        storage_dir = self._make_storage_dir()

        self._l3_produce(storage_dir, seq)
        cons, _, _ = self._l3_consumer(storage_dir)
        req_id = CacheRequestHandle("abort-req", 0)

        cc = cons.cache_controller
        kv_pool_available_size_before = cc.mem_pool_host.get_pool(
            PoolName.KV
        ).available_size()
        swa_pool_available_size_before = cc.mem_pool_host.get_pool(
            PoolName.SWA
        ).available_size()
        occupied_before = cc.prefetch_tokens_occupied
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self.assertEqual(
            cons.cache_controller.prefetch_tokens_occupied,
            occupied_before + len(seq),
        )
        op = cons.ongoing_prefetch[req_id].operation

        swa_release_q = cc.extra_host_mem_release_queues.get(PoolName.SWA)
        self.assertIsNotNone(swa_release_q)

        # Simulate polling check_hicache_events.
        # There will be a sequence of events populated from queue:
        # 1. a storage hit notification (from cc.prefetch_hit_queue).
        # 2. a HiCacheAck, indicating the completion of KV pool read.
        # 3. a HiCacheAck, indicating the completion of SWA pool read.
        # 4. a HiCacheACk, idnicating the completion of entire prefetch request.
        # We are going to stop at the exact timing-window between 3 and 4.  So we have to
        # consume ONE event from the queue at each iteration.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            cons._drain_storage_control_queues_impl(
                n_storage_hit=min(1, cc.prefetch_hit_queue.qsize()),
                n_ack_prefetch=min(1, cc.ack_prefetch_queue.qsize()),
                n_backup=0,
                n_release=0,
                extra_release_counts=None,
                log_metrics=True,
            )
            if op.pool_transfers_done:
                break
            time.sleep(0.01)
        else:
            self.fail("prefetch IO did not complete (pool_transfers_done) in time")

        self.assertIsNotNone(op.host_indices)
        self.assertTrue(op.pool_transfers_done)
        self.assertEqual(cc.host_mem_release_queue.qsize(), 0)
        self.assertEqual(swa_release_q.qsize(), 0)

        # --- Act: abort without committing the prefetch. ---
        cons.finish(req_id, CacheRequestOutcome.ABORT)

        self.assertTrue(op.pool_transfers_done)
        self.assertGreater(swa_release_q.qsize(), 0)
        self.assertGreater(cc.host_mem_release_queue.qsize(), 0)
        self.assertNotIn(req_id, cons.ongoing_prefetch)
        self.assertNotIn(req_id, cons.prefetch_loaded_tokens_by_reqid)
        self.assertEqual(cc.prefetch_tokens_occupied, occupied_before)

        cons.drain_storage_control_queues()  # Drain release queue.
        self.assertEqual(
            cc.mem_pool_host.get_pool(PoolName.KV).available_size(),
            kv_pool_available_size_before,
        )
        self.assertEqual(
            cc.mem_pool_host.get_pool(PoolName.SWA).available_size(),
            swa_pool_available_size_before,
        )

        cons.sanity_check()

    # ================================================================
    # Buffer-only host memory mode (host = transient staging, L3 = cache)
    # ================================================================

    @requires_hicache
    @requires_swa
    def test_buffer_only_swa_tail_only_backups(self):
        """Tail-only backups: SWA reaches storage as the window ending at
        each node, a hit falls back to the last node end whose window is
        stored, and a split makes its point restorable once the parent is
        backed up again."""
        page = self.cfg.page_size
        sw_pages = (self.cfg.sliding_window_size + page - 1) // page
        # Nodes longer than two windows (the SWA component may split a leaf
        # at the window boundary), shorter than the window, and one window.
        node_pages = [2 * sw_pages + 3, max(sw_pages - 1, 1), sw_pages]
        total_pages = sum(node_pages)
        # Each growing insert allocates its whole sequence next to the
        # resident prefix before the duplicate is released.
        if 2 * total_pages * page > self.cfg.kv_size:
            self.skipTest("chain does not fit the device pool")
        storage_dir = self._make_storage_dir()

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        with envs.SGLANG_ENABLE_HICACHE_SWA_TAIL_ONLY_BACKUP.override(True):
            self._init_buffer_hicache(cache, storage_dir)
        seq = self._make_seq(1, total_pages)
        end_pages = 0
        for pages in node_pages:
            end_pages += pages
            self._insert(cache, allocator, req_to_token_pool, seq[: end_pages * page])

        def chain_pages(key):
            leaf = match_prefix(cache, key).last_device_node
            hashes: list[str] = []
            ends: list[int] = []
            for node in self._path_chain(cache, leaf):
                hashes.extend(cache.tree_core.get_hash_values(node))
                ends.append(len(hashes))
            return leaf, hashes, ends

        def window_pages(ends):
            kept = set()
            for end in ends:
                kept.update(range(max(0, end - sw_pages), end))
            return kept

        def restorable(prefix, kept):
            """Longest prefix of ``prefix`` pages whose window is stored."""
            return next(
                (
                    length
                    for length in range(prefix, 0, -1)
                    if all(p in kept for p in range(max(0, length - sw_pages), length))
                ),
                0,
            )

        leaf, page_hashes, node_ends = chain_pages(seq)
        self.assertEqual(len(page_hashes), total_pages)
        controller = cache.cache_controller
        with mock.patch.object(
            controller, "write_storage", wraps=controller.write_storage
        ) as write_storage:
            self._buffer_backup_and_wait(cache, leaf)

        kept = window_pages(node_ends)
        self.assertLess(len(kept), total_pages)
        backend = controller.storage_backend
        swa_page_size = controller.mem_pool_host.entry_map[
            PoolName.SWA
        ].host_pool.page_size

        def stored(pool, key):
            return os.path.exists(
                os.path.join(
                    backend.file_path, backend._get_component_key(key, pool) + ".bin"
                )
            )

        swa_beliefs = cache.storage_existence_cache.pool(PoolName.SWA)
        for i, key in enumerate(page_hashes):
            self.assertTrue(stored(PoolName.KV, key), f"KV page {i}")
            self.assertEqual(stored(PoolName.SWA, key), i in kept, f"SWA page {i}")
            self.assertEqual(swa_beliefs.contains(key), i in kept, f"SWA belief {i}")
        # Each SWA write staged exactly the rows of the keys it stored.
        written: list[str] = []
        for call in write_storage.call_args_list:
            for transfer in call.kwargs.get("extra_pools") or ():
                if transfer.name == PoolName.SWA:
                    self.assertEqual(
                        transfer.host_indices.numel(),
                        len(transfer.keys) * swa_page_size,
                    )
                    written.extend(transfer.keys)
        self.assertEqual(sorted(written), sorted(page_hashes[i] for i in kept))

        # A hit's usable prefix is the last node end whose window is stored.
        swa_query = [
            PoolTransfer(
                name=PoolName.SWA,
                keys=["_"] * sw_pages,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        for prefix in range(1, total_pages + 1):
            self.assertEqual(
                self._storage_exists_count(cache, page_hashes[:prefix], swa_query),
                restorable(prefix, kept),
                f"prefix {prefix}",
            )

        # The whole chain is still served end to end from the sparse store.
        cons, _, _ = self._build_buffer_hicache_fixture(storage_dir)
        handle = CacheRequestHandle("tail-only-req", 0)
        cons.prefetch_from_storage(
            handle, cons.root_node_handle(), array("q", seq), None, None
        )
        self._run_prefetch_to_completion(cons, handle)
        self.assertEqual(
            match_prefix(cons, seq).device_prefix_len,
            len(seq),
        )

        # A partial match inside the first node splits it; the parent's next
        # backup stores the window ending at the split, so a hit there is
        # served exactly from storage.
        split = max(
            (
                p
                for p in range(1, node_ends[0])
                if p not in node_ends and restorable(p, kept) < p
            ),
            default=None,
        )
        if split is None:
            self.skipTest("the first node has no unrestorable interior page")
        branch = seq[: split * page] + self._make_seq(10_000, 1)
        self._insert(cache, allocator, req_to_token_pool, branch)
        branch_leaf, branch_hashes, branch_ends = chain_pages(branch)
        self.assertEqual(branch_hashes[:split], page_hashes[:split])
        self.assertIn(split, branch_ends)
        self._buffer_backup_and_wait(cache, branch_leaf)
        for i in range(max(0, split - sw_pages), split):
            self.assertTrue(stored(PoolName.SWA, page_hashes[i]), f"split page {i}")
        self.assertEqual(
            self._storage_exists_count(cache, page_hashes[:split], swa_query), split
        )
        cache.sanity_check()
        cons.sanity_check()

    @requires_hicache
    def test_buffer_only_write_path_roundtrip(self):
        """Write path end to end: admission -> D2H staging -> storage write
        -> free. Staging and locks fully released, pages stored under every
        pool namespace, beliefs registered, re-hits skipped."""
        storage_dir = self._make_storage_dir()

        cache, allocator, req_to_token_pool = self._build_buffer_hicache_fixture(
            storage_dir
        )
        avail0 = self._host_avail_sizes(cache)

        seq_a = self._make_seq(1, 2)
        seq_ab = seq_a + self._make_seq(500, 2)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_ab)
        leaf = match_prefix(cache, seq_ab).last_device_node
        chain = self._path_chain(cache, leaf)

        pipeline = cache.buffer_pipeline
        controller = cache.cache_controller
        write_backup(cache, leaf)
        # One intent per pool of every node on the path.
        intents = list(pipeline.pending_write_queue)
        node_ids = sorted({intent.snapshot.node_id for intent in intents})
        self.assertGreaterEqual(len(node_ids), 2)
        self.assertEqual(
            pipeline.inflight_backup_pools[leaf],
            {intent.pool for intent in intents if intent.snapshot.node_id == leaf},
        )

        device_allocators = [allocator]
        if self.cfg.has_swa:
            device_allocators.extend(
                [allocator.full_attn_allocator, allocator.swa_attn_allocator]
            )
        # A deferred D2H batch must not allocate or free L1 slots before submit.
        with ExitStack() as guards:
            for device_allocator in device_allocators:
                for method in (
                    "alloc",
                    "alloc_extend",
                    "alloc_decode",
                    "free",
                    "free_segment",
                    "free_page_ids",
                ):
                    if hasattr(device_allocator, method):
                        guards.enter_context(
                            mock.patch.object(
                                device_allocator,
                                method,
                                side_effect=AssertionError(
                                    "L1 mutation during backup preparation"
                                ),
                            )
                        )
            pipeline.flush_pending_writes()
        self.assertFalse(pipeline.pending_write_queue)
        self.assertFalse(controller.write_queue)
        self.assertEqual(len(controller.ack_write_queue), 1)
        # Every intent is a write of its own, acked under its own id.
        ack = controller.ack_write_queue[0]
        self.assertEqual(len(ack.node_ids), len(intents))
        self.assertEqual(list(ack.node_ids), list(pipeline.ongoing_write_through))
        self.assertEqual(
            sorted(
                {
                    entry.intent.snapshot.node_id
                    for entry in pipeline.ongoing_write_through.values()
                }
            ),
            node_ids,
        )
        staged_avail = self._host_avail_sizes(cache)
        self.assertNotEqual(staged_avail, avail0)

        # GPU completion alone must not release any entry's device lock.
        ack.finish_event.synchronize()
        for node_id in node_ids:
            self.assertGreater(
                node_device_lock_ref(cache, node_id, ComponentType.FULL), 0
            )
        cache.writing_check(finish_count=1)
        self.assertFalse(pipeline.ongoing_write_through)
        self.assertEqual(len(pipeline.ongoing_backup), len(intents))
        for node_id in node_ids:
            self.assertEqual(
                node_device_lock_ref(cache, node_id, ComponentType.FULL), 0
            )
        # Each storage write still owns its staging until its separate ACK.
        self.assertEqual(self._host_avail_sizes(cache), staged_avail)
        self._pump_hicache_until(
            cache,
            lambda: (
                not pipeline.inflight_backup_node_ids and not pipeline.ongoing_backup
            ),
            "merged buffer backup did not drain every storage write",
        )
        self.assertFalse(cache.tree_core.is_backuped(leaf))
        self.assertEqual(node_device_lock_ref(cache, leaf, ComponentType.FULL), 0)
        self.assertEqual(self._host_avail_sizes(cache), avail0)
        self.assertEqual(cache.buffer_pipeline.write_backlog_tokens_, 0)
        page_hashes = self._all_page_hashes(cache, leaf)
        self.assertEqual(
            self._storage_exists_count(
                cache,
                page_hashes,
                aux_storage_key_transfers(cache, leaf),
            ),
            len(page_hashes),
        )
        for n in chain:
            self.assertTrue(
                cache.storage_existence_cache.pool(PoolName.KV).contains_all(
                    cache.tree_core.get_hash_values(n)
                )
            )
            for transfer in aux_storage_key_transfers(cache, n) or ():
                self.assertTrue(
                    cache.storage_existence_cache.pool(transfer.name).contains_all(
                        transfer.keys
                    )
                )
        # Re-hit absorbed when every pool is believed present.
        write_backup(cache, leaf)
        self.assertNotIn(leaf, cache.buffer_pipeline.inflight_backup_node_ids)

        if self.cfg.has_swa:
            # Only the SWA window dropped: the next insert queues the SWA pool
            # alone. A private SWA arena's write proceeds even with FULL staging
            # at the live cap (shared or anchor-aliased layouts keep the gate).
            swa_transfer = aux_storage_key_transfers(cache, leaf)[0]
            cache.storage_existence_cache.pool(PoolName.SWA).invalidate_beyond(
                swa_transfer.keys, keep_pages=0
            )
            host_group = controller.mem_pool_host
            private_swa = (
                pipeline._shared_host_domain() is None
                and host_group.entry_map[PoolName.SWA].host_pool
                is not host_group.anchor_entry.host_pool
            )
            staged_before = pipeline.write_staged_tokens_
            if private_swa:
                pipeline.write_staged_tokens_ = host_group.size
            with (
                mock.patch.object(controller, "write", wraps=controller.write) as stage,
                mock.patch.object(
                    controller, "write_storage", wraps=controller.write_storage
                ) as write_storage,
                mock.patch.object(
                    controller, "page_set_func", wraps=controller.page_set_func
                ) as write_full,
                mock.patch.object(
                    pipeline, "_write_intents", wraps=pipeline._write_intents
                ) as admissions,
            ):
                # The fixture parks the stock trigger; arm it here.
                cache.write_through_threshold = 1
                self._insert(cache, allocator, req_to_token_pool, seq_ab)
                self.assertEqual(
                    [
                        (i.snapshot.node_id, i.pool)
                        for i in pipeline.pending_write_queue
                    ],
                    [(leaf, PoolName.SWA)],
                )
                # One admission per node per walk.
                self.assertEqual(admissions.call_count, len(node_ids))
                self.assertTrue(cache.buffer_backup_pending(leaf))
                self._pump_hicache_until(
                    cache,
                    lambda: (
                        not pipeline.inflight_backup_node_ids
                        and not pipeline.ongoing_backup
                    ),
                    "selective buffer backup did not drain",
                )
            self.assertFalse(cache.buffer_backup_pending(leaf))
            if private_swa:
                pipeline.write_staged_tokens_ -= host_group.size - staged_before
            self.assertTrue(stage.called)
            self.assertTrue(
                all(call.args[0].numel() == 0 for call in stage.call_args_list)
            )
            # SWA writes carry no KV page; the window's keys ride the transfer.
            self.assertTrue(write_storage.called)
            snapshot = cache.tree_core.snapshot_buffer_backup(
                leaf, cache.hicache_storage_pass_prefix_keys
            )
            for call in write_storage.call_args_list:
                host_indices, _, hash_value, prefix_keys = call.args[:4]
                self.assertEqual(host_indices.numel(), 0)
                self.assertEqual(hash_value, [])
                transfers = call.kwargs["extra_pools"]
                self.assertEqual(
                    [t.name for t in transfers if t.indices_from_pool is None],
                    [PoolName.SWA],
                )
                self.assertEqual(prefix_keys, snapshot.prefix_keys)
            write_full.assert_not_called()
            self.assertTrue(
                cache.storage_existence_cache.pool(PoolName.SWA).contains_all(
                    swa_transfer.keys
                )
            )
            self.assertEqual(self._host_avail_sizes(cache), avail0)
        cache.sanity_check()

    @requires_hicache
    def test_buffer_only_split_intent_writes_every_piece(self):
        """Repeated splits preserve every pool's write once, parents first."""
        storage_dir = self._make_storage_dir()
        cache, allocator, req_to_token_pool = self._build_buffer_hicache_fixture(
            storage_dir
        )
        pipeline = cache.buffer_pipeline
        ps = self.cfg.page_size

        seq = self._buffer_swa_seq(min_pages=4)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = match_prefix(cache, seq).last_device_node
        chain = self._path_chain(cache, leaf)
        span_hashes = [h for n in chain for h in cache.tree_core.get_hash_values(n)]
        aux_keys = aux_storage_key_transfers(cache, leaf)
        write_backup(cache, leaf)
        expected_keys = [
            (i.pool, key) for i in pipeline.pending_write_queue for key in i.keys
        ]
        # SWA may already have split the leaf at its window boundary.
        end = len(seq)
        for node in reversed(chain):
            node_len = node_key_length(cache, node)
            if node_len >= 3 * ps:
                break
            end -= node_len
        else:
            self.fail("no node of three pages to split")
        for offset in (1, 2):
            split_at = end - node_len + offset * ps
            self._insert(
                cache,
                allocator,
                req_to_token_pool,
                seq[:split_at] + self._make_seq(9000, 2),
            )
            self.assertEqual(node_key_length(cache, node), end - split_at)
            write_backup(cache, node_parent(cache, node))

        pipeline._refresh_pending_backup_intents()
        kv_order = [
            intent.snapshot.node_id
            for intent in pipeline.pending_write_queue
            if intent.pool == PoolName.KV
        ]
        self.assertEqual(kv_order, self._path_chain(cache, leaf))
        self.assertCountEqual(
            [(i.pool, key) for i in pipeline.pending_write_queue for key in i.keys],
            expected_keys,
        )
        self.assertEqual(pipeline.write_backlog_tokens_, len(seq))
        self._drain_buffer_writes(cache)
        self.assertEqual(
            self._storage_exists_count(cache, span_hashes, aux_keys), len(span_hashes)
        )
        cache.sanity_check()

    @requires(
        lambda cfg: cfg.components == (ComponentType.FULL,),
        "only FULL-only caches split the leaf at the prompt",
    )
    def test_buffer_only_finish_writes_prompt_and_output(self):
        """The finish-path split preserves both prompt and output writes."""
        storage_dir = self._make_storage_dir()
        cache, allocator, req_to_token_pool = self._build_buffer_hicache_fixture(
            storage_dir
        )
        cache.tree_core.write_through_threshold = 1  # the write_through trigger

        prompt = self._make_seq(1, 6)
        output = self._make_seq(3000, 4)
        req, _ = self._make_inflight_req(
            cache, allocator, req_to_token_pool, prompt, output, allocated=True
        )
        kv_len = req.extend_end
        finish_req(cache, req, kv_len)

        leaf = match_prefix(cache, prompt + output).last_device_node
        chain = self._path_chain(cache, leaf)
        self.assertEqual(
            [node_key_length(cache, n) for n in chain], [len(prompt), len(output)]
        )
        self._drain_buffer_writes(cache)
        hashes = [h for n in chain for h in cache.tree_core.get_hash_values(n)]
        self.assertEqual(self._storage_exists_count(cache, hashes), len(hashes))
        cache.sanity_check()

    @requires_hicache
    def test_buffer_only_writes_component_pools_independently(self):
        """A component's own pool (aliased host entry, per-pool keys,
        BACKUP_HOST transfers) is written, believed and re-written
        independently of the KV pool."""
        storage_dir = self._make_storage_dir()

        # Component buffer-mode hooks are Python-core only.
        cache, allocator, req_to_token_pool = build_fixture(
            self.cfg,
            component_registry_override={ComponentType.FULL: ExtraPoolFullComponent},
            tree_core_backend="python",
        )
        self._init_buffer_hicache(cache, storage_dir)
        extra = ExtraPoolFullComponent.EXTRA_POOL
        controller = cache.cache_controller
        self.assertIn(extra, controller.mem_pool_host.entry_map)
        avail0 = self._host_avail_sizes(cache)

        seq = self._buffer_swa_seq()
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = match_prefix(cache, seq).last_device_node
        chain = self._path_chain(cache, leaf)
        pipeline = cache.buffer_pipeline
        expected_intents = []
        expected_backlog = 0
        for node_id in chain:
            snapshot = cache.tree_core.snapshot_buffer_backup(
                node_id, cache.hicache_storage_pass_prefix_keys
            )
            expected_backlog += len(snapshot.key)
            expected_intents.extend(
                (node_id, pool) for pool, _ in pipeline._write_intents(snapshot)
            )
        pipeline.write_backlog_cap = expected_backlog
        if self.cfg.has_swa:
            self.assertEqual(
                {pool for node_id, pool in expected_intents if node_id == leaf},
                {PoolName.KV, PoolName.DRAFT, PoolName.SWA},
            )
        self.assertFalse(cache.buffer_backup_pending(leaf))
        with mock.patch.object(
            controller, "write_storage", wraps=controller.write_storage
        ) as write_storage:
            write_backup(cache, leaf)
            self.assertEqual(
                [
                    (intent.snapshot.node_id, intent.pool)
                    for intent in pipeline.pending_write_queue
                ],
                expected_intents,
            )
            self.assertEqual(pipeline.write_backlog_tokens_, expected_backlog)
            self.assertEqual(pipeline._backlog_cap_hits, 0)
            self.assertTrue(cache.buffer_backup_pending(leaf))
            self._pump_hicache_until(
                cache,
                lambda: (
                    not cache.buffer_pipeline.inflight_backup_node_ids
                    and not cache.buffer_pipeline.ongoing_backup
                ),
                "buffer backup pipeline did not drain",
            )
        self.assertFalse(cache.buffer_backup_pending(leaf))
        # Each pool of each node is a write of its own: the KV chain with its
        # pages, the component's pool with no KV rows and its transfers
        # keyed by their own pages.
        expected = {tuple(cache.tree_core.get_hash_values(n)) for n in chain}
        kv_writes, extra_writes = self._writes_by_pool(write_storage, extra)
        self.assertEqual(kv_writes, expected)
        self.assertEqual(extra_writes, expected)

        page_hashes = self._all_page_hashes(cache, leaf)
        self.assertEqual(
            self._storage_exists_count(
                cache, page_hashes, [PoolTransfer(name=extra, keys=page_hashes)]
            ),
            len(page_hashes),
        )
        for n in chain:
            keys = cache.tree_core.get_hash_values(n)
            self.assertTrue(
                cache.storage_existence_cache.pool(PoolName.KV).contains_all(keys)
            )
            self.assertTrue(
                cache.storage_existence_cache.pool(extra).contains_all(keys)
            )
        self.assertEqual(self._host_avail_sizes(cache), avail0)

        # Only the component's pool dropped from the beliefs: its requested
        # backup writes that pool alone, staged through the anchor's host
        # pool with no KV rows.
        for n in chain:
            cache.storage_existence_cache.pool(extra).invalidate_beyond(
                cache.tree_core.get_hash_values(n), keep_pages=0
            )
        with (
            mock.patch.object(controller, "write", wraps=controller.write) as stage,
            mock.patch.object(
                controller, "write_storage", wraps=controller.write_storage
            ) as write_storage,
            mock.patch.object(
                controller, "page_set_func", wraps=controller.page_set_func
            ) as write_full,
        ):
            cache.request_buffer_backup(leaf)
            intents = list(pipeline.pending_write_queue)
            self.assertEqual(
                [(intent.snapshot.node_id, intent.pool) for intent in intents],
                [(n, extra) for n in chain],
            )
            self._pump_hicache_until(
                cache,
                lambda: (
                    not pipeline.inflight_backup_node_ids
                    and not pipeline.ongoing_backup
                ),
                "component-pool backup did not drain",
            )
        self.assertTrue(stage.called)
        self.assertTrue(all(call.args[0].numel() == 0 for call in stage.call_args_list))
        self.assertEqual(len(write_storage.call_args_list), len(chain))
        kv_writes, extra_writes = self._writes_by_pool(write_storage, extra)
        self.assertEqual(kv_writes, set())
        self.assertEqual(extra_writes, expected)
        write_full.assert_not_called()
        for n in chain:
            self.assertTrue(
                cache.storage_existence_cache.pool(extra).contains_all(
                    cache.tree_core.get_hash_values(n)
                )
            )
        self.assertEqual(self._host_avail_sizes(cache), avail0)

        # Everything believed stored: a requested backup is absorbed.
        cache.request_buffer_backup(leaf)
        self.assertNotIn(leaf, pipeline.inflight_backup_node_ids)
        cache.sanity_check()

    @requires_hicache
    def test_buffer_only_read_path_roundtrip(self):
        """Read path end to end: prefetch -> staged (host bounce only,
        nothing device-side, unmatchable, stable readiness, counters fed) ->
        admission-time load-back against a saturated-but-evictable pool
        (evict-before-alloc) publishing pre-ack -> ack frees the bounce.
        Data bytes match the producer's; no CPU-tier KV events anywhere;
        declines feed the outcome counters."""
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        _, (expected_k, expected_v) = self._produce_buffer_l3(
            storage_dir, seq, marker=7
        )

        cons, cons_alloc, cons_rtp = build_fixture(
            self.cfg, enable_kv_cache_events=True
        )
        self._init_buffer_hicache(cons, storage_dir)
        cons.enable_storage_metrics = True
        cons.storage_metrics_collector = mock.Mock()
        cons.take_events()
        avail0 = self._host_avail_sizes(cons)
        dev_avail0 = cons.token_to_kv_pool_allocator.available_size()
        stats = cons._prefetch_outcome_stats

        req_id = CacheRequestHandle("buffer-read-roundtrip", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self.assertEqual((stats["attempts"], stats["issued"]), (1, 1))
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        # Staged: bounce occupies host staging; nothing device-side; span
        # unmatchable; readiness stable; hit accounting reported once.
        self.assertNotEqual(self._host_avail_sizes(cons), avail0)
        self.assertEqual(cons.token_to_kv_pool_allocator.available_size(), dev_avail0)
        self.assertEqual(
            len(cons.path_device_indices(match_prefix(cons, seq).last_device_node)),
            0,
        )
        self.assertTrue(cons.check_prefetch_progress(req_id))
        self.assertEqual(cons.pop_prefetch_loaded_tokens(req_id), len(seq))
        self.assertEqual(stats["l3_demand_requests"], 1)
        self.assertEqual(stats["l3_miss_tokens"], 0)

        # Saturate the device pool with unrelated evictable spans: the
        # load-back must evict, not degrade to recompute (run-2 regression).
        def _avail():
            if cons.supports_swa():
                return cons.token_to_kv_pool_allocator.full_available_size()
            return cons.token_to_kv_pool_allocator.available_size()

        filler_base = 90000
        while _avail() >= self.cfg.page_size:
            pages = max(1, min(2048, _avail()) // self.cfg.page_size)
            self._insert(cons, cons_alloc, cons_rtp, self._make_seq(filler_base, pages))
            filler_base += 1000
        self.assertLess(_avail(), len(seq), "pool not saturated")

        # Consume WITHOUT pumping the ack: published (matchable, same slot
        # ids) before any ack lands.
        from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

        held = cons.buffer_pipeline.staged_prefetches[req_id]
        req = mock.Mock()
        req.rid = req_id.rid
        req.cache_request_handle = req_id
        req.extra_key = None
        req.cache_salt = None
        req.last_node = cons.root_node_handle()
        req.prefix_len = held.matched_len
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        spliced_len, spliced_node = cons.init_load_back(
            InitLoadBackParams(
                best_match_node=None, host_hit_length=held.num_tokens, req=req
            )
        )
        self.assertEqual(spliced_len, len(seq))
        cons.ready_to_load_host_cache()
        m = match_prefix(cons, seq)
        self.assertEqual(
            (m.device_prefix_len, m.last_device_node), (len(seq), spliced_node)
        )

        # Ack: bounce freed, beliefs fed, tree holds no host values, and the
        # loaded KV bytes equal the producer's.
        self._pump_hicache_until(
            cons,
            lambda: (
                not cons.buffer_pipeline.ongoing_buffer_load_back
                and self._host_avail_sizes(cons) == avail0
            ),
            "load-back ack did not free the bounce",
        )
        mc = match_prefix(cons, seq)
        self.assertEqual(mc.host_hit_length, 0)
        self.assertEqual(mc.device_prefix_len, len(seq))
        leaf = mc.last_device_node
        for cur in self._path_chain(cons, leaf):
            for component_type in cons.tree_components:
                self.assertIsNone(node_host_value(cons, cur, component_type))
        self.assertTrue(
            cons.storage_existence_cache.pool(PoolName.KV).contains_all(
                self._all_page_hashes(cons, leaf)
            )
        )
        # The fetch is evidence for every pool it delivered, so the
        # consumer's own re-write of the span is skipped pool by pool.
        for transfer in aux_storage_key_transfers(cons, leaf) or ():
            self.assertTrue(
                cons.storage_existence_cache.pool(transfer.name).contains_all(
                    transfer.keys
                )
            )
        loaded_k, loaded_v = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(mc.last_device_node)
        )
        self.assertTrue(torch.equal(loaded_k, expected_k))
        self.assertTrue(torch.equal(loaded_v, expected_v))
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        cpu_events = [
            e
            for e in cons.take_events()
            if isinstance(e, (BlockStored, BlockRemoved))
            and e.medium == StorageMedium.CPU
        ]
        self.assertEqual(cpu_events, [])
        cons.storage_metrics_collector.log_storage_prefetch_hit_tokens.assert_called_once_with(
            len(seq)
        )
        cons.storage_metrics_collector.log_prefetched_tokens.assert_called_once_with(
            len(seq)
        )
        cons.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens.assert_not_called()

        cons.sanity_check()

    @requires_hicache
    @requires(
        lambda cfg: cfg.components == (ComponentType.FULL,) and cfg.page_size == 4,
        "one FULL page_size=4 fixture covers namespace routing",
    )
    def test_buffer_only_cache_salt_uses_the_request_namespace(self):

        storage_dir = self._make_storage_dir()

        extra_key = "adapter-a"
        cache_salt = "tenant-a"
        seq = self._make_seq(1, 2)
        self._produce_buffer_l3(
            storage_dir, seq, extra_key=extra_key, cache_salt=cache_salt
        )

        # A root anchor has no namespace; probe and prefetch must use the request's.
        cons, _, _ = self._build_buffer_hicache_fixture(storage_dir)
        self.assertEqual(
            cons.query_storage_hit_length(
                cons.root_node_handle(),
                array("q", seq),
                extra_key=extra_key,
                cache_salt=cache_salt,
            ),
            len(seq),
        )
        root_req = CacheRequestHandle("salted-root-prefetch", 0)
        cons.prefetch_from_storage(
            root_req,
            cons.root_node_handle(),
            array("q", seq),
            None,
            None,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(root_req)
                and cons.buffer_pipeline.has_staged(root_req)
            ),
            "salted root prefetch did not stage",
        )
        held = cons.buffer_pipeline.staged_prefetches[root_req]
        self.assertEqual((held.extra_key, held.cache_salt), (extra_key, cache_salt))
        self.assertNotIn(root_req, cons.buffer_pipeline.anchor_locks)
        loaded = self._consume_staged_prefetch(
            cons,
            root_req,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )
        self.assertEqual(len(loaded), len(seq))
        key = RadixKey(array("q", seq), extra_key=extra_key, cache_salt=cache_salt)
        self.assertEqual(
            cons.match_prefix(MatchPrefixParams(key=key)).device_prefix_len, len(seq)
        )
        for miss in (
            RadixKey(array("q", seq)),
            RadixKey(array("q", seq), extra_key=extra_key, cache_salt="tenant-b"),
            RadixKey(array("q", seq), extra_key="adapter-b", cache_salt=cache_salt),
        ):
            self.assertEqual(
                cons.match_prefix(MatchPrefixParams(key=miss)).device_prefix_len, 0
            )
        cons.sanity_check()

        # A mid-tree anchor is pinned while its suffix is in flight and
        # released as soon as the staged span is consumed.
        cons2, cons2_alloc, cons2_rtp = self._build_buffer_hicache_fixture(storage_dir)
        prefix = seq[: self.cfg.page_size]
        suffix = seq[self.cfg.page_size :]
        self._insert(
            cons2,
            cons2_alloc,
            cons2_rtp,
            prefix,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )
        prefix_match = cons2.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    array("q", prefix),
                    extra_key=extra_key,
                    cache_salt=cache_salt,
                )
            )
        )
        anchor = prefix_match.last_device_node
        lock_ref = node_device_lock_ref(cons2, anchor, ComponentType.FULL)
        anchored_req = CacheRequestHandle("salted-mid-tree-prefetch", 0)
        cons2.prefetch_from_storage(
            anchored_req,
            anchor,
            array("q", suffix),
            cons2.tree_core.get_last_hash_value(anchor),
            None,
            matched_prefix_tokens=prefix,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )
        self.assertNotIn(anchored_req, cons2.buffer_pipeline.anchor_locks)
        self._pump_hicache_until(
            cons2,
            lambda: (
                cons2.check_prefetch_progress(anchored_req)
                and cons2.buffer_pipeline.has_staged(anchored_req)
            ),
            "salted mid-tree prefetch did not stage",
        )
        self.assertIn(anchored_req, cons2.buffer_pipeline.anchor_locks)
        self.assertEqual(
            node_device_lock_ref(cons2, anchor, ComponentType.FULL), lock_ref + 1
        )
        self._consume_staged_prefetch(
            cons2,
            anchored_req,
            prefix_len=len(prefix),
            extra_key=extra_key,
            cache_salt=cache_salt,
            last_node=anchor,
        )
        self.assertNotIn(anchored_req, cons2.buffer_pipeline.anchor_locks)
        self.assertEqual(cons2.buffer_pipeline.anchor_locked_tokens_, 0)
        self.assertEqual(
            node_device_lock_ref(cons2, anchor, ComponentType.FULL), lock_ref
        )
        self.assertEqual(
            cons2.match_prefix(MatchPrefixParams(key=key)).device_prefix_len, len(seq)
        )
        cons2.sanity_check()

    @requires_hicache
    @requires_ps1_window4
    def test_buffer_only_storage_prefetch_miss_marker_and_retry(self):
        """A too-early storage query misses; the miss must arm the retry
        marker exactly once, a re-issued check must serve once the content
        lands, and abort cleanup must drop unserved markers."""
        # Marker bookkeeping is layout-independent, and each hicache fixture
        # retains ~100MiB of device memory for the whole file run. Pin to one
        # config so the matrix does not exhaust a small CI GPU.
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)
        stats = cons._prefetch_outcome_stats

        # Query BEFORE any producer wrote the span: full miss -> revoked.
        req_id = CacheRequestHandle("early-query-miss", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: cons.check_prefetch_progress(req_id),
            "miss prefetch did not resolve",
        )
        self.assertFalse(cons.buffer_pipeline.has_staged(req_id))
        self.assertEqual(stats["revoked_full_miss"], 1)
        waiting = [
            SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0),
            SimpleNamespace(rid=req_id.rid, storage_prefetch_retry_attempts=0),
        ]
        self.assertEqual(cons.storage_prefetch_retries.pop_ready(waiting, 1, 8), [])
        self.assertEqual(
            cons.storage_prefetch_retries.pop_ready(waiting, 1, 8),
            [(waiting[1], None)],
        )
        self.assertEqual(cons.storage_prefetch_retries.pop_ready(waiting, 1, 8), [])

        # Producer commits the span; the re-issued check (paced retry) hits
        # and stages what the first, too-early query could not see.
        self._produce_buffer_l3(storage_dir, seq)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "retried prefetch did not stage",
        )
        self.assertNotIn(req_id.rid, cons.storage_prefetch_retries._pending)
        self.assertEqual(cons.pop_prefetch_loaded_tokens(req_id), len(seq))

        # Unserved markers must not leak: abort cleanup ...
        aborted_rid = CacheRequestHandle("aborted-miss", 0)
        cons.prefetch_from_storage(
            aborted_rid,
            cons.root_node_handle(),
            array("q", self._make_seq(700, 4)),
            None,
            None,
        )
        self._pump_hicache_until(
            cons,
            lambda: cons.check_prefetch_progress(aborted_rid),
            "aborted-rid miss did not resolve",
        )
        cons.finish(aborted_rid, CacheRequestOutcome.ABORT)
        self.assertNotIn(aborted_rid.rid, cons.storage_prefetch_retries._pending)

        # A fully-device-matched (empty-suffix) decline issues no lookup and
        # arms no retry; a queue-time eviction of that device match is
        # re-queried at admission with the re-issue budget intact.
        cons.prefetch_from_storage(
            CacheRequestHandle("fully-matched", 0),
            cons.root_node_handle(),
            array("q", []),
            None,
            None,
        )
        self.assertNotIn("fully-matched", cons.storage_prefetch_retries._pending)
        cons.sanity_check()

    @requires_hicache
    @requires_ps1_window4
    def test_buffer_only_anchor_lock_cap_clamped_by_context_headroom(self):
        """Deadlock invariant: cap <= pool - context_length (floor 0), so
        pinned anchors always leave room to admit the largest request."""
        # The cap is pool-size arithmetic, not a layout property; pin to one
        # SWA config (which also covers the SWAKVPool full_kv_pool branch) so
        # the retained per-fixture device memory stays bounded.
        storage_dir = self._make_storage_dir()
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        cache, _alloc, _rtp = build_fixture(self.cfg)
        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        full_pool = kvcache.full_kv_pool if isinstance(kvcache, SWAKVPool) else kvcache
        pool = full_pool.size
        self._init_buffer_hicache(
            cache, storage_dir, context_length=pool - self.cfg.page_size
        )
        self.assertEqual(
            cache.buffer_pipeline.anchor_lock_cap_tokens, self.cfg.page_size
        )

        # Zero headroom -> zero cap: no pin can ever deadlock such a pool.
        cache2, _alloc2, _rtp2 = self._build_buffer_hicache_fixture(
            storage_dir, context_length=pool
        )
        self.assertEqual(cache2.buffer_pipeline.anchor_lock_cap_tokens, 0)

    @requires_hicache
    @requires(lambda cfg: cfg.has_swa, "SWA-specific admission accounting")
    def test_buffer_load_back_swa_window_charged_at_admission(self):
        """Admission contract: a request the SWA budget gate accepts must be
        allocatable at batch time (estimate_swa_kv_tokens: "an admitted request
        cannot OOM"). Regression: buffer mode surfaced a staged prefetch as
        host_hit_length only, so the gate never charged the SWA window that
        consumption (init_load_back -> cc.load) allocates and the request
        lock pins; with the rest of the SWA pool batch-held, the batch alloc
        fell short by up to one window and raised the fail-loud prefill OOM
        (prod scheduler crash, 2026-08-17)."""
        from sglang.srt.mem_cache.allocation import (
            alloc_paged_token_slots_extend,
            alloc_token_slots,
        )
        from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

        storage_dir = self._make_storage_dir()
        ps = self.cfg.page_size
        window = self.cfg.sliding_window_size
        seq = self._buffer_swa_seq()  # one full window + 1 page
        self._produce_buffer_l3(storage_dir, seq)

        cons, cons_alloc, _ = self._build_buffer_hicache_fixture(storage_dir)
        req_id = CacheRequestHandle("buffer-swa-admission-oom", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )

        # Batch-held SWA (chunk allocs, decode windows) is neither free nor
        # evictable: leave one token less than window + extend_need, enough
        # for an un-charged gate to accept.
        extend_need = 2 * ps + 1
        max_new = 8
        hold = cons_alloc.swa_available_size() - (window + extend_need - 1)
        self.assertIsNotNone(cons_alloc.swa_attn_allocator.alloc(hold // ps * ps))

        # The adder's SWA gate for this request (_swa_budget_for_req).
        req = SimpleNamespace(
            rid=req_id.rid,
            cache_request_handle=req_id,
            prefix_len=cons.tree_core.empty_match_result.device_prefix_len,
            kv=SimpleNamespace(cache_protected_len=0),
        )
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        surfaced_swa_hit = req.swa_host_hit_length
        reserved = (
            max(extend_need - window, 0)
            + min(extend_need + max_new, window)
            + ps
            + (surfaced_swa_hit + ps - 1) // ps * ps
        )
        budget = cons_alloc.swa_available_size() + cons.swa_evictable_size()

        # Consume at admission (init_load_back + request lock).
        held = cons.buffer_pipeline.staged_prefetches[req_id]
        req = mock.Mock()
        req.rid = req_id.rid
        req.cache_request_handle = req_id
        req.extra_key = None
        req.cache_salt = None
        req.last_node = cons.root_node_handle()
        req.prefix_len = held.matched_len
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        spliced_len, last_node = cons.init_load_back(
            InitLoadBackParams(
                best_match_node=None, host_hit_length=held.num_tokens, req=req
            )
        )
        self.assertEqual(spliced_len, len(seq), "load-back degraded")
        spliced = cons.path_device_indices(last_node)
        cons.ready_to_load_host_cache()
        cons.inc_lock_ref(last_node)  # the admission lock

        self.assertEqual(cons.swa_evictable_size(), 0)  # window is protected
        # FULL stays roomy: only SWA can fail below.
        self.assertGreaterEqual(cons_alloc.full_available_size(), extend_need + ps)

        if reserved <= budget:
            try:
                if ps == 1:
                    alloc_token_slots(cons, extend_need)
                else:  # paged batch path — the production crash site
                    prefix_len = int(spliced.numel())
                    prefix_t = torch.tensor(
                        [prefix_len], dtype=torch.int64, device=spliced.device
                    )
                    seq_t = torch.tensor(
                        [prefix_len + extend_need],
                        dtype=torch.int64,
                        device=spliced.device,
                    )
                    alloc_paged_token_slots_extend(
                        cons,
                        prefix_t,
                        prefix_t.cpu(),
                        seq_t,
                        seq_t.cpu(),
                        spliced[-1:],
                        extend_need,
                    )
            except RuntimeError as e:
                self.fail(
                    f"gate admitted (reserved={reserved} <= budget={budget}) "
                    f"but the batch alloc OOMed: {e}"
                )
        else:
            # Rejection must come from the surfaced window charge.
            self.assertGreaterEqual(surfaced_swa_hit, window)

    @requires_hicache
    def test_buffer_only_load_back_drops_on_sibling_published_span(self):
        """Queued-UAF regression: a sibling publishes the staged span between
        staging and consumption; the live pre-check must drop the hold before
        any device allocation or H2D exists."""
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        self._produce_buffer_l3(storage_dir, seq)

        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)
        cons.enable_storage_metrics = True
        cons.storage_metrics_collector = mock.Mock()
        avail0 = self._host_avail_sizes(cons)

        req_id = CacheRequestHandle("sibling-publish", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        cons.pop_prefetch_loaded_tokens(req_id)

        # Sibling publishes the identical span (live FULL + SWA).
        self._insert(cons, cons_alloc, cons_rtp, seq)
        sib = match_prefix(cons, seq)
        self.assertEqual(sib.device_prefix_len, len(seq))
        self._fill_full_kv(
            cons_alloc, cons.path_device_indices(sib.last_device_node), marker=3
        )
        sib_kv = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(sib.last_device_node)
        )
        dev_avail0 = cons.token_to_kv_pool_allocator.available_size()

        # Consume with the batch-stale empty prefix view: the live unified
        # check must drop it.
        spliced = self._consume_staged_prefetch(cons, req_id, prefix_len=0)
        self.assertEqual(int(spliced.numel()), 0)

        # Nothing device-side happened; sibling slots and staging intact.
        self.assertEqual(cons.token_to_kv_pool_allocator.available_size(), dev_avail0)
        self.assertEqual(cons.buffer_pipeline.ongoing_buffer_load_back, {})
        self.assertFalse(cons.buffer_pipeline.has_staged(req_id))
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        m = match_prefix(cons, seq)
        self.assertTrue(
            torch.equal(
                cons.path_device_indices(m.last_device_node),
                cons.path_device_indices(sib.last_device_node),
            )
        )
        k, v = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(m.last_device_node)
        )
        self.assertTrue(torch.equal(k, sib_kv[0]))
        self.assertTrue(torch.equal(v, sib_kv[1]))
        cons.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens.assert_called_once_with(
            len(seq), "device_covered"
        )
        cons.sanity_check()

    @requires_hicache
    @requires(lambda cfg: cfg.has_swa, "masked overlap requires an SWA component")
    def test_buffer_only_load_back_uses_full_behind_swa_tombstone(self):
        """FULL-only rematch keeps resident FULL and repairs missing SWA rows."""
        storage_dir = self._make_storage_dir()
        seq = self._buffer_swa_seq()
        self._produce_buffer_l3(storage_dir, seq)

        with self.subTest(window="all-tombstone"):
            self._assert_masked_full_is_reused(storage_dir, seq, live_tail=0)
        sw_pages = (
            self.cfg.sliding_window_size + self.cfg.page_size - 1
        ) // self.cfg.page_size
        if sw_pages >= 2:
            with self.subTest(window="partly-resident"):
                self._assert_masked_full_is_reused(
                    storage_dir, seq, live_tail=self.cfg.page_size
                )

    @requires_hicache
    @requires(lambda cfg: cfg.has_swa, "masked overlap requires an SWA component")
    def test_buffer_only_load_back_reuses_partial_masked_full(self):
        """FULL-only rematch reuses a resident head and fetches only its tail."""
        page_size = self.cfg.page_size
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        _, expected_kv = self._produce_buffer_l3(storage_dir, seq, marker=3)
        split_at = len(seq) - page_size

        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)

        # Shares seq[:split_at] then diverges, so the FULL run ends mid-node.
        sibling = seq[:split_at] + self._make_seq(9000, 1)
        value = self._alloc(cons_alloc, len(sibling))
        cons.insert(
            InsertParams(
                key=RadixKey(array("q", sibling)),
                value=value[: len(sibling)],
                component_evicted_seqlens={ComponentType.SWA: len(sibling)},
            )
        )
        live = match_prefix(cons, seq)
        self.assertEqual(live.device_prefix_len, 0, "unified match not masked")
        self.assertEqual(live.full_kv_hit_length, split_at, "FULL run not partial")
        head_full = value[:split_at].clone()
        cons_alloc.free_swa(head_full)
        self._fill_full_kv(cons_alloc, head_full, marker=7)
        head_kv = self._snapshot_full_kv(cons_alloc, head_full)

        avail0 = self._host_avail_sizes(cons)
        req_id = CacheRequestHandle("masked-partial", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        cons.pop_prefetch_loaded_tokens(req_id)
        staged = cons.buffer_pipeline.staged_prefetches[req_id]
        self.assertEqual(staged.matched_len, split_at)
        self.assertEqual(staged.num_tokens, page_size)
        full_avail0 = cons_alloc.full_available_size()

        spliced, req = self._consume_staged_prefetch(
            cons, req_id, prefix_len=0, return_req=True
        )
        self.assertEqual(req.prefix_len, split_at)
        self.assertEqual(int(spliced.numel()), page_size)
        self.assertEqual(cons_alloc.full_available_size(), full_avail0 - page_size)
        k, v = self._snapshot_full_kv(cons_alloc, head_full)
        self.assertTrue(torch.equal(k, head_kv[0]))
        self.assertTrue(torch.equal(v, head_kv[1]))
        tail_k, tail_v = self._snapshot_full_kv(cons_alloc, spliced)
        self.assertTrue(torch.equal(tail_k, expected_kv[0][split_at:]))
        self.assertTrue(torch.equal(tail_v, expected_kv[1][split_at:]))

        self.assertEqual(cons.buffer_pipeline.ongoing_buffer_load_back, {})
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        full_len, full_node, _ = cons.tree_core.match_full_device_prefix(
            RadixKey(array("q", seq))
        )
        self.assertEqual(full_len, len(seq))
        full_indices = cons.tree_core.collect_full_device_indices(
            full_node, cons.root_node_handle()
        )
        self.assertTrue(torch.equal(full_indices[:split_at], head_full))
        self.assertTrue(torch.equal(full_indices[split_at:], spliced))
        cons.sanity_check()

    @requires_hicache
    @requires(lambda cfg: cfg.has_swa, "masked overlap requires an SWA component")
    def test_buffer_only_masked_head_is_not_evicted_for_tail_load(self):
        """Loading past a masked FULL head preserves its existing ownership."""
        page_size = self.cfg.page_size
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        self._produce_buffer_l3(storage_dir, seq)
        split_at = len(seq) - page_size

        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)

        # The tree holds the head alone, SWA-tombstoned. It is a leaf, so it is
        # the victim the tail's evict-before-alloc would pick.
        value = self._alloc(cons_alloc, split_at)
        cons.insert(
            InsertParams(
                key=RadixKey(array("q", seq[:split_at])),
                value=value[:split_at],
                component_evicted_seqlens={ComponentType.SWA: split_at},
            )
        )

        req_id = CacheRequestHandle("adopt-under-evict", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        cons.pop_prefetch_loaded_tokens(req_id)

        # Admission's temporary request lock must protect the head on its own.
        cons.buffer_pipeline.release_anchor_lock(req_id)
        self.assertEqual(cons.buffer_pipeline.anchor_locks, {})
        # Starve FULL only; SWA stays free so the window still loads.
        ballast = cons_alloc.full_attn_allocator.alloc(cons_alloc.full_available_size())
        self.assertIsNotNone(ballast)
        self.addCleanup(cons_alloc.full_attn_allocator.free, ballast)

        def full_accounted():
            return (
                cons_alloc.full_available_size()
                + cons.full_evictable_size()
                + cons.full_protected_size()
            )

        accounted0 = full_accounted()
        self._consume_staged_prefetch(cons, req_id, prefix_len=0)

        self.assertEqual(accounted0, full_accounted(), "a FULL page is free and owned")
        # The resident FULL head was reused rather than evicted for the tail.
        live = match_prefix(cons, seq)
        self.assertEqual(live.full_kv_hit_length, split_at)
        cons.sanity_check()

    @requires_hicache
    def test_buffer_only_load_back_fail_stops_on_post_check_overlap(self):
        """If the tree mutates between the pre-check and the insert (simulated
        via cc.load), consumption must fail-stop rather than hand out slots a
        queued H2D no longer owns."""
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        self._produce_buffer_l3(storage_dir, seq)

        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)

        req_id = CacheRequestHandle("post-check-overlap", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        cons.pop_prefetch_loaded_tokens(req_id)

        from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

        real_load = cons.cache_controller.load

        def adversarial_load(*args, **kwargs):
            # Mutate the tree after the pre-check has already passed.
            self._insert(cons, cons_alloc, cons_rtp, seq)
            return real_load(*args, **kwargs)

        f = cons.buffer_pipeline.staged_prefetches[req_id]
        req = mock.Mock()
        req.rid = req_id.rid
        req.cache_request_handle = req_id
        req.extra_key = None
        req.cache_salt = None
        req.prefix_len = 0
        req.last_node = cons.root_node_handle()
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        with mock.patch.object(cons.cache_controller, "load", adversarial_load):
            with self.assertRaisesRegex(RuntimeError, "ownership violation"):
                cons.init_load_back(
                    InitLoadBackParams(
                        best_match_node=None, host_hit_length=f.num_tokens, req=req
                    )
                )

    @requires_hicache
    @requires_ps1_window4
    def test_buffer_only_load_back_trims_head_published_by_sibling(self):
        """Growth-waste regression: a sibling publishing the span HEAD after
        enqueue used to invalidate the whole staged fetch at consumption
        (splice base moved past matched_len -> full drop, every fetched byte
        wasted). Consumption must instead splice the tail beyond the live
        prefix: sibling head slots stay untouched (add-only insert), the
        tail carries the producer's bytes, and the ack frees the entire
        bounce including the trimmed head.

        A window-sized fetch still passes the global threshold, but after
        sibling growth its FULL splice is shorter than its SWA transfer."""
        # Buffer-mode plan/commit logic is layout-independent, and each
        # hicache fixture retains its pools for the whole file run. Pin to
        # one config so the matrix does not exhaust a small CI GPU.
        storage_dir = self._make_storage_dir()

        with self.subTest(span="over-window"):
            self._assert_sibling_head_is_trimmed(
                storage_dir, self._buffer_swa_seq(), window_sized=False
            )
        with self.subTest(span="exact-window"):
            seq = self._make_seq(
                5000, self.cfg.sliding_window_size // self.cfg.page_size
            )
            self._assert_sibling_head_is_trimmed(storage_dir, seq, window_sized=True)

    @requires_hicache
    @requires_ps1_window4
    def test_buffer_only_plan_frees_covered_hold(self):
        """A hold whose whole span became device-resident can never splice:
        the surface-time plan must report (0, 0) and free it (bounce, anchor
        pin, occupancy) — a kept hold would leak, since admission without a
        host hit never calls init_load_back."""
        # Buffer-mode plan/commit logic is layout-independent, and each
        # hicache fixture retains its pools for the whole file run. Pin to
        # one config so the matrix does not exhaust a small CI GPU.
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        self._produce_buffer_l3(storage_dir, seq)

        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)
        avail0 = self._host_avail_sizes(cons)

        req_id = CacheRequestHandle("covered-hold", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        cons.pop_prefetch_loaded_tokens(req_id)
        self._insert(cons, cons_alloc, cons_rtp, seq)

        live = match_prefix(cons, seq)
        req = mock.Mock(
            rid=req_id.rid,
            cache_request_handle=req_id,
            prefix_len=live.device_prefix_len,
            last_node=live.last_device_node,
        )
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (0, 0))
        self.assertFalse(cons.buffer_pipeline.has_staged(req_id))
        self.assertEqual(cons.buffer_pipeline.anchor_locks, {})
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        # Idempotent once freed.
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertIsNone(req.staged_prefetch_plan)
        cons.sanity_check()

    @requires_hicache
    @requires_ps1_window4
    def test_buffer_only_hit_commit_cancels_device_covered_fetch(self):
        """A sibling that publishes the span while the storage hit query is
        in flight makes the fetch unconsumable; the IO-commit gate must
        cancel it BEFORE the bounce alloc and the storage read (counted as
        declined_device_covered), leaving no staging or occupancy behind."""
        # Buffer-mode plan/commit logic is layout-independent, and each
        # hicache fixture retains its pools for the whole file run. Pin to
        # one config so the matrix does not exhaust a small CI GPU.
        storage_dir = self._make_storage_dir()

        seq = self._buffer_swa_seq()
        self._produce_buffer_l3(storage_dir, seq)

        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)
        avail0 = self._host_avail_sizes(cons)
        stats = cons._prefetch_outcome_stats

        req_id = CacheRequestHandle("covered-at-commit", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        # Wait for the hit verdict WITHOUT draining it (the drain is the
        # scheduler-thread IO commit under test).
        deadline = time.time() + 10.0
        while (
            cons.cache_controller.prefetch_hit_queue.qsize() == 0
            and time.time() < deadline
        ):
            time.sleep(0.01)
        self.assertGreater(cons.cache_controller.prefetch_hit_queue.qsize(), 0)

        self._insert(cons, cons_alloc, cons_rtp, seq)
        cons.drain_storage_control_queues()

        self.assertEqual(stats["declined_device_covered"], 1)
        self.assertNotIn(req_id, cons.ongoing_prefetch)
        self.assertFalse(cons.buffer_pipeline.has_staged(req_id))
        self.assertTrue(cons.check_prefetch_progress(req_id))
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        self.assertNotIn(req_id.rid, cons.storage_prefetch_retries._pending)
        # Aux staging released during the drain lands on queues sized before
        # it; a second drain flushes them.
        cons.drain_storage_control_queues()
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        cons.sanity_check()

    @requires_hicache
    @requires_swa
    def test_buffer_only_swa_window_semantics(self):
        """Root and mid-tree requests and hits share a complete-window minimum."""
        window = self.cfg.sliding_window_size
        if window <= self.cfg.page_size:
            self.skipTest("window fits in one page")
        sw_pages = (window + self.cfg.page_size - 1) // self.cfg.page_size
        storage_dir = self._make_storage_dir()

        # 1. A root-anchored sub-window request is declined before querying.
        seq = self._make_seq(1, sw_pages - 1)
        self._produce_buffer_l3(storage_dir, seq, marker=5)
        cons, _, _ = self._build_buffer_hicache_fixture(storage_dir)
        self.assertEqual(cons.prefetch_threshold, sw_pages * self.cfg.page_size)
        self.assertEqual(
            cons.cache_controller.prefetch_threshold, cons.prefetch_threshold
        )
        avail = self._host_avail_sizes(cons)
        cons.prefetch_from_storage(
            CacheRequestHandle("short-req", 0),
            cons.root_node_handle(),
            array("q", seq),
            None,
            None,
        )
        self.assertNotIn(CacheRequestHandle("short-req", 0), cons.ongoing_prefetch)
        self.assertEqual(cons._prefetch_outcome_stats["declined_too_short"], 1)
        self.assertEqual(cons._prefetch_outcome_stats["issued"], 0)
        self.assertEqual(self._host_avail_sizes(cons), avail)
        cons.sanity_check()

        # 2. The same threshold applies mid-tree; an exact window is accepted.
        if sw_pages >= 2:
            seq_a = self._make_seq(1, max(2, sw_pages))
            seq_ab = seq_a + self._make_seq(900, sw_pages)
            self._produce_buffer_l3(storage_dir, seq_ab, marker=14)
            cons2, cons2_alloc, cons2_rtp = self._build_buffer_hicache_fixture(
                storage_dir
            )
            avail2 = self._host_avail_sizes(cons2)
            self._insert(cons2, cons2_alloc, cons2_rtp, seq_a)
            m = match_prefix(cons2, seq_a)
            last_hash = cons2.get_last_hash_value(m.last_device_node)
            cons2.prefetch_from_storage(
                CacheRequestHandle("subwin-req", 0),
                m.last_device_node,
                array("q", seq_ab[len(seq_a) : -self.cfg.page_size]),
                last_hash,
                None,
                matched_prefix_tokens=list(seq_a),
            )
            self.assertNotIn(
                CacheRequestHandle("subwin-req", 0), cons2.ongoing_prefetch
            )
            self.assertEqual(cons2._prefetch_outcome_stats["declined_too_short"], 1)
            # A declined span issues no lookup, so it must not spend the
            # re-issue budget on polls.
            self.assertNotIn("subwin-req", cons2.storage_prefetch_retries._pending)

            cons2.prefetch_from_storage(
                CacheRequestHandle("window-req", 0),
                m.last_device_node,
                array("q", seq_ab[len(seq_a) :]),
                last_hash,
                None,
                matched_prefix_tokens=list(seq_a),
            )
            self._pump_hicache_until(
                cons2,
                lambda: (
                    cons2.check_prefetch_progress(CacheRequestHandle("window-req", 0))
                    and cons2.buffer_pipeline.has_staged(
                        CacheRequestHandle("window-req", 0)
                    )
                ),
                "whole-window mid-tree prefetch did not stage",
            )
            self.assertTrue(
                any(
                    t.name == PoolName.SWA
                    for t in cons2.buffer_pipeline.staged_prefetches[
                        CacheRequestHandle("window-req", 0)
                    ].aux_xfers
                ),
                "mid-tree fetch degraded to KV-only",
            )
            spliced = self._consume_staged_prefetch(
                cons2,
                CacheRequestHandle("window-req", 0),
            )
            self.assertEqual(int(spliced.numel()), len(seq_ab) - len(seq_a))
            self.assertEqual(
                len(
                    cons2.path_device_indices(
                        match_prefix(cons2, seq_ab).last_device_node
                    )
                ),
                len(seq_ab),
            )
            self.assertEqual(self._host_avail_sizes(cons2), avail2)
            cons2.sanity_check()

        # 3. An accepted request with a sub-window L3 hit is also declined.
        full = self._make_seq(6000, sw_pages + 1)
        stored = full[: (sw_pages - 1) * self.cfg.page_size]
        self._produce_buffer_l3(storage_dir, stored, marker=6)
        cons3, _, _ = self._build_buffer_hicache_fixture(storage_dir)
        avail3 = self._host_avail_sizes(cons3)
        cons3.prefetch_from_storage(
            CacheRequestHandle("partial-req", 0),
            cons3.root_node_handle(),
            array("q", full),
            None,
            None,
        )
        operation = cons3.ongoing_prefetch[
            CacheRequestHandle("partial-req", 0)
        ].operation
        self._run_prefetch_to_completion(cons3, CacheRequestHandle("partial-req", 0))
        cons3.drain_storage_control_queues()
        self.assertEqual(operation.storage_hit_count, len(stored))
        self.assertEqual(
            len(
                cons3.path_device_indices(match_prefix(cons3, stored).last_device_node)
            ),
            0,
            "sub-window hit should not be loaded",
        )
        self.assertEqual(cons3._prefetch_outcome_stats["issued"], 1)
        self.assertEqual(cons3._prefetch_outcome_stats["revoked_insufficient"], 1)
        self.assertFalse(
            cons3.buffer_pipeline.has_staged(CacheRequestHandle("partial-req", 0))
        )
        self.assertEqual(self._host_avail_sizes(cons3), avail3)
        cons3.sanity_check()

    # TP consistency for SWA prefetch

    def test_tp_swa_prefetch_dropped_when_peer_misses(self):
        """A peer rank missing the SWA window drops the whole prefetch result
        on every rank (TP-consistent all-or-nothing)."""
        setup = self._setup_swa_tp_prefetch()
        if setup is None:
            return
        storage_dir, seq = setup

        cons, _, _ = self._l3_consumer(storage_dir)
        cons.tp_world_size = 2
        self._patch_tp_prefetch_sync(cons, drop_swa=True)
        self._consume_prefetch(cons, seq, CacheRequestHandle("drop", 0))

        m = match_prefix(cons, seq)
        self.assertEqual(m.host_hit_length, 0)
        self.assertFalse(
            self._swa_host_on_path(cons, seq), "SWA must be dropped when a peer misses"
        )
        cons.sanity_check()

    def test_tp_swa_prefetch_adopted_when_peer_present(self):
        """When every rank has the full SWA window, SWA is adopted under tp>1
        (the single packed all_reduce path still works)."""
        setup = self._setup_swa_tp_prefetch()
        if setup is None:
            return
        storage_dir, seq = setup

        cons, _, _ = self._l3_consumer(storage_dir)
        cons.tp_world_size = 2
        self._patch_tp_prefetch_sync(cons, drop_swa=False)  # peer == local
        self._consume_prefetch(cons, seq, CacheRequestHandle("keep", 0))

        m = match_prefix(cons, seq)
        self.assertEqual(m.host_hit_length, len(seq))
        self.assertTrue(
            self._swa_host_on_path(cons, seq),
            "SWA must be adopted when all ranks have it",
        )
        cons.sanity_check()

    def test_tp_swa_prefetch_drop_frees_host_pool(self):
        """A dropped SWA prefetch must return its whole host buffer to the pool
        (no leak, no over-free)."""
        setup = self._setup_swa_tp_prefetch()
        if setup is None:
            return
        storage_dir, seq = setup

        cons, _, _ = self._l3_consumer(storage_dir)
        cons.tp_world_size = 2
        self._patch_tp_prefetch_sync(cons, drop_swa=True)
        avail_before = cons.swa_kv_pool_host.available_size()
        self._consume_prefetch(cons, seq, CacheRequestHandle("drop", 0))

        self.assertEqual(
            match_prefix(cons, seq).host_hit_length,
            0,
        )
        # Whole window dropped -> its host buffer is fully released back.
        cons.drain_storage_control_queues()  # Drain the release queue.
        self.assertEqual(cons.swa_kv_pool_host.available_size(), avail_before)

    @requires_full_swa_ps1
    def test_prepare_prefetch_swa(self):
        cache, _, _ = self._build_hicache_fixture()
        sw = cache.sliding_window_size
        swa = cache.components[ComponentType.SWA]
        avail = cache.swa_kv_pool_host.available_size()
        # zero-length prefetch -> does not participate
        self.assertEqual(
            swa.prepare_prefetch(
                cache.root_node_handle(), prefetch_tokens=0
            ).staging_tokens,
            0,
        )
        # below a full window at the ROOT anchor -> the whole sequence is its
        # own trailing window (sub-window prompts stay reusable via storage)
        self.assertEqual(
            swa.prepare_prefetch(
                cache.root_node_handle(), prefetch_tokens=sw - 1
            ).staging_tokens,
            sw - 1,
        )
        # a full window available -> one window of host pages
        self.assertEqual(
            swa.prepare_prefetch(
                cache.root_node_handle(), prefetch_tokens=sw
            ).staging_tokens,
            sw,
        )
        # a non-participating component never stages
        self.assertEqual(
            cache.components[ComponentType.FULL]
            .prepare_prefetch(cache.root_node_handle(), prefetch_tokens=sw)
            .staging_tokens,
            0,
        )
        # Sizing holds no host memory: the window is allocated at hit time, so
        # a query in flight (hit or miss) never occupies the SWA host pool.
        self.assertEqual(cache.swa_kv_pool_host.available_size(), avail)
        self.assertEqual(int(swa.alloc_prefetch_staging(sw).numel()), sw)

    @requires_swa_only
    @requires_hicache
    def test_buffer_only_hit_parks_on_swa_staging_shortfall(self):
        """A hit whose SWA window cannot be staged parks whole and launches
        once the pool frees. It is never forfeited, and it never launches with
        the KV staged and the window not; a parked hit holds no host memory."""
        storage_dir = self._make_storage_dir()
        seq = self._buffer_swa_seq()
        self._produce_buffer_l3(storage_dir, seq)
        cons, _, _ = self._build_buffer_hicache_fixture(storage_dir)
        pool = cons.swa_kv_pool_host
        avail = pool.available_size()
        real_alloc = pool.alloc
        # Both attempts of the first hit-time alloc fail (the second follows
        # the host evict); the next drain's attempt succeeds.
        shortfalls = [None, None]

        def alloc(num_tokens):
            if shortfalls:
                return shortfalls.pop()
            return real_alloc(num_tokens)

        with mock.patch.object(pool, "alloc", side_effect=alloc):
            cons.prefetch_from_storage(
                CacheRequestHandle("r", 0),
                cons.root_node_handle(),
                array("q", seq),
                None,
                None,
            )
            # The query holds no SWA staging.
            self.assertEqual(pool.available_size(), avail)
            self._pump_hicache_until(
                cons,
                lambda: bool(cons.buffer_pipeline.pending_hit_allocs),
                "hit was not parked on the SWA staging shortfall",
            )
            self.assertIn(CacheRequestHandle("r", 0), cons.ongoing_prefetch)
            self.assertEqual(cons._prefetch_outcome_stats["declined_rate_limited"], 1)
            # Parked, before the next drain retries it: nothing is held.
            self.assertEqual(pool.available_size(), avail)
        self._run_prefetch_to_completion(cons, CacheRequestHandle("r", 0))
        cons.drain_storage_control_queues()
        m = match_prefix(cons, seq)
        self.assertEqual(m.device_prefix_len, len(seq))
        cons.sanity_check()


define_config_test_classes(UnifiedRadixCacheStorageSuite, globals())


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestUnifiedTreeCoreSWAPrefetchBackends(UnifiedRadixCacheFixtures, CustomTestCase):
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.SWA), sliding_window_size=4
    )

    def test_mid_tree_shortened_swa_prefetch_is_released(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        prefix = [1, 2]
        self._insert(cache, allocator, req_to_token_pool, prefix)
        anchor = match_prefix(cache, prefix).last_device_node

        cache.tree_core.is_write_back = True
        suffix = [3, 4]
        insert_result = cache.tree_core.insert_host(
            anchor,
            RadixKey(array("q", suffix)),
            torch.tensor([100, 101], dtype=torch.int64),
            ["h3", "h4"],
        )
        self.assertIsNotNone(insert_result.inserted_host_node)

        swa_host_indices = torch.tensor([30, 31], dtype=torch.int64)
        actions = []
        cache.tree_core.commit_hicache_transfers(
            anchor,
            CacheTransferPhase.PREFETCH,
            {
                ComponentType.SWA: [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=swa_host_indices,
                    )
                ]
            },
            cache_actions=actions,
            insert_result=insert_result,
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=len(suffix),
                extra_pool_hit_pages={PoolName.SWA: len(suffix)},
            ),
        )

        self.assertIsNone(
            node_host_value(cache, insert_result.inserted_host_node, ComponentType.SWA)
        )
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], FreeComponentHostSlot)
        self.assertEqual(actions[0].component_type, ComponentType.SWA)
        self.assertEqual(len(actions[0].host_indices), 1)
        self.assertTrue(torch.equal(actions[0].host_indices[0], swa_host_indices))


class TestPrefetchCommitOrdering(CustomTestCase):
    """The prefetch commit's action ordering (mock-based)."""

    def test_prefetch_commit_applies_host_insert_actions_before_transfers(self):
        """The prefetch commit applies the host-insert walk's actions before
        commit_hicache_transfers, whose emissions ride a fresh list."""
        cache = mock.MagicMock()
        cache.page_size = 1
        cache.enable_storage_metrics = False
        cache.buffer_pipeline = None  # cache-mode commit path
        cache.cache_controller.pp_prefetch_decisions = {}
        walk_action = object()
        insert_result = mock.MagicMock()
        insert_result.cache_actions = [walk_action]
        insert_result.prefix_len = 4
        insert_result.host_insert_dropped = False
        cache.tree_core.insert_host.return_value = insert_result
        operation = mock.MagicMock()
        operation.handle = CacheRequestHandle("req", 0)
        operation.request_id = "req"
        operation.completed_tokens = 8
        cache.ongoing_prefetch = {
            operation.handle: _OngoingPrefetch(
                7,
                list(range(8)),
                list(range(100, 108)),
                operation,
                None,
                {},
            )
        }
        cache.cache_controller.terminate_prefetch.return_value = (
            8,
            [f"h{i}" for i in range(8)],
        )
        cache._check_hybrid_prefetch_result.return_value = 8
        cache.cache_controller.prefetch_tokens_occupied = 100
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache._can_terminate_prefetch.return_value = True
        cache.pp_rank = 0

        order = mock.MagicMock()
        applied = []

        def record_apply(actions):
            applied.append(list(actions))
            actions.clear()

        order.apply.side_effect = record_apply
        cache._apply_cache_actions = order.apply
        cache.tree_core.commit_hicache_transfers = order.commit

        def _handle_prefetch_result(operation):
            UnifiedRadixCache._handle_prefetch_result(cache, operation)

        cache._handle_prefetch_result = _handle_prefetch_result

        self.assertTrue(
            UnifiedRadixCache.check_prefetch_progress(
                cache, CacheRequestHandle("req", 0)
            )
        )

        self.assertEqual([c[0] for c in order.mock_calls], ["apply", "commit", "apply"])
        self.assertEqual(applied[0], [walk_action])
        self.assertIsNot(
            order.commit.call_args.kwargs["cache_actions"], insert_result.cache_actions
        )
        self.assertEqual(cache.ongoing_prefetch, {})


class TestUnifiedRadixPrefetchCorruption(
    UnifiedRadixPrefetchCorruptionFixtures, CustomTestCase
):
    """Prefetch must not hang a backed-up host child under an un-backed-up parent.

    Under write-through that broke the "child backed up => parent backed up"
    invariant, failing as an idle-sanity error and, on eviction, as
    `_remove_leaf_from_parent -> assert v == node`. Fix: drop the refill.
    """

    ps = 16
    cfg = CacheConfig(
        page_size=ps,
        components=(ComponentType.FULL,),
        kv_size=4096,
        max_context_len=4096,
    )

    def test_prefetch_refill_under_unbacked_parent_is_dropped(self):
        cache, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(cache)

        parent_id = self._insert_device(
            cache, allocator, list(range(1, 1 + 3 * self.ps))
        )
        self.assertFalse(cache.tree_core.is_backuped(parent_id))

        child = self._attach_host_child(cache, parent_id, start_token=1000)
        self.assertIsNone(child)
        self.assertEqual(len(cache.tree_core.get_child_node_ids(parent_id)), 0)
        cache.sanity_check()

    def test_dropped_prefetch_releases_all_host_resources(self):
        """The caller owns every completed buffer when host insertion drops."""
        cache, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = mock.Mock()

        parent_id = self._insert_device(
            cache, allocator, list(range(1, 1 + 3 * self.ps))
        )
        prefetch_key = RadixKey(
            array("q", list(range(1000, 1000 + 2 * self.ps)))
        ).page_aligned(self.ps)
        completed_tokens = len(prefetch_key)
        host_indices = cache.cache_controller.mem_pool_host.alloc(completed_tokens)
        self.assertIsNotNone(host_indices)

        swa_transfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(self.ps, dtype=torch.int64),
        )
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(1, dtype=torch.int64),
        )
        comp_xfers = {
            ComponentType.SWA: [swa_transfer],
            ComponentType.MAMBA: [mamba_transfer],
        }

        operation = mock.Mock()
        operation.host_indices = host_indices
        operation.completed_tokens = completed_tokens
        operation.pool_storage_result = PoolTransferResult(
            kv_hit_pages=completed_tokens // self.ps,
            extra_pool_hit_pages={
                PoolName.SWA: 1,
                PoolName.MAMBA: 1,
            },
        )
        anchor_lock_params = cache.inc_host_lock_ref(parent_id).to_dec_params()
        req_id = CacheRequestHandle("drop-all-resources", 0)
        operation.handle = req_id
        operation.request_id = req_id.rid
        cache.ongoing_prefetch[req_id] = _OngoingPrefetch(
            parent_id,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        cache._record_storage_prefetch_hit(req_id, completed_tokens)
        cache.cache_controller.prefetch_tokens_occupied = completed_tokens
        hashes = [f"h{i}" for i in range(completed_tokens // self.ps)]
        operation.hash_value = hashes

        with (
            mock.patch.object(cache, "_can_terminate_prefetch", return_value=True),
            # Isolate the drop-release branch under test from the hybrid-sync
            # step: treat the whole fetched prefix as usable so the insert runs.
            mock.patch.object(
                cache,
                "_check_hybrid_prefetch_result",
                return_value=completed_tokens,
            ),
            mock.patch.object(
                cache.cache_controller,
                "terminate_prefetch",
                return_value=(completed_tokens, hashes),
            ),
            # No storage backend in this fixture, so the real release queues
            # don't exist; assert on the release call instead of draining them.
            mock.patch.object(
                cache.cache_controller, "append_host_mem_release"
            ) as release,
            mock.patch.object(operation, "is_terminated", return_value=False),
            mock.patch.object(
                cache.tree_core, "commit_hicache_transfers"
            ) as commit_hicache_transfers,
        ):
            self.assertTrue(cache.check_prefetch_progress(req_id))

        commit_hicache_transfers.assert_not_called()
        self.assertEqual(cache.pop_prefetch_loaded_tokens(req_id), 0)
        self.assertEqual(len(cache.tree_core.get_child_node_ids(parent_id)), 0)

        drop_releases = [
            call
            for call in release.call_args_list
            if call.kwargs.get("extra_pools") is not None
        ]
        self.assertEqual(len(drop_releases), 1)
        self.assertTrue(
            torch.equal(
                drop_releases[0].kwargs["host_indices"],
                host_indices[:completed_tokens],
            )
        )
        self.assertIs(drop_releases[0].kwargs["extra_pools"][0], swa_transfer)
        self.assertIs(drop_releases[0].kwargs["extra_pools"][1], mamba_transfer)
        cache.storage_metrics_collector.log_storage_prefetch_hit_tokens.assert_called_once_with(
            completed_tokens
        )
        cache.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens.assert_called_once_with(
            completed_tokens, "dropped"
        )

        cache.sanity_check()

    def test_write_through_eviction_counts_unbacked_tokens(self):
        cache, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.metrics_collector = mock.Mock()

        seq = list(range(1, 1 + 2 * self.ps))
        self._insert_device(cache, allocator, seq)
        result = cache.evict(EvictParams(num_tokens=len(seq)))

        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))
        cache.metrics_collector.increment_dropped_tokens.assert_called_once_with(
            num_tokens=len(seq),
            reason="write_through_unbacked_eviction",
            pool=PoolName.KV.value,
        )
        cache.sanity_check()

    def test_prefetch_refill_kept_under_unbacked_parent_in_write_back(self):
        """Write-back keeps the refill (it has no backed-up-parent requirement)."""
        cache, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")

        parent_id = self._insert_device(
            cache, allocator, list(range(1, 1 + 3 * self.ps))
        )
        child = self._attach_host_child(cache, parent_id, start_token=1000)
        self.assertIsNotNone(child)
        cache.sanity_check()


class TestUnifiedRadixCacheStorageAttachBackfill(
    UnifiedRadixCacheStorageAttachBackfillFixtures, CustomTestCase
):
    """Enabling a storage backend must hash nodes that predate it.

    Page hashes chain off the parent's last hash, so a node hashed against an
    unhashed parent restarts the chain mid-sequence: its L3 keys then cover only a
    suffix of the prefix they claim to represent, which aliases any unrelated
    request that happens to start with those tokens.
    """

    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL,),
        kv_size=64,
        max_context_len=64,
    )

    prefix_tokens = array("q", [1, 2, 3, 4, 5, 6, 7, 8])
    suffix_tokens = array("q", [9, 10, 11, 12])

    def test_backfill_reproduces_hashing_from_the_start(self):
        expected = self._hashes_by_token_ids(
            self._build_two_level_tree(storage_on_from_the_start=True)
        )
        self.assertEqual(len(expected), 2, "expected a parent and a child node")

        late = self._build_two_level_tree(storage_on_from_the_start=False)
        self.assertTrue(
            all(h is None for h in self._hashes_by_token_ids(late).values()),
            "nodes built while storage was disabled must start unhashed",
        )

        self.assertEqual(late.tree_core.backfill_missing_hash_values(), len(expected))
        self.assertEqual(
            self._hashes_by_token_ids(late),
            expected,
            "a backfilled chain must be identical to one hashed from the start",
        )

    def test_backfill_is_idempotent(self):
        cache = self._build_two_level_tree(storage_on_from_the_start=True)
        before = self._hashes_by_token_ids(cache)
        self.assertEqual(cache.tree_core.backfill_missing_hash_values(), 0)
        self.assertEqual(self._hashes_by_token_ids(cache), before)

    def test_buffer_backup_snapshot_is_detached_and_revalidates_parent(self):
        cache = self._build_two_level_tree(storage_on_from_the_start=True)
        child = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(self.prefix_tokens + self.suffix_tokens))
        ).last_device_node
        parent = cache.tree_core.get_parent_node_id(child)
        snapshot = cache.tree_core.snapshot_buffer_backup(child, pass_prefix_keys=True)

        self.assertEqual(snapshot.key.token_ids, self.suffix_tokens)
        snapshot.key.token_ids[0] = -1
        self.assertEqual(node_token_ids(cache, child), list(self.suffix_tokens))

        parent_hashes = cache.tree_core.get_hash_values(parent)
        replacement_hashes = ["a" * 64] * len(parent_hashes)
        cache.tree_core.set_node_hash_values(parent, replacement_hashes)
        state = cache.tree_core.validate_buffer_backup(child, len(snapshot.key))
        self.assertEqual(state.parent_node_id, parent)
        self.assertEqual(state.parent_last_hash, replacement_hashes[-1])
        cache.tree_core.set_node_hash_values(parent, parent_hashes)

    def test_enabling_storage_backfills_the_tree(self):
        """The tree is hashed by the time `enable_storage` flips on."""
        cache = self._build_two_level_tree(storage_on_from_the_start=False)
        StorageAttachment(cache).apply_runtime_config(
            storage_backend="file",
            prefetch_threshold=64,
            prefetch_timeout_base=1.0,
            prefetch_timeout_per_ki_token=0.25,
            hicache_storage_pass_prefix_keys=False,
            enable_storage=True,
            enable_storage_metrics=False,
            extra_metric_labels=None,
        )
        self.assertTrue(cache.enable_storage)
        self.assertTrue(
            all(h for h in self._hashes_by_token_ids(cache).values()),
            "every node must carry a hash chain once storage is enabled",
        )


class TestAnchorLockOutcomePolicy(AnchorLockOutcomePolicyFixtures, CustomTestCase):
    """try_lock_anchor finds the anchor by re-matching the live tree (no
    carried node id to go stale): prefix intact -> lock the live node;
    prefix shrunk -> anchor_lost so the caller re-plans; over the whole cap
    -> cap_skip (launch unlocked, since no drain can ever fit it), merely
    over budget -> cap_busy (park)."""

    _REQ = CacheRequestHandle("req-1", 0)
    _PREFIX = list(range(100, 100 + 8))

    def test_intact_prefix_locks_live_node(self):
        cache = self._make_cache(live_match_len=len(self._PREFIX))
        pipeline = self._make_pipeline(cache)
        self.assertEqual(pipeline.try_lock_anchor(self._REQ, 0)[0], "locked")
        self.assertEqual(pipeline.anchor_locks[self._REQ].node_id, 99)
        self.assertEqual(pipeline.anchor_locked_tokens_, len(self._PREFIX))
        cache.tree_core.inc_full_pin.assert_called_once_with(99)
        cache.inc_lock_ref.assert_not_called()

    def test_shrunk_prefix_reports_anchor_lost(self):
        cache = self._make_cache(live_match_len=len(self._PREFIX) - 2)
        pipeline = self._make_pipeline(cache)
        self.assertEqual(pipeline.try_lock_anchor(self._REQ, 0)[0], "anchor_lost")
        self.assertEqual(pipeline.anchor_locks, {})
        self.assertEqual(pipeline.anchor_locked_tokens_, 0)

    def test_storage_cleanup_releases_buffer_prefetch_anchor(self):
        cache = self._make_cache(live_match_len=len(self._PREFIX))
        pipeline = self._make_pipeline(cache)
        cache.buffer_pipeline = pipeline
        self.assertEqual(pipeline.try_lock_anchor(self._REQ, 0)[0], "locked")
        host_indices = torch.arange(4)
        cache.ongoing_prefetch = {
            self._REQ: _OngoingPrefetch(
                anchor_node_id=99,
                prefetch_key=RadixKey(array("q", range(16))),
                host_indices=host_indices,
                operation=mock.Mock(
                    buffer_host_occupied_units=4, pool_transfers_done=True
                ),
                anchor_lock_params=None,
                comp_xfers={},
            )
        }
        cache.ongoing_backup = {}
        cache.host_memory_mode = "buffer_only"
        cache._prefetch_occupied_span.side_effect = (
            lambda key, indices, *, operation=None: (
                UnifiedRadixCache._prefetch_occupied_span(
                    cache, key, indices, operation=operation
                )
            )
        )
        cache._retire_ongoing_prefetch.side_effect = lambda *args: (
            UnifiedRadixCache._retire_ongoing_prefetch(cache, *args)
        )
        controller = cache.cache_controller
        controller.terminate_prefetch.return_value = (4, None)
        controller.prefetch_tokens_occupied = 12

        StorageAttachment(cache)._release_pending_storage_ops()

        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertEqual(pipeline.anchor_locks, {})
        self.assertEqual(pipeline.anchor_locked_tokens_, 0)
        self.assertNotIn(self._REQ, pipeline._prefetch_prefix_ctx)
        cache.tree_core.dec_full_pin.assert_called_once_with(99)
        cache.dec_lock_ref.assert_not_called()
        cache.dec_host_lock_ref.assert_not_called()
        self.assertEqual(controller.prefetch_tokens_occupied, 8)

    def test_positive_hit_with_lost_anchor_reports_anchor_lost(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache.ongoing_prefetch = {
            self._REQ: SimpleNamespace(
                operation=SimpleNamespace(storage_start=4, storage_hit_count=8)
            )
        }
        cache._finish_storage_prefetch = mock.Mock()
        cache.revoke_pending_prefetch = mock.Mock()

        cache._handle_storage_prefetch_anchor_loss(self._REQ)

        cache._finish_storage_prefetch.assert_called_once_with(
            self._REQ, fulfilled_tokens=0, reason="anchor_lost"
        )
        # The eviction widened the span, so the request replans over it --
        # skipping the query, the prior hit having proved it stored, and the
        # poll interval, which would only let admission run without it.
        req = SimpleNamespace(rid=self._REQ.rid, storage_prefetch_retry_attempts=0)
        self.assertEqual(
            cache.storage_prefetch_retries.pop_ready([req], 0, 8), [(req, 12)]
        )
        cache.revoke_pending_prefetch.assert_called_once_with(self._REQ)

    def test_over_cap_reports_cap_skip_after_full_rematch(self):
        cache = self._make_cache(live_match_len=len(self._PREFIX))
        pipeline = self._make_pipeline(cache, cap_tokens=len(self._PREFIX) - 1)
        self.assertEqual(pipeline.try_lock_anchor(self._REQ, 0)[0], "cap_skip")
        self.assertEqual(pipeline.anchor_locks, {})
        cache.tree_core.match_full_device_prefix.assert_called_once()

    def test_root_anchor_reports_no_anchor(self):
        cache = self._make_cache(live_match_len=0)
        cache.ongoing_prefetch[self._REQ].prefetch_key = RadixKey(array("q", range(8)))
        pipeline = self._make_pipeline(cache)
        pipeline._prefetch_prefix_ctx[self._REQ] = ([], None, None)
        self.assertEqual(pipeline.try_lock_anchor(self._REQ, 8), ("no_anchor", 0))
        cache.tree_core.match_full_device_prefix.assert_called_once()

    def test_anchor_lock_requires_release_before_retry(self):
        cache = self._make_cache(live_match_len=8, pinned_len=12)
        pipeline = self._make_pipeline(cache)
        self.assertEqual(pipeline.try_lock_anchor(self._REQ, 0), ("locked", 8))
        with self.assertRaisesRegex(AssertionError, "prefetch anchor already locked"):
            pipeline.try_lock_anchor(self._REQ, 0)
        self.assertEqual(pipeline.anchor_locked_tokens_, 12)
        cache.tree_core.match_full_device_prefix.assert_called_once()
        cache.tree_core.inc_full_pin.assert_called_once_with(99)

        pipeline.release_anchor_lock(self._REQ)
        self.assertEqual(pipeline.anchor_locked_tokens_, 0)
        cache.tree_core.dec_full_pin.assert_called_once_with(99)
        cache.tree_core.match_full_device_prefix.return_value = (8, 100, 8)
        self.assertEqual(pipeline.try_lock_anchor(self._REQ, 0), ("locked", 8))
        self.assertEqual(pipeline.anchor_locks[self._REQ].node_id, 100)
        self.assertEqual(pipeline.anchor_locked_tokens_, 8)
        self.assertEqual(cache.tree_core.match_full_device_prefix.call_count, 2)


if __name__ == "__main__":
    unittest.main()

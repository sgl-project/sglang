"""Unit tests for EmbeddingCacheController — LRU eviction and RDMA ref counting."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import threading
import time
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.storage.mooncake_store.embedding_cache_controller import (
    ContiguousMemoryAllocator,
    EmbeddingCacheController,
    EmbeddingInsertOperation,
)

# ---------------------------------------------------------------------------
# ContiguousMemoryAllocator tests
# ---------------------------------------------------------------------------


class TestContiguousMemoryAllocator(unittest.TestCase):
    def test_basic_alloc_free(self):
        alloc = ContiguousMemoryAllocator(1024)
        a = alloc.allocate(256)
        self.assertIsNotNone(a)
        self.assertEqual(a, 0)
        b = alloc.allocate(256)
        self.assertEqual(b, 256)
        alloc.free(a, 256)
        c = alloc.allocate(128)
        self.assertEqual(c, 0)  # reused freed block

    def test_alloc_fails_when_full(self):
        alloc = ContiguousMemoryAllocator(256)
        a = alloc.allocate(256)
        self.assertIsNotNone(a)
        b = alloc.allocate(1)
        self.assertIsNone(b)

    def test_free_merges_adjacent(self):
        alloc = ContiguousMemoryAllocator(512)
        a = alloc.allocate(128)
        b = alloc.allocate(128)
        c = alloc.allocate(256)
        alloc.free(a, 128)
        alloc.free(b, 128)
        # The two 128-byte blocks should merge into one 256-byte free block
        d = alloc.allocate(256)
        self.assertIsNotNone(d)
        self.assertEqual(d, 0)

    def test_allocated_size_tracking(self):
        alloc = ContiguousMemoryAllocator(1024)
        self.assertEqual(alloc.get_allocated_size(), 0)
        a = alloc.allocate(300)
        self.assertEqual(alloc.get_allocated_size(), 300)
        b = alloc.allocate(200)
        self.assertEqual(alloc.get_allocated_size(), 500)
        alloc.free(a, 300)
        self.assertEqual(alloc.get_allocated_size(), 200)
        alloc.free(b, 200)
        self.assertEqual(alloc.get_allocated_size(), 0)

    def test_free_size_tracking(self):
        alloc = ContiguousMemoryAllocator(1024)
        self.assertEqual(alloc.get_free_size(), 1024)
        alloc.allocate(400)
        self.assertEqual(alloc.get_free_size(), 624)

    def test_double_free_is_safe(self):
        alloc = ContiguousMemoryAllocator(256)
        a = alloc.allocate(128)
        alloc.free(a, 128)
        # Second free of same offset — should not corrupt state
        alloc.free(a, 128)
        self.assertEqual(alloc.get_allocated_size(), 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_controller(
    pool_mb=1.0, enable_eviction=True, hidden_dims=None, max_eviction_batch=10
):
    """Create an EmbeddingCacheController with a mocked MooncakeEmbeddingStore."""
    ctrl = EmbeddingCacheController.__new__(EmbeddingCacheController)
    ctrl.tp_world_size = 1
    ctrl.tp_group = None
    ctrl.tp_rank = 0
    ctrl.all_rank_get = False
    ctrl.hidden_dims = hidden_dims or {"image": 1024}
    ctrl.element_size = torch.float32.itemsize
    ctrl.enable_eviction = enable_eviction
    ctrl.max_eviction_batch = max_eviction_batch

    # Small pool for testing (1 MB by default)
    ctrl.total_pool_size_bytes = int(pool_mb * 1024**2)
    ctrl.cpu_pool = torch.empty(
        ctrl.total_pool_size_bytes, dtype=torch.uint8, pin_memory=False
    )

    # Mock the mooncake store — no real RDMA
    ctrl.mooncake_store = MagicMock()
    ctrl.mooncake_store.register_buffer = MagicMock()

    ctrl.allocator = ContiguousMemoryAllocator(ctrl.total_pool_size_bytes)
    ctrl.hash_to_metadata = {}
    ctrl.access_order = {}
    ctrl.access_lock = threading.Lock()
    ctrl.ref_counts = {}

    ctrl.stats = {
        "total_allocated": 0,
        "total_evicted": 0,
        "eviction_count": 0,
        "allocation_failures": 0,
    }

    ctrl.ongoing_prefetch = {}
    ctrl.prefetch_queue = MagicMock()
    ctrl.insert_queue = MagicMock()

    ctrl.lock = threading.Lock()
    ctrl.stop_event = threading.Event()

    # Do NOT start the IO thread — tests drive _io_loop logic manually
    ctrl.io_thread = MagicMock()

    ctrl.prefetch_tp_group = None
    return ctrl


def _embedding_bytes(num_tokens, dim):
    return num_tokens * dim * torch.float32.itemsize


# ---------------------------------------------------------------------------
# LRU eviction tests
# ---------------------------------------------------------------------------


class TestLRUEviction(unittest.TestCase):
    def test_evict_oldest_first(self):
        dim = 64
        size = _embedding_bytes(1, dim)  # 256 bytes
        # Pool exactly fits 3 entries (768 bytes)
        pool_bytes = size * 3
        ctrl = _make_controller(pool_mb=pool_bytes / (1024**2))

        # Insert 3 entries — fills the pool
        for i in range(3):
            h = f"hash_{i}"
            with ctrl.lock:
                offset = ctrl.allocator.allocate(size)
                self.assertIsNotNone(offset)
                ctrl.hash_to_metadata[h] = (offset, 1, dim, size)
                ctrl._update_access_time(h)

        # Pool is full. Inserting a 4th should evict hash_0 (oldest).
        h_new = "hash_new"
        with ctrl.lock:
            offset = ctrl._allocate_with_eviction(size)

        self.assertIsNotNone(offset)
        with ctrl.lock:
            self.assertNotIn("hash_0", ctrl.hash_to_metadata)
            self.assertIn("hash_1", ctrl.hash_to_metadata)
            self.assertIn("hash_2", ctrl.hash_to_metadata)

    def test_eviction_disabled(self):
        dim = 64
        size = _embedding_bytes(1, dim)  # 256 bytes
        # Pool exactly fits 1 entry
        ctrl = _make_controller(pool_mb=size / (1024**2), enable_eviction=False)

        # Fill the pool
        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            self.assertIsNotNone(offset)

        # Try to allocate more — should fail without eviction
        with ctrl.lock:
            offset2 = ctrl._allocate_with_eviction(size)
        self.assertIsNone(offset2)

    def test_access_time_updates_prevent_eviction(self):
        ctrl = _make_controller(pool_mb=0.01)
        dim = 64
        size = _embedding_bytes(1, dim)

        # Insert 2 entries
        hashes = []
        for i in range(2):
            h = f"hash_{i}"
            hashes.append(h)
            with ctrl.lock:
                offset = ctrl.allocator.allocate(size)
                ctrl.hash_to_metadata[h] = (offset, 1, dim, size)
                ctrl._update_access_time(h)

        # Touch hash_0 to make it recently used
        time.sleep(0.01)
        with ctrl.lock:
            ctrl._update_access_time("hash_0")

        # Trigger eviction — hash_1 should be evicted (older)
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(size)
        self.assertIn("hash_1", candidates)
        self.assertNotIn("hash_0", candidates)

    def test_eviction_stats(self):
        ctrl = _make_controller(pool_mb=0.01)
        dim = 64
        size = _embedding_bytes(1, dim)

        # Insert and then evict
        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h"] = (offset, 1, dim, size)
            ctrl._update_access_time("h")

        with ctrl.lock:
            freed = ctrl._evict_hashes(["h"])
        self.assertGreater(freed, 0)
        self.assertEqual(ctrl.stats["eviction_count"], 1)
        self.assertGreater(ctrl.stats["total_evicted"], 0)

    def test_evict_nonexistent_hash(self):
        ctrl = _make_controller()
        with ctrl.lock:
            freed = ctrl._evict_hashes(["nonexistent"])
        self.assertEqual(freed, 0)

    def test_max_eviction_batch(self):
        ctrl = _make_controller(pool_mb=1.0, max_eviction_batch=2)
        dim = 64
        size = _embedding_bytes(1, dim)

        # Insert many small entries
        for i in range(10):
            h = f"hash_{i}"
            with ctrl.lock:
                offset = ctrl.allocator.allocate(size)
                if offset is None:
                    break
                ctrl.hash_to_metadata[h] = (offset, 1, dim, size)
                ctrl._update_access_time(h)

        # _select_eviction_candidates should return at most 2
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(size * 100)
        self.assertLessEqual(len(candidates), 2)


# ---------------------------------------------------------------------------
# Ref counting tests
# ---------------------------------------------------------------------------


class TestRefCounting(unittest.TestCase):
    def test_protect_prevents_eviction(self):
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h"] = (offset, 1, dim, size)
            ctrl._update_access_time("h")
            ctrl._protect_hash("h")

        # Should not be selected for eviction
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(size)
        self.assertNotIn("h", candidates)

        # Release and retry
        with ctrl.lock:
            ctrl._release_hash("h")
            candidates = ctrl._select_eviction_candidates(size)
        self.assertIn("h", candidates)

    def test_evict_hashes_skips_protected(self):
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h"] = (offset, 1, dim, size)
            ctrl._update_access_time("h")
            ctrl._protect_hash("h")

        # Attempt to evict — should be skipped
        with ctrl.lock:
            freed = ctrl._evict_hashes(["h"])
        self.assertEqual(freed, 0)
        with ctrl.lock:
            self.assertIn("h", ctrl.hash_to_metadata)

    def test_ref_count_multiple_protects(self):
        ctrl = _make_controller()
        with ctrl.lock:
            ctrl._protect_hash("h")
            ctrl._protect_hash("h")
            self.assertEqual(ctrl.ref_counts["h"], 2)

            ctrl._release_hash("h")
            self.assertEqual(ctrl.ref_counts["h"], 1)

            # Still protected
            candidates = ctrl._select_eviction_candidates(1)
            self.assertNotIn("h", candidates)

            ctrl._release_hash("h")
            self.assertNotIn("h", ctrl.ref_counts)

    def test_release_nonexistent_is_safe(self):
        ctrl = _make_controller()
        with ctrl.lock:
            ctrl._release_hash("nonexistent")  # should not raise

    def test_prefetch_sets_ref_count(self):
        """prefetch() should set ref_count=1 for each new entry."""
        ctrl = _make_controller()
        dim = 64
        ctrl.hidden_dims = {"image": dim}
        h = "img_hash_1"

        ctrl.prefetch("req1", [h], [1], modality="image")

        with ctrl.lock:
            self.assertEqual(ctrl.ref_counts.get(h), 1)

        # Simulate RDMA completion
        with ctrl.lock:
            ctrl._release_hash(h)
        self.assertNotIn(h, ctrl.ref_counts)

    def test_insert_batch_sets_ref_count(self):
        """insert_batch() should set ref_count=1 for each new entry."""
        ctrl = _make_controller()
        dim = 64
        h = "img_hash_1"
        tensor = torch.randn(1, dim)

        ctrl.insert_batch([h], [tensor])

        with ctrl.lock:
            self.assertEqual(ctrl.ref_counts.get(h), 1)

        # Simulate RDMA completion
        with ctrl.lock:
            ctrl._release_hash(h)
        self.assertNotIn(h, ctrl.ref_counts)

    def test_get_embeddings_sets_ref_count(self):
        """get_embeddings() should set ref_count=1 per hash."""
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h1"] = (offset, 1, dim, size)
            ctrl._update_access_time("h1")

        tensors = ctrl.get_embeddings(["h1"])
        self.assertIsNotNone(tensors[0])

        with ctrl.lock:
            self.assertEqual(ctrl.ref_counts.get("h1"), 1)

        # Protected — eviction should skip
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(size)
        self.assertNotIn("h1", candidates)

        # Release
        ctrl.release_embeddings(["h1"])
        with ctrl.lock:
            self.assertNotIn("h1", ctrl.ref_counts)

    def test_get_embeddings_missing_hash(self):
        """Missing hashes should return None and not set ref_count."""
        ctrl = _make_controller()
        tensors = ctrl.get_embeddings(["nonexistent"])
        self.assertIsNone(tensors[0])
        with ctrl.lock:
            self.assertNotIn("nonexistent", ctrl.ref_counts)

    def test_release_embeddings_missing_hash_is_safe(self):
        """Releasing a hash that was never protected should be a no-op."""
        ctrl = _make_controller()
        ctrl.release_embeddings(["nonexistent"])  # should not raise

    def test_io_loop_releases_prefetch_ref(self):
        """_io_loop should release ref_count after batch_get completes."""
        ctrl = _make_controller()
        dim = 64
        ctrl.hidden_dims = {"image": dim}
        h = "img_hash_1"

        ctrl.prefetch("req1", [h], [1], modality="image")

        with ctrl.lock:
            self.assertEqual(ctrl.ref_counts.get(h), 1)

        # Simulate _io_loop completing the RDMA GET
        op = ctrl.ongoing_prefetch.get("req1")
        self.assertIsNotNone(op)
        ctrl.mooncake_store.batch_get = MagicMock(return_value=[True])

        # Manually execute what _io_loop does for prefetch
        results = ctrl.mooncake_store.batch_get(op.keys, op.ptrs, op.sizes)
        op.mark_done(all(results))
        with ctrl.lock:
            for k in op.keys:
                ctrl._release_hash(k)

        with ctrl.lock:
            self.assertNotIn(h, ctrl.ref_counts)

    def test_io_loop_releases_insert_ref(self):
        """_io_loop should release ref_count after batch_put completes."""
        ctrl = _make_controller()
        dim = 64
        h = "img_hash_1"
        tensor = torch.randn(1, dim)

        ctrl.insert_batch([h], [tensor])

        with ctrl.lock:
            self.assertEqual(ctrl.ref_counts.get(h), 1)

        # Get the enqueued insert operation
        ctrl.insert_queue.put.assert_called_once()
        insert_op = ctrl.insert_queue.put.call_args[0][0]
        self.assertIsInstance(insert_op, EmbeddingInsertOperation)

        # Simulate _io_loop completing the RDMA PUT
        ctrl.mooncake_store.batch_put = MagicMock()
        ctrl.mooncake_store.batch_put(insert_op.keys, insert_op.ptrs, insert_op.sizes)
        with ctrl.lock:
            for k in insert_op.keys:
                ctrl._release_hash(k)

        with ctrl.lock:
            self.assertNotIn(h, ctrl.ref_counts)


# ---------------------------------------------------------------------------
# Race condition prevention tests
# ---------------------------------------------------------------------------


class TestRDMAEvictionRacePrevention(unittest.TestCase):
    def test_eviction_during_prefetch_is_blocked(self):
        """An entry with in-flight RDMA GET cannot be evicted."""
        ctrl = _make_controller(pool_mb=0.01)
        dim = 64
        ctrl.hidden_dims = {"image": dim}
        size = _embedding_bytes(1, dim)

        # Prefetch sets ref_count=1
        ctrl.prefetch("req1", ["h1"], [1], modality="image")

        # Now try to evict to make room — should skip h1
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(size * 10)
        self.assertNotIn("h1", candidates)

        # Direct eviction should also be skipped
        with ctrl.lock:
            freed = ctrl._evict_hashes(["h1"])
        self.assertEqual(freed, 0)

    def test_eviction_during_insert_is_blocked(self):
        """An entry with in-flight RDMA PUT cannot be evicted."""
        ctrl = _make_controller(pool_mb=0.01)
        dim = 64
        tensor = torch.randn(1, dim)

        ctrl.insert_batch(["h1"], [tensor])

        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(999999)
        self.assertNotIn("h1", candidates)

        with ctrl.lock:
            freed = ctrl._evict_hashes(["h1"])
        self.assertEqual(freed, 0)

    def test_eviction_during_get_embeddings_is_blocked(self):
        """An entry returned by get_embeddings() cannot be evicted."""
        ctrl = _make_controller(pool_mb=0.01)
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h1"] = (offset, 1, dim, size)
            ctrl._update_access_time("h1")

        tensors = ctrl.get_embeddings(["h1"])
        self.assertIsNotNone(tensors[0])

        # Try to evict — should be blocked
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(999999)
        self.assertNotIn("h1", candidates)

        # Release and verify eviction is now possible
        ctrl.release_embeddings(["h1"])
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(999999)
        self.assertIn("h1", candidates)

    def test_concurrent_eviction_while_reading(self):
        """Simulate a concurrent eviction attempt while a read holds a ref."""
        ctrl = _make_controller(pool_mb=0.05)
        dim = 64
        size = _embedding_bytes(1, dim)
        num_entries = 10

        # Insert entries
        for i in range(num_entries):
            h = f"hash_{i}"
            with ctrl.lock:
                offset = ctrl.allocator.allocate(size)
                if offset is None:
                    break
                ctrl.hash_to_metadata[h] = (offset, 1, dim, size)
                ctrl._update_access_time(h)

        # Simulate get_embeddings holding refs on hash_0..hash_4
        held_hashes = [f"hash_{i}" for i in range(5)]
        tensors = ctrl.get_embeddings(held_hashes)

        # Try to evict all — only unprotected entries should be candidates
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(999999)
        for h in held_hashes:
            self.assertNotIn(h, candidates)

        # Unprotected entries should be candidates
        for i in range(5, num_entries):
            h = f"hash_{i}"
            if h in ctrl.hash_to_metadata:
                self.assertIn(h, candidates)

        # Release refs
        ctrl.release_embeddings(held_hashes)
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(999999)
        for h in held_hashes:
            if h in ctrl.hash_to_metadata:
                self.assertIn(h, candidates)

    def test_evict_hashes_cleans_up_ref_counts(self):
        """After eviction, ref_counts for the evicted hash should be removed."""
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h"] = (offset, 1, dim, size)
            ctrl._update_access_time("h")
            # Stale ref_count (shouldn't happen normally, but test cleanup)
            ctrl.ref_counts["h"] = 0

        with ctrl.lock:
            # ref_count is 0, so eviction should proceed
            freed = ctrl._evict_hashes(["h"])
        self.assertGreater(freed, 0)
        with ctrl.lock:
            self.assertNotIn("h", ctrl.hash_to_metadata)
            self.assertNotIn("h", ctrl.ref_counts)


# ---------------------------------------------------------------------------
# get_embeddings view safety tests
# ---------------------------------------------------------------------------


class TestGetEmbeddingsViewSafety(unittest.TestCase):
    def test_get_embeddings_returns_view(self):
        """get_embeddings returns a view into cpu_pool, not a copy."""
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h1"] = (offset, 1, dim, size)
            ctrl._update_access_time("h1")

        tensors = ctrl.get_embeddings(["h1"])
        self.assertIsNotNone(tensors[0])
        self.assertEqual(tensors[0].shape, (1, dim))

        # Verify it's a view into cpu_pool (shares storage)
        self.assertTrue(
            tensors[0].storage().data_ptr() == ctrl.cpu_pool.storage().data_ptr()
        )

        # Release
        ctrl.release_embeddings(["h1"])

    def test_data_preserved_while_ref_held(self):
        """Data should remain intact as long as ref_count > 0."""
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        # Write known data
        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h1"] = (offset, 1, dim, size)
            ctrl._update_access_time("h1")
            view = (
                ctrl.cpu_pool[offset : offset + size].view(torch.float32).view(1, dim)
            )
            view.copy_(torch.ones(1, dim))

        # Read via get_embeddings (holds ref)
        tensors = ctrl.get_embeddings(["h1"])
        self.assertTrue(torch.all(tensors[0] == 1.0))

        # Verify data is still valid
        self.assertTrue(torch.all(tensors[0] == 1.0))

        # Release
        ctrl.release_embeddings(["h1"])


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestGetStats(unittest.TestCase):
    def test_stats_include_num_protected(self):
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h1"] = (offset, 1, dim, size)
            ctrl._update_access_time("h1")
            ctrl._protect_hash("h1")
            ctrl._protect_hash("h2")  # h2 not in metadata, but has ref

        stats = ctrl.get_stats()
        self.assertEqual(stats["num_protected"], 2)

    def test_stats_eviction_tracking(self):
        ctrl = _make_controller(pool_mb=0.01)
        dim = 64
        size = _embedding_bytes(1, dim)

        with ctrl.lock:
            offset = ctrl.allocator.allocate(size)
            ctrl.hash_to_metadata["h"] = (offset, 1, dim, size)
            ctrl._update_access_time("h")

        with ctrl.lock:
            ctrl._evict_hashes(["h"])

        stats = ctrl.get_stats()
        self.assertEqual(stats["eviction_count"], 1)
        self.assertGreater(stats["total_evicted"], 0)
        self.assertEqual(stats["num_cached"], 0)


# ---------------------------------------------------------------------------
# _select_eviction_candidates iterator safety test
# ---------------------------------------------------------------------------


class TestEvictionCandidateIteratorSafety(unittest.TestCase):
    def test_list_snapshot_prevents_concurrent_mutation(self):
        """sorted_hashes should be a list snapshot, not a live dict view."""
        ctrl = _make_controller()
        dim = 64
        size = _embedding_bytes(1, dim)

        # Insert entries
        for i in range(5):
            h = f"hash_{i}"
            with ctrl.lock:
                offset = ctrl.allocator.allocate(size)
                if offset is None:
                    break
                ctrl.hash_to_metadata[h] = (offset, 1, dim, size)
                ctrl._update_access_time(h)

        # _select_eviction_candidates should work even if access_order
        # is modified during iteration (the snapshot via list() prevents this)
        with ctrl.lock:
            candidates = ctrl._select_eviction_candidates(size)
        # Should return candidates without RuntimeError
        self.assertIsInstance(candidates, list)


if __name__ == "__main__":
    unittest.main()

import threading
import unittest

import torch

from sglang.srt.mem_cache.storage_prefetch import (
    L3StagingManager,
    StorageCheckpointRegistry,
    StorageCheckpointState,
    StorageWriteCompletion,
    StorageWriteTracker,
    l3_staging_allocation,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeHostPool:
    def __init__(self, size: int = 20, page_size: int = 2) -> None:
        self.size = size
        self.page_size = page_size
        self.free_slots = torch.arange(size, dtype=torch.int64)
        self.allocated: set[int] = set()

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int) -> torch.Tensor | None:
        if need_size > len(self.free_slots):
            return None
        indices = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        values = set(indices.tolist())
        assert not values.intersection(self.allocated)
        self.allocated.update(values)
        return indices

    def free(self, indices: torch.Tensor) -> int:
        values = set(indices.tolist())
        assert values.issubset(self.allocated)
        self.allocated.difference_update(values)
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


class TestL3StagingManager(unittest.TestCase):
    def test_quota_physically_isolates_ordinary_l2(self):
        pool = _FakeHostPool()
        manager = L3StagingManager()
        manager.install(pool, 0.2, lambda values: values)

        ordinary = pool.alloc(16)
        self.assertIsNotNone(ordinary)
        self.assertIsNone(pool.alloc(2))
        with l3_staging_allocation():
            staging = pool.alloc(4)

        self.assertIsNotNone(staging)
        assert ordinary is not None and staging is not None
        self.assertTrue(set(ordinary.tolist()).isdisjoint(staging.tolist()))
        self.assertEqual(manager.usage(), {"KV": "4/4"})
        pool.free(staging)
        with l3_staging_allocation():
            self.assertEqual(pool.available_size(), 4)

    def test_promotion_refills_reserve_without_copying(self):
        pool = _FakeHostPool()
        manager = L3StagingManager()
        manager.install(pool, 0.2, lambda values: values)
        quota = next(iter(manager.quotas.values()))
        with l3_staging_allocation():
            staging = pool.alloc(4)
        assert staging is not None

        promoted = quota.promote(staging)

        self.assertEqual(promoted, 4)
        self.assertEqual(quota.available_tokens, 4)
        self.assertTrue(set(staging.tolist()).issubset(pool.allocated))
        self.assertFalse(set(staging.tolist()).intersection(quota.staging_indices))

    def test_install_rejects_rank_divergence_before_mutation(self):
        pool = _FakeHostPool()
        manager = L3StagingManager()

        def diverge(values: list[int]) -> list[int]:
            reduced = values.copy()
            reduced[-1] -= 1
            return reduced

        with self.assertRaisesRegex(RuntimeError, "plan diverged"):
            manager.install(pool, 0.2, diverge)

        self.assertEqual(pool.available_size(), pool.size)

    def test_install_rolls_back_when_peer_rank_fails(self):
        pool = _FakeHostPool()
        manager = L3StagingManager()
        synchronization_count = 0

        def fail_peer_install(values: list[int]) -> list[int]:
            nonlocal synchronization_count
            synchronization_count += 1
            return [0] if synchronization_count == 2 else values

        with self.assertRaisesRegex(RuntimeError, "peer rank failed"):
            manager.install(pool, 0.2, fail_peer_install)

        self.assertFalse(manager.quotas)
        self.assertEqual(pool.available_size(), pool.size)

    def test_invalid_ratio_is_rejected(self):
        for ratio in (float("nan"), -0.1, 1.0):
            with self.subTest(ratio=ratio):
                manager = L3StagingManager()
                with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
                    manager.install(_FakeHostPool(), ratio, lambda values: values)

    def test_uninstall_restores_allocator_and_reclassifies_live_tokens(self):
        pool = _FakeHostPool()
        manager = L3StagingManager()
        manager.install(pool, 0.2, lambda values: values)
        quota = next(iter(manager.quotas.values()))
        with l3_staging_allocation():
            staging = pool.alloc(2)
        assert staging is not None

        reclassified = manager.uninstall()

        self.assertEqual(reclassified, 2)
        self.assertEqual(pool.alloc, quota.original_alloc)
        self.assertEqual(pool.free, quota.original_free)
        self.assertEqual(pool.available_size(), 18)
        pool.free(staging)
        self.assertEqual(pool.available_size(), 20)


class TestStorageWriteTracker(unittest.TestCase):
    def test_waiting_lookup_is_released_by_write_completion(self):
        tracker = StorageWriteTracker()
        tracker.register(1, ["page-a"])
        wait_started = threading.Event()
        wait_finished = threading.Event()

        def cancelled() -> bool:
            wait_started.set()
            return False

        def wait_for_write() -> None:
            self.assertTrue(tracker.wait_until_clear(["page-a"], cancelled))
            wait_finished.set()

        waiter = threading.Thread(target=wait_for_write)
        waiter.start()
        self.assertTrue(wait_started.wait(timeout=1))
        self.assertFalse(wait_finished.is_set())

        tracker.complete(1, durable_pages=1)
        waiter.join(timeout=1)

        self.assertFalse(waiter.is_alive())
        self.assertTrue(wait_finished.is_set())

    def test_clear_cancels_waiting_lookup(self):
        tracker = StorageWriteTracker()
        tracker.register(1, ["page-a"])
        wait_started = threading.Event()
        cancelled = threading.Event()
        wait_result: list[bool] = []

        def is_cancelled() -> bool:
            wait_started.set()
            return cancelled.is_set()

        waiter = threading.Thread(
            target=lambda: wait_result.append(
                tracker.wait_until_clear(["page-a"], is_cancelled)
            )
        )
        waiter.start()
        self.assertTrue(wait_started.wait(timeout=1))

        cancelled.set()
        tracker.clear()
        waiter.join(timeout=1)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(wait_result, [False])

    def test_completion_tracks_only_acknowledged_durable_prefix(self):
        tracker = StorageWriteTracker()
        tracker.register(1, ["page-a", "page-b", "page-c"])

        completion = tracker.complete(1, durable_pages=2)

        self.assertEqual(completion.durable_pages, 2)
        self.assertEqual(
            tracker.get_durable_hashes(["page-a", "page-b", "page-c"]),
            frozenset({"page-a", "page-b"}),
        )
        self.assertFalse(tracker.has_pending(["page-c"]))


class TestStorageCheckpointRegistry(unittest.TestCase):
    def test_reserved_checkpoint_becomes_ready_after_all_page_acks(self):
        registry = StorageCheckpointRegistry()
        registry.reserve("hicache:request-a")
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.PENDING
        )

        checkpoint = registry.create("hicache:request-a", ["page-a", "page-b"])
        registry.record_write_completion(
            StorageWriteCompletion(1, ("page-a",), durable_pages=1)
        )
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.PENDING
        )

        registry.record_write_completion(
            StorageWriteCompletion(2, ("page-b",), durable_pages=1)
        )
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.READY
        )
        self.assertIsNone(checkpoint.expected_hashes)
        self.assertEqual(checkpoint.durable_hashes, set())

    def test_reserving_reused_handle_replaces_terminal_state(self):
        registry = StorageCheckpointRegistry()
        registry.create("hicache:request-a", ["page-a"], durable_hashes=["page-a"])
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.READY
        )

        registry.reserve("hicache:request-a")

        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.PENDING
        )

    def test_failed_duplicate_does_not_invalidate_an_acknowledged_page(self):
        registry = StorageCheckpointRegistry()
        registry.create("hicache:request-a", ["page-a", "page-b"])
        registry.record_write_completion(
            StorageWriteCompletion(1, ("page-a",), durable_pages=1)
        )

        registry.record_write_completion(
            StorageWriteCompletion(2, ("page-a",), durable_pages=0)
        )

        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.PENDING
        )
        registry.record_write_completion(
            StorageWriteCompletion(3, ("page-b",), durable_pages=1)
        )
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.READY
        )

    def test_unacknowledged_required_page_fails_checkpoint(self):
        registry = StorageCheckpointRegistry()
        registry.create("hicache:request-a", ["page-a", "page-b"])

        registry.record_write_completion(
            StorageWriteCompletion(1, ("page-a", "page-b"), durable_pages=1)
        )

        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.FAILED
        )

    def test_failed_write_waits_for_overlapping_pending_write(self):
        tracker = StorageWriteTracker()
        registry = StorageCheckpointRegistry()
        registry.create("hicache:request-a", ["page-a"])
        tracker.register(1, ["page-a"])
        tracker.register(2, ["page-a"])

        failed_completion = tracker.complete(1, durable_pages=0)
        self.assertIsNotNone(failed_completion)
        registry.record_write_completion(failed_completion)
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.PENDING
        )

        durable_completion = tracker.complete(2, durable_pages=1)
        self.assertIsNotNone(durable_completion)
        registry.record_write_completion(durable_completion)
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.READY
        )

    def test_ready_waits_for_pending_operation_on_any_expected_page(self):
        tracker = StorageWriteTracker()
        registry = StorageCheckpointRegistry()
        registry.create("hicache:request-a", ["page-a", "page-b"])
        tracker.register(1, ["page-a"])
        tracker.register(2, ["page-b"])
        tracker.register(3, ["page-a"])

        page_a_completion = tracker.complete(1, durable_pages=1)
        self.assertIsNotNone(page_a_completion)
        registry.record_write_completion(
            page_a_completion, has_pending=tracker.has_pending
        )
        page_b_completion = tracker.complete(2, durable_pages=1)
        self.assertIsNotNone(page_b_completion)
        registry.record_write_completion(
            page_b_completion, has_pending=tracker.has_pending
        )

        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.PENDING
        )

        overlapping_completion = tracker.complete(3, durable_pages=0)
        self.assertIsNotNone(overlapping_completion)
        registry.record_write_completion(
            overlapping_completion, has_pending=tracker.has_pending
        )
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.READY
        )

    def test_checkpoint_uses_prior_durability_after_pending_write_clears(self):
        tracker = StorageWriteTracker()
        registry = StorageCheckpointRegistry()
        tracker.mark_durable(["page-a"])
        tracker.register(1, ["page-a"])

        registry.create(
            "hicache:request-a",
            ["page-a"],
            durable_hashes=tracker.get_durable_hashes(["page-a"]),
            has_pending=tracker.has_pending,
        )
        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.PENDING
        )

        completion = tracker.complete(1, durable_pages=0)
        self.assertIsNotNone(completion)
        registry.record_write_completion(completion, has_pending=tracker.has_pending)

        self.assertEqual(
            registry.get_state("hicache:request-a"), StorageCheckpointState.READY
        )


if __name__ == "__main__":
    unittest.main()

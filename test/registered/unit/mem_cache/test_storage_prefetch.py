import threading
import unittest

from sglang.srt.mem_cache.storage_prefetch import (
    StorageCheckpointRegistry,
    StorageCheckpointState,
    StorageWriteCompletion,
    StorageWriteTracker,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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

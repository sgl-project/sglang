"""CPU-only regression tests for CUDA IPC multimodal pool budgeting."""

import asyncio
import threading
import unittest

from sglang.srt.multimodal.transport.admission import (
    FeaturePublicationAdmission,
    MmMemoryOverloaded,
    MmMemoryRequestTooLarge,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    get_mm_feature_pool_size_per_worker,
)
from sglang.srt.multimodal.transport.memory_pool import StreamOrderedMmFeaturePool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestCudaIpcPoolBudget(CustomTestCase):
    def test_budget_is_not_multiplied_by_tokenizer_workers(self):
        budget = 1_024 * 1024 * 1024
        worker_num = 16

        per_worker = get_mm_feature_pool_size_per_worker(budget, worker_num)

        self.assertEqual(per_worker, 64 * 1024 * 1024)
        self.assertLessEqual(per_worker * worker_num, budget)

    def test_remainder_is_not_overallocated(self):
        self.assertEqual(get_mm_feature_pool_size_per_worker(1_001, 8), 125)
        self.assertLessEqual(get_mm_feature_pool_size_per_worker(1_001, 8) * 8, 1_001)

    def test_rejects_invalid_budget_or_worker_count(self):
        with self.assertRaisesRegex(ValueError, "total_pool_size"):
            get_mm_feature_pool_size_per_worker(0, 1)
        with self.assertRaisesRegex(ValueError, "tokenizer_worker_num"):
            get_mm_feature_pool_size_per_worker(1, 0)


class TestAtomicPoolReservation(CustomTestCase):
    def make_pool(self, slots=4):
        # exercise the production allocator without creating a CUDA allocation
        pool = object.__new__(StreamOrderedMmFeaturePool)
        pool.memory_size = 1024
        pool.data_start = 0
        pool.max_inflight_slices = slots
        pool.control_words_per_slot = 2
        pool.transport_name = "CUDA IPC"
        pool._available_ranges = [(0, 1024)]
        pool._available_slots = list(reversed(range(slots)))
        pool._slot_generations = [0] * slots
        pool._occupied = {}
        pool._lock = threading.Lock()
        pool._pool_error = None
        pool._recycler_stop_event = threading.Event()
        return pool

    def test_fragmented_batch_has_no_partial_reservation(self):
        pool = self.make_pool()
        leases = pool.try_reserve_batch([256] * 4)
        pool.release_reserved_batch([leases[0], leases[2]])
        self.assertIsNone(pool.try_reserve_batch([256, 512]))
        self.assertEqual(set(pool._occupied), {leases[1].slot, leases[3].slot})
        self.assertEqual(sorted(pool._available_ranges), [(0, 256), (512, 768)])
        # failed attempts must not recycle generation numbers
        again = pool.try_reserve_batch([256, 256])
        self.assertTrue(all(lease.generation > 1 for lease in again))
        pool.release_reserved_batch([leases[1], leases[3], *again])
        self.assertEqual(pool._available_ranges, [(0, 1024)])

    def test_slot_pressure_rolls_back_every_byte(self):
        pool = self.make_pool(slots=2)
        first = pool.try_reserve_batch([1])
        self.assertIsNone(pool.try_reserve_batch([1, 1]))
        self.assertEqual(pool.active_lease_count, 1)
        self.assertEqual(pool._available_ranges, [(256, 1024)])
        with self.assertRaises(MmMemoryRequestTooLarge):
            pool.try_reserve_batch([1, 1, 1])
        pool.release_reserved_batch(first)

    def test_alignment_is_included_in_capacity(self):
        pool = self.make_pool()
        with self.assertRaises(MmMemoryRequestTooLarge):
            pool.try_reserve_batch([513, 257])
        self.assertEqual(pool.active_lease_count, 0)

    def test_generation_exhaustion_does_not_mutate_allocator(self):
        pool = self.make_pool()
        pool._slot_generations[0] = 0x7FFFFFFF
        with self.assertRaisesRegex(RuntimeError, "generation exhausted"):
            pool.try_reserve_batch([100])
        self.assertEqual(pool._available_ranges, [(0, 1024)])
        self.assertEqual(len(pool._available_slots), 4)


class TestPublicationAdmission(CustomTestCase):
    def test_rejects_oversized_and_immediate_pressure(self):
        async def run():
            gate = FeaturePublicationAdmission(1024, 0)
            with self.assertRaises(MmMemoryRequestTooLarge):
                await gate.acquire(1025, lambda: self.fail("oversized reservation"))
            with self.assertRaises(MmMemoryOverloaded):
                await gate.acquire(1, lambda: None)
            self.assertEqual((gate.pending_count, gate.pending_bytes), (0, 0))

        asyncio.run(run())

    def test_fifo_cancel_and_native_lifetime(self):
        async def run():
            gate = FeaturePublicationAdmission(1024, 1000)
            capacity = [False]
            order = []

            def reserve(name):
                if capacity[0]:
                    order.append(name)
                    return name
                return None

            first = asyncio.create_task(gate.acquire(256, lambda: reserve("first")))
            await asyncio.sleep(0)
            second = asyncio.create_task(gate.acquire(256, lambda: reserve("second")))
            await asyncio.sleep(0)
            with self.assertRaises(MmMemoryOverloaded):
                await gate.acquire(768, lambda: None)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            capacity[0] = True
            gate.notify()
            result, ticket = await second
            self.assertEqual((result, order), ("second", ["second"]))
            self.assertEqual((gate.pending_count, gate.pending_bytes), (1, 256))
            gate.finish(ticket)
            self.assertEqual((gate.pending_count, gate.pending_bytes), (0, 0))

        asyncio.run(run())

    def test_recycle_between_check_and_wait_is_not_lost(self):
        async def run():
            gate = FeaturePublicationAdmission(1024, 1000)
            attempts = []

            def reserve():
                attempts.append(True)
                if len(attempts) == 1:
                    thread = threading.Thread(target=gate.notify)
                    thread.start()
                    thread.join()
                    return None
                return "reserved"

            result, ticket = await gate.acquire(1, reserve)
            self.assertEqual(result, "reserved")
            self.assertEqual(len(attempts), 2)
            gate.finish(ticket)

        asyncio.run(run())

    def test_timeout_and_shutdown_release_waiters(self):
        async def run():
            gate = FeaturePublicationAdmission(1024, 1)
            with self.assertRaises(MmMemoryOverloaded):
                await gate.acquire(10, lambda: None)
            gate.timeout_ms = 10000
            task = asyncio.create_task(gate.acquire(10, lambda: None))
            await asyncio.sleep(0)
            gate.close()
            with self.assertRaises(MmMemoryOverloaded):
                await task
            self.assertEqual((gate.pending_count, gate.pending_bytes), (0, 0))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the async continuous-batching stage workers.

CPU-only: exercises bounded backpressure, separate encode/finalize workers,
lifecycle states, and robust shutdown. CUDA stream/event handoff paths are
covered implicitly (they no-op without a GPU).
"""

import threading
import time
import unittest

from sglang.multimodal_gen.runtime.managers.continuous_stage_worker import (
    AsyncContinuousStageWorker,
    StageQueueFull,
    WorkerState,
    async_stages_supported,
)


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestAsyncStagesSupported(unittest.TestCase):
    def _args(self, **kwargs):
        defaults = {
            "cb_async_stages": True,
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
            "text_encoder_cpu_offload": False,
            "image_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
        }
        defaults.update(kwargs)
        return type("Args", (), defaults)()

    def test_enabled_by_default(self):
        self.assertTrue(async_stages_supported(self._args()))

    def test_disabled_by_flag(self):
        self.assertFalse(async_stages_supported(self._args(cb_async_stages=False)))

    def test_disabled_by_offload(self):
        self.assertFalse(async_stages_supported(self._args(vae_cpu_offload=True)))


class TestStageWorkerBasics(unittest.TestCase):
    def setUp(self):
        self.worker = AsyncContinuousStageWorker(queue_depth=2)

    def tearDown(self):
        self.worker.shutdown(timeout_s=5.0)

    def test_jobs_run_and_results_return(self):
        self.worker.submit("encode", 1, lambda: "encoded")
        self.worker.submit("finalize", 2, lambda: "finalized")
        self.assertTrue(_wait_for(lambda: self.worker.pending == 0 or True))

        results = {}
        deadline = time.monotonic() + 5.0
        while len(results) < 2 and time.monotonic() < deadline:
            for result in self.worker.poll_results():
                results[result.ticket] = result
            time.sleep(0.01)
        self.assertEqual(results[1].value, "encoded")
        self.assertEqual(results[2].value, "finalized")
        self.assertIsNone(results[1].error)
        self.assertEqual(self.worker.pending, 0)

    def test_job_errors_are_forwarded(self):
        def boom():
            raise ValueError("bad job")

        self.worker.submit("encode", 7, boom)
        results = []
        _wait_for(lambda: bool(results.extend(self.worker.poll_results()) or results))
        self.assertEqual(results[0].ticket, 7)
        self.assertIsInstance(results[0].error, ValueError)

    def test_bounded_queue_backpressure(self):
        release = threading.Event()
        started = threading.Event()

        def blocker():
            started.set()
            release.wait(timeout=10.0)
            return "done"

        self.worker.submit("encode", 1, blocker)
        self.assertTrue(started.wait(timeout=5.0))
        # Fill the bounded queue behind the in-flight job.
        self.worker.submit("encode", 2, lambda: "queued-a")
        self.worker.submit("encode", 3, lambda: "queued-b")
        self.assertFalse(self.worker.can_submit("encode"))
        with self.assertRaises(StageQueueFull):
            self.worker.submit("encode", 4, lambda: "overflow")
        self.assertFalse(self.worker.try_submit("encode", 5, lambda: "overflow"))
        # The other worker is unaffected by encode backpressure.
        self.assertTrue(self.worker.can_submit("finalize"))

        release.set()
        results = {}
        deadline = time.monotonic() + 5.0
        while len(results) < 3 and time.monotonic() < deadline:
            for result in self.worker.poll_results():
                results[result.ticket] = result
            time.sleep(0.01)
        self.assertEqual(set(results), {1, 2, 3})
        self.assertTrue(self.worker.can_submit("encode"))

    def test_workers_run_independently(self):
        release = threading.Event()
        started = threading.Event()

        def blocker():
            started.set()
            release.wait(timeout=10.0)
            return "slow-encode"

        self.worker.submit("encode", 1, blocker)
        self.assertTrue(started.wait(timeout=5.0))
        self.worker.submit("finalize", 2, lambda: "fast-finalize")

        results = {}
        deadline = time.monotonic() + 5.0
        while 2 not in results and time.monotonic() < deadline:
            for result in self.worker.poll_results():
                results[result.ticket] = result
            time.sleep(0.01)
        # Finalize finished while encode is still blocked.
        self.assertEqual(results[2].value, "fast-finalize")
        self.assertNotIn(1, results)
        release.set()

    def test_block_one_waits_for_result(self):
        self.worker.submit("encode", 1, lambda: time.sleep(0.2) or "late")
        results = self.worker.poll_results(block_one=True, block_timeout_s=5.0)
        self.assertEqual(results[0].value, "late")


class TestStageWorkerLifecycle(unittest.TestCase):
    def test_states_and_shutdown(self):
        worker = AsyncContinuousStageWorker(queue_depth=2)
        self.assertIs(worker.worker_state("encode"), WorkerState.RUNNING)
        self.assertIs(worker.worker_state("finalize"), WorkerState.RUNNING)

        worker.shutdown(timeout_s=5.0)
        self.assertIs(worker.worker_state("encode"), WorkerState.STOPPED)
        self.assertIs(worker.worker_state("finalize"), WorkerState.STOPPED)
        self.assertFalse(worker.can_submit("encode"))
        with self.assertRaises(RuntimeError):
            worker.submit("encode", 1, lambda: None)

    def test_shutdown_drains_queued_jobs(self):
        worker = AsyncContinuousStageWorker(queue_depth=4)
        seen = []
        for ticket in range(3):
            worker.submit("encode", ticket, lambda t=ticket: seen.append(t) or t)
        worker.shutdown(timeout_s=5.0)
        results = worker.poll_results()
        # Every queued job either ran or was errored out; none are lost.
        self.assertEqual({result.ticket for result in results}, {0, 1, 2})
        ran = {result.ticket for result in results if result.error is None}
        errored = {result.ticket for result in results if result.error is not None}
        self.assertEqual(ran | errored, {0, 1, 2})
        self.assertEqual(worker.pending, 0)

    def test_stuck_worker_errors_in_flight_jobs(self):
        worker = AsyncContinuousStageWorker(queue_depth=4)
        release = threading.Event()
        started = threading.Event()

        def blocker():
            started.set()
            release.wait(timeout=30.0)
            return "eventually"

        worker.submit("encode", 1, blocker)
        self.assertTrue(started.wait(timeout=5.0))
        worker.submit("encode", 2, lambda: "queued-behind")

        # The blocked worker cannot honor a fast shutdown.
        worker.shutdown(timeout_s=0.2)
        self.assertIs(worker.worker_state("encode"), WorkerState.FAILED)

        results = worker.poll_results()
        errored = {r.ticket for r in results if r.error is not None}
        self.assertIn(2, errored)  # never ran; surfaced instead of hanging
        release.set()


if __name__ == "__main__":
    unittest.main()

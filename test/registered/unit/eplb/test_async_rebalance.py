"""Exercise the production protocol with independent rank and read workers."""

import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.eplb.async_rebalance import (
    AsyncEplbCoordinator,
    Phase,
    RebalanceAborted,
    UnsafeRebalanceFailure,
    plan_layer_pull,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Collective:
    def __init__(self, size):
        self.barrier = threading.Barrier(size, timeout=3)
        self.values = [None] * size

    def exchange(self, rank, value):
        self.values[rank] = value
        self.barrier.wait()
        result = tuple(self.values)
        self.barrier.wait()
        return result

    def rank(self, rank):
        collective = self

        class Control:
            def vote(self, epoch, phase, layer, ready, error):
                values = collective.exchange(rank, (epoch, phase, layer, ready, error))
                if len({v[:3] for v in values}) != 1:
                    raise RuntimeError("Protocol mismatch")
                return all(v[3] for v in values), any(v[4] for v in values)

            def drained(self, success):
                return all(collective.exchange(rank, success))

        return Control()


class Runtime:
    def __init__(self, rank, live):
        self.rank, self.live = rank, live
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.installed = []
        self.fail_prepare = self.fail_read = self.fail_drain = False
        self.closed = False

    def prepare(self, placement):
        if self.fail_prepare:
            raise ValueError("invalid placement")
        self.plan = plan_layer_pull(tuple(self.live), placement, self.rank, 2, 2)
        self.staging = {}
        return (0,)

    def preparation_ready(self):
        return True

    def stage(self, layer):
        slots = len(self.live) // 2
        for source, destination in self.plan.local_copies:
            self.staging[destination] = self.live[self.rank * slots + source]

        def read():
            self.entered.set()
            if not self.gate.wait(timeout=3):
                raise TimeoutError("test read did not unblock")
            if self.fail_read:
                raise OSError("read rejected")
            for rank, source, destination in self.plan.reads:
                self.staging[destination] = self.live[rank * slots + source]

        return read

    def install(self, layer):
        for source, destination in self.plan.installs:
            self.live[self.rank * (len(self.live) // 2) + destination] = self.staging[
                source
            ]
        self.installed.append(layer)

    def synchronize(self):
        if self.fail_drain:
            raise RuntimeError("DMA completion unknown")

    def close(self):
        self.closed = True


class TestAsyncRebalance(unittest.TestCase):
    def setUp(self):
        self.live = [0, 1, 2, 3]
        collective = Collective(2)
        self.runtimes = [Runtime(i, self.live) for i in range(2)]
        self.coordinators = [
            AsyncEplbCoordinator(collective.rank(i), runtime, timeout=1)
            for i, runtime in enumerate(self.runtimes)
        ]
        self.ranks = ThreadPoolExecutor(max_workers=2)

    def tearDown(self):
        for runtime in self.runtimes:
            runtime.gate.set()
        for coordinator in self.coordinators:
            coordinator._executor.shutdown(wait=True)
        self.ranks.shutdown(wait=True)

    def together(self, method):
        futures = [self.ranks.submit(getattr(c, method)) for c in self.coordinators]
        results = []
        for future in futures:
            try:
                results.append(future.result(timeout=4))
            except Exception as error:
                results.append(error)
        return results

    def start(self, placement=(2, 3, 0, 1)):
        for coordinator in self.coordinators:
            coordinator.start(lambda: placement)

    def until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            results = self.together("poll")
            if any(isinstance(r, Exception) for r in results):
                return results
            if time.monotonic() > deadline:
                self.fail("protocol failed to progress")
            time.sleep(0.001)
        return results

    def test_slow_reader_blocks_install_then_allows_two_epochs(self):
        self.runtimes[1].gate.clear()
        self.start()
        self.until(lambda: all(c.phase == Phase.STAGING for c in self.coordinators))
        self.assertTrue(self.runtimes[1].entered.wait(1))
        for _ in range(3):
            self.assertEqual(self.together("poll"), [False, False])
        self.assertEqual(self.live, [0, 1, 2, 3])
        self.assertFalse(any(r.installed for r in self.runtimes))
        self.runtimes[1].gate.set()
        self.until(lambda: all(c.phase == Phase.IDLE for c in self.coordinators))
        self.assertEqual(self.live, [2, 3, 0, 1])
        self.start((1, 0, 3, 2))
        self.until(lambda: all(c.phase == Phase.IDLE for c in self.coordinators))
        self.assertEqual(self.live, [1, 0, 3, 2])
        self.assertEqual(self.together("close"), [None, None])
        self.assertTrue(all(r.closed for r in self.runtimes))

    def test_one_rank_error_aborts_every_rank_without_install(self):
        for failure in ("fail_prepare", "fail_read"):
            with self.subTest(failure=failure):
                setattr(self.runtimes[1], failure, True)
                self.start()
                result = self.until(lambda: False)
                self.assertTrue(all(isinstance(r, RebalanceAborted) for r in result))
                self.assertTrue(
                    all(c.phase == Phase.DISABLED for c in self.coordinators)
                )
                self.assertEqual(self.live, [0, 1, 2, 3])
                self.assertEqual(self.together("reset"), [None, None])
                setattr(self.runtimes[1], failure, False)

    def test_cancel_waits_for_active_reader(self):
        self.runtimes[1].gate.clear()
        self.start()
        self.until(lambda: all(c.phase == Phase.STAGING for c in self.coordinators))
        self.assertTrue(self.runtimes[1].entered.wait(1))
        futures = [self.ranks.submit(c.cancel) for c in self.coordinators]
        self.assertFalse(any(f.done() for f in futures))
        self.runtimes[1].gate.set()
        for future in futures:
            future.result(timeout=3)
        self.assertEqual(self.live, [0, 1, 2, 3])

    def test_unknown_dma_completion_fails_every_rank(self):
        self.runtimes[1].fail_drain = True
        result = self.together("cancel")
        self.assertTrue(all(isinstance(r, UnsafeRebalanceFailure) for r in result))
        self.assertTrue(all(c.phase == Phase.FAILED for c in self.coordinators))
        self.assertFalse(any(r.closed for r in self.runtimes))

    def test_mismatched_epochs_fail_stop(self):
        self.start()
        self.coordinators[1].epoch += 1
        result = self.together("poll")
        self.assertTrue(all(isinstance(r, UnsafeRebalanceFailure) for r in result))

    def test_random_plans_preserve_experts_with_duplicates_and_idle_ranks(self):
        rng = random.Random(917)
        for _ in range(200):
            old = list(range(6)) + [rng.randrange(6) for _ in range(6)]
            new = old.copy()
            rng.shuffle(old)
            rng.shuffle(new)
            output = old.copy()
            for rank in range(4):
                plan = plan_layer_pull(tuple(old), tuple(new), rank, 4, 2)
                staged = {dst: old[rank * 3 + src] for src, dst in plan.local_copies}
                for peer, src, dst in plan.reads:
                    staged[dst] = old[peer * 3 + src]
                for src, dst in plan.installs:
                    output[rank * 3 + dst] = staged[src]
                self.assertEqual(len(set(staged.values())), len(staged))
            self.assertEqual(output, new)

    def test_invalid_capacity_or_expert_set_rejected(self):
        for old, new, size in [((0,), (0,), 0), ((0, 1), (0, 2), 2), ((0,), (0,), 2)]:
            with self.assertRaises(ValueError):
                plan_layer_pull(old, new, 0, size, 1)


if __name__ == "__main__":
    unittest.main()

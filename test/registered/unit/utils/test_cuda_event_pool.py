import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.utils import cuda_event_pool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# About 2 ms at a 2 GHz SM clock: long enough for an unordered copy to run
# before the producer's write.
_SLEEP_CYCLES = 4_000_000


class FakeEvent:
    def __init__(self) -> None:
        self.calls = []

    def record(self, stream) -> None:
        self.calls.append(("record", stream))

    def wait(self, stream) -> None:
        self.calls.append(("wait", stream))


class FakeStream:
    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.waited_streams = []

    def wait_stream(self, stream) -> None:
        self.waited_streams.append(stream)


def _clear_pools(test: unittest.TestCase) -> None:
    pools = patch.dict(cuda_event_pool._pools, clear=True)
    pools.start()
    test.addCleanup(pools.stop)


class TestWaitStream(unittest.TestCase):
    def setUp(self) -> None:
        _clear_pools(self)
        capturing = patch("torch.cuda.is_current_stream_capturing", return_value=False)
        self.capturing = capturing.start()
        self.addCleanup(capturing.stop)

    def test_records_on_src_waits_on_dst_and_returns_event(self) -> None:
        first, second = FakeEvent(), FakeEvent()
        pool = deque([first, second])
        cuda_event_pool._pools[0] = pool
        src, dst = FakeStream(), FakeStream()

        cuda_event_pool.wait_stream(dst, src)

        self.assertEqual(first.calls, [("record", src), ("wait", dst)])
        self.assertEqual(list(pool), [second, first])
        self.assertEqual(dst.waited_streams, [])

    def test_falls_back_to_pytorch_path(self) -> None:
        event = FakeEvent()
        cases = [
            # With no pool at all, even a stream without device_index works.
            ("disabled", {}, object(), False),
            ("device not prewarmed", {1: deque([event])}, FakeStream(0), False),
            ("capturing", {0: deque([event])}, FakeStream(0), True),
            ("pool empty", {0: deque()}, FakeStream(0), False),
        ]
        for name, pools, src, capturing in cases:
            with self.subTest(name), patch.dict(cuda_event_pool._pools, pools):
                event.calls.clear()
                self.capturing.return_value = capturing
                dst = FakeStream(0)

                cuda_event_pool.wait_stream(dst, src)

                self.assertEqual(dst.waited_streams, [src])
                self.assertEqual(event.calls, [])

    def test_failure_propagates_and_drops_event(self) -> None:
        event = FakeEvent()
        event.wait = Mock(side_effect=RuntimeError("wait failed"))
        pool = deque([event])
        cuda_event_pool._pools[0] = pool

        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            cuda_event_pool.wait_stream(FakeStream(), FakeStream())

        self.assertEqual(len(pool), 0)

    def test_concurrent_callers_never_share_an_event(self) -> None:
        lock = threading.Lock()
        held, shared = set(), []

        class CheckedEvent(FakeEvent):
            def record(self, stream) -> None:
                with lock:
                    if self in held:
                        shared.append(self)
                    held.add(self)
                time.sleep(0)  # Let other callers run while this event is held.

            def wait(self, stream) -> None:
                with lock:
                    held.discard(self)

        # Fewer events than threads, so callers also hit the empty pool.
        events = [CheckedEvent() for _ in range(4)]
        cuda_event_pool._pools[0] = deque(events)

        def worker() -> None:
            src, dst = FakeStream(), FakeStream()
            for _ in range(2000):
                cuda_event_pool.wait_stream(dst, src)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(shared, [])
        self.assertCountEqual(cuda_event_pool._pools[0], events)


class TestInitCudaEventPool(unittest.TestCase):
    def test_flag_gates_prewarm(self) -> None:
        from sglang.srt.environ import envs
        from sglang.srt.model_executor import model_runner

        platform = SimpleNamespace(is_cuda=lambda: True)
        device = SimpleNamespace(gpu_id=3)
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                envs.SGLANG_ENABLE_CUDA_EVENT_POOL.override(enabled),
                patch.object(model_runner, "current_platform", platform),
                patch.object(model_runner, "get_device", return_value=device),
                patch.object(model_runner, "prewarm_cuda_event_pool") as prewarm,
            ):
                model_runner.ModelRunner.init_cuda_event_pool(SimpleNamespace())

                if enabled:
                    prewarm.assert_called_once_with(3)
                else:
                    prewarm.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestWaitStreamOnDevice(unittest.TestCase):
    def setUp(self) -> None:
        _clear_pools(self)
        torch.cuda.synchronize()

    def assert_pooled(self):
        return patch.object(
            torch.cuda.Stream, "wait_stream", side_effect=AssertionError("fallback")
        )

    def assert_ordered(self, src, dst, value, observed) -> None:
        """Each copy on ``dst`` must see the delayed write on ``src``."""
        for device in {src.device_index, dst.device_index}:
            torch.cuda.synchronize(device)
        for i in range(observed.numel()):
            with torch.cuda.stream(src):
                torch.cuda._sleep(_SLEEP_CYCLES)
                value.fill_(i + 1)
            cuda_event_pool.wait_stream(dst, src)
            with torch.cuda.stream(dst):
                observed[i : i + 1].copy_(value)
            cuda_event_pool.wait_stream(src, dst)
        torch.cuda.synchronize()

        expected = list(range(1, observed.numel() + 1))
        self.assertEqual(observed.cpu().tolist(), expected)

    def test_orders_delayed_writes(self) -> None:
        cuda_event_pool.prewarm_cuda_event_pool(0)
        pool = cuda_event_pool._pools[0]
        self.assertEqual(len(pool), cuda_event_pool._POOL_SIZE)
        # Every CUDA event is created by the prewarm, not on the hot path.
        self.assertTrue(all(event.cuda_event != 0 for event in pool))
        src, dst = torch.cuda.Stream(), torch.cuda.Stream()
        value = torch.zeros(1, device="cuda")
        observed = torch.zeros(16, device="cuda")

        with self.assert_pooled():
            self.assert_ordered(src, dst, value, observed)

    def test_reused_event_keeps_enqueued_wait(self) -> None:
        with patch.object(cuda_event_pool, "_POOL_SIZE", 1):
            cuda_event_pool.prewarm_cuda_event_pool(0)
        (event,) = cuda_event_pool._pools[0]
        a, b, c, d = (torch.cuda.Stream() for _ in range(4))
        value = torch.zeros(1, device="cuda")
        observed = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

        with self.assert_pooled():
            with torch.cuda.stream(a):
                torch.cuda._sleep(5 * _SLEEP_CYCLES)
                value.fill_(42)
            cuda_event_pool.wait_stream(b, a)
            # Re-record the only event on idle C at once; B's wait came first.
            cuda_event_pool.wait_stream(d, c)
            with torch.cuda.stream(b):
                observed.copy_(value)
        torch.cuda.synchronize()

        self.assertEqual(observed.item(), 42)
        self.assertIs(cuda_event_pool._pools[0][0], event)

    def test_graph_capture_uses_pytorch_path_and_replays(self) -> None:
        cuda_event_pool.prewarm_cuda_event_pool(0)
        pool = cuda_event_pool._pools[0]
        first = pool[0]
        side = torch.cuda.Stream()
        x = torch.arange(4, device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            x.add_(1)
            cuda_event_pool.wait_stream(side, torch.cuda.current_stream())
            with torch.cuda.stream(side):
                y.copy_(x * 2)
            cuda_event_pool.wait_stream(torch.cuda.current_stream(), side)

        # The pooled path would have rotated the pool.
        self.assertIs(pool[0], first)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(y.cpu().tolist(), [4.0, 6.0, 8.0, 10.0])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 GPUs")
    def test_cross_device_orders_delayed_writes(self) -> None:
        cuda_event_pool.prewarm_cuda_event_pool(0)
        cuda_event_pool.prewarm_cuda_event_pool(1)
        src, dst = torch.cuda.Stream(device=1), torch.cuda.Stream(device=0)
        value = torch.zeros(1, device="cuda:1")
        observed = torch.zeros(16, device="cuda:0")

        with self.assert_pooled():
            self.assert_ordered(src, dst, value, observed)


if __name__ == "__main__":
    unittest.main()

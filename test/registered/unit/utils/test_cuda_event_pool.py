from __future__ import annotations

import contextlib
import unittest

import torch

from sglang.srt.utils import cuda_event_pool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class FakeDevice:
    def __init__(self, index: int) -> None:
        self.index = index


class FakeEvent:
    created = 0

    def __init__(self, enable_timing: bool = False) -> None:
        assert not enable_timing
        self.identifier = FakeEvent.created
        FakeEvent.created += 1
        self.records: list[int] = []

    def record(self, stream: FakeStream) -> None:
        self.records.append(stream.device.index)


class FakeStream:
    def __init__(self, device: int = 0) -> None:
        self.device = FakeDevice(device)
        self.waited_events: list[FakeEvent] = []
        self.recorded_events: list[FakeEvent] = []
        self.synchronize_calls = 0
        self.fail_next_wait = False

    def wait_event(self, event: FakeEvent) -> None:
        if self.fail_next_wait:
            self.fail_next_wait = False
            raise RuntimeError("injected wait failure")
        self.waited_events.append(event)

    def record_event(self, event: FakeEvent | None = None) -> FakeEvent:
        if event is None:
            event = FakeEvent()
        event.record(self)
        self.recorded_events.append(event)
        return event

    def wait_stream(self, stream: FakeStream) -> None:
        self.wait_event(stream.record_event())

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class FakeCuda:
    Stream = FakeStream
    Event = FakeEvent

    def __init__(self) -> None:
        self.capturing = False
        self._current_device = 0

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return self._current_device

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    @contextlib.contextmanager
    def device(self, index: int):
        previous = self._current_device
        self._current_device = index
        try:
            yield
        finally:
            self._current_device = previous


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()


class TestCudaEventPool(unittest.TestCase):
    def setUp(self) -> None:
        cuda_event_pool._reset_cuda_event_pool_for_testing()
        FakeEvent.created = 0
        self.torch = FakeTorch()

    def tearDown(self) -> None:
        cuda_event_pool._reset_cuda_event_pool_for_testing()

    def test_reuses_materialized_events_in_fifo_order(self) -> None:
        self.assertTrue(
            cuda_event_pool.install_cuda_event_pool(
                torch_module=self.torch, pool_size=4
            )
        )
        source = FakeStream(0)
        destination = FakeStream(0)

        for _ in range(5):
            destination.wait_stream(source)

        self.assertEqual(FakeEvent.created, 4)
        self.assertEqual(len(destination.waited_events), 5)
        self.assertIs(destination.waited_events[0], destination.waited_events[4])
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["pooled_calls"], 5)
        self.assertEqual(snapshot["events_materialized"], 4)
        self.assertEqual(snapshot["devices"]["0"]["max_in_use"], 1)

    def test_capture_uses_original_pytorch_path(self) -> None:
        cuda_event_pool.install_cuda_event_pool(torch_module=self.torch, pool_size=2)
        self.torch.cuda.capturing = True

        FakeStream(0).wait_stream(FakeStream(0))

        self.assertEqual(FakeEvent.created, 1)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["capture_bypasses"], 1)
        self.assertEqual(snapshot["original_calls"], 1)
        self.assertEqual(snapshot["events_materialized"], 0)

    def test_pool_is_per_source_device(self) -> None:
        cuda_event_pool.install_cuda_event_pool(torch_module=self.torch, pool_size=3)

        FakeStream(0).wait_stream(FakeStream(0))
        FakeStream(1).wait_stream(FakeStream(1))

        self.assertEqual(FakeEvent.created, 6)
        self.assertEqual(
            set(cuda_event_pool.cuda_event_pool_stats()["devices"]), {"0", "1"}
        )

    def test_exhaustion_falls_back_without_growing_pool(self) -> None:
        cuda_event_pool.install_cuda_event_pool(torch_module=self.torch, pool_size=2)
        cuda_event_pool.prewarm_cuda_event_pool(0)
        internal_pool = cuda_event_pool._POOLS[0]
        leases = [internal_pool.acquire(), internal_pool.acquire()]
        self.assertNotIn(None, leases)

        with self.assertLogs(cuda_event_pool.logger, level="WARNING") as logs:
            FakeStream(0).wait_stream(FakeStream(0))

        self.assertEqual(FakeEvent.created, 3)
        self.assertIn("CUDA event pool exhausted", logs.output[0])
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["pool_exhaustions"], 1)
        self.assertEqual(snapshot["original_calls"], 1)
        for event in leases:
            internal_pool.release(event)

    def test_failed_event_is_quarantined(self) -> None:
        cuda_event_pool.install_cuda_event_pool(torch_module=self.torch, pool_size=2)
        destination = FakeStream(0)
        destination.fail_next_wait = True

        with self.assertRaisesRegex(RuntimeError, "injected wait failure"):
            destination.wait_stream(FakeStream(0))

        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["pooled_call_failures"], 1)
        self.assertEqual(snapshot["devices"]["0"]["quarantined"], 1)
        self.assertEqual(snapshot["devices"]["0"]["available"], 1)

    def test_install_is_idempotent_and_checks_pool_size(self) -> None:
        self.assertTrue(
            cuda_event_pool.install_cuda_event_pool(
                torch_module=self.torch, pool_size=2
            )
        )
        self.assertFalse(
            cuda_event_pool.install_cuda_event_pool(
                torch_module=self.torch, pool_size=2
            )
        )
        with self.assertRaisesRegex(ValueError, "already installed"):
            cuda_event_pool.install_cuda_event_pool(
                torch_module=self.torch, pool_size=3
            )

    def test_uninstall_restores_original_method(self) -> None:
        original = FakeStream.wait_stream
        cuda_event_pool.install_cuda_event_pool(torch_module=self.torch, pool_size=2)

        self.assertIsNot(FakeStream.wait_stream, original)
        self.assertTrue(cuda_event_pool.uninstall_cuda_event_pool())
        self.assertIs(FakeStream.wait_stream, original)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCudaEventPoolOnDevice(unittest.TestCase):
    def tearDown(self) -> None:
        torch.cuda.synchronize()
        cuda_event_pool._reset_cuda_event_pool_for_testing()

    def test_repeated_cross_stream_dependencies_reuse_events(self) -> None:
        cuda_event_pool._reset_cuda_event_pool_for_testing()
        cuda_event_pool.install_cuda_event_pool(torch_module=torch, pool_size=8)
        cuda_event_pool.prewarm_cuda_event_pool()

        source = torch.cuda.Stream()
        destination = torch.cuda.Stream()
        value = torch.zeros(1, device="cuda")
        observed = torch.zeros_like(value)
        for _ in range(256):
            with torch.cuda.stream(source):
                value.add_(1)
            destination.wait_stream(source)
            with torch.cuda.stream(destination):
                observed.copy_(value)
            source.wait_stream(destination)

        torch.cuda.synchronize()
        self.assertEqual(observed.item(), 256)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["events_materialized"], 8)
        self.assertEqual(snapshot["pooled_calls"], 512)
        self.assertEqual(snapshot["pool_exhaustions"], 0)


if __name__ == "__main__":
    unittest.main()

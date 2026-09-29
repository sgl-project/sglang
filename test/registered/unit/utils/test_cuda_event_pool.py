from __future__ import annotations

import contextlib
import importlib.util
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import cuda_event_pool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# About 2 ms at a 2 GHz SM clock: long enough for an unordered consumer to run
# before the producer's write.
_SLEEP_CYCLES = 4_000_000


class FakeGenericEvent:
    """Stands in for ``torch.Event``."""

    def record(self, stream: FakeGenericStream) -> None:
        pass


class FakeEvent:
    """Stands in for ``torch.cuda.Event``, bound to its first recording device."""

    created = 0

    def __init__(self, enable_timing: bool = False) -> None:
        assert not enable_timing
        self.identifier = FakeEvent.created
        FakeEvent.created += 1
        self.device_index: int | None = None
        self.records: list[int] = []

    def record(self, stream: FakeStream) -> None:
        index = stream.device_index
        if self.device_index is None:
            self.device_index = index
        elif self.device_index != index:
            raise RuntimeError(
                f"Event device {self.device_index} does not match recording "
                f"stream's device {index}."
            )
        self.records.append(index)


class FakeGenericStream:
    """Stands in for ``torch.Stream``, the base class of ``torch.cuda.Stream``."""

    def __init__(self, device: int = 0) -> None:
        self.device_index = device
        self.recorded_events: list[object] = []

    def record_event(self, event: object | None = None) -> object:
        if event is None:
            event = FakeGenericEvent()
        elif type(event) is not FakeGenericEvent:
            raise RuntimeError("expected event to be a torch.Event object")
        event.record(self)
        self.recorded_events.append(event)
        return event


class FakeStream(FakeGenericStream):
    def __init__(self, device: int = 0) -> None:
        super().__init__(device)
        self.waited_events: list[object] = []
        self.synchronize_calls = 0
        self.fail_next_wait = False

    def wait_event(self, event: object) -> None:
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

    def wait_stream(self, stream: FakeGenericStream) -> None:
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

    def install(self, pool_size: int, devices: tuple[int, ...] = (0,)) -> None:
        self.assertTrue(
            cuda_event_pool.install_cuda_event_pool(
                torch_module=self.torch, pool_size=pool_size
            )
        )
        for device in devices:
            self.assertIsNotNone(cuda_event_pool.prewarm_cuda_event_pool(device))

    def test_reuses_materialized_events_in_fifo_order(self) -> None:
        self.install(pool_size=4)
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

    def test_accepts_stream_keyword(self) -> None:
        self.install(pool_size=2)
        destination = FakeStream(0)

        destination.wait_stream(stream=FakeStream(0))

        self.assertEqual(len(destination.waited_events), 1)
        self.assertEqual(cuda_event_pool.cuda_event_pool_stats()["pooled_calls"], 1)

    def test_capture_uses_original_pytorch_path(self) -> None:
        cuda_event_pool.install_cuda_event_pool(torch_module=self.torch, pool_size=2)
        self.torch.cuda.capturing = True

        FakeStream(0).wait_stream(FakeStream(0))

        self.assertEqual(FakeEvent.created, 1)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["capture_bypasses"], 1)
        self.assertEqual(snapshot["original_calls"], 1)
        self.assertEqual(snapshot["events_materialized"], 0)

    def test_unprewarmed_device_uses_original_path(self) -> None:
        self.install(pool_size=3, devices=(0,))

        FakeStream(1).wait_stream(FakeStream(1))

        # Only the fallback's own event: nothing is materialized on the hot path.
        self.assertEqual(FakeEvent.created, 3 + 1)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["unprewarmed_fallbacks"], 1)
        self.assertEqual(snapshot["original_calls"], 1)
        self.assertEqual(snapshot["events_materialized"], 3)
        self.assertEqual(set(snapshot["devices"]), {"0"})

    def test_pool_is_per_source_device(self) -> None:
        self.install(pool_size=3, devices=(0, 1))

        FakeStream(0).wait_stream(FakeStream(0))
        FakeStream(1).wait_stream(FakeStream(1))

        self.assertEqual(FakeEvent.created, 6)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(set(snapshot["devices"]), {"0", "1"})
        self.assertEqual(snapshot["pooled_calls"], 2)

    def test_cross_device_wait_uses_source_device_pool(self) -> None:
        self.install(pool_size=2, devices=(0, 1))
        source = FakeStream(1)
        destination = FakeStream(0)

        for _ in range(3):
            destination.wait_stream(source)

        # FakeEvent rejects a recording stream on another device, like torch.
        self.assertEqual(
            [event.device_index for event in destination.waited_events], [1, 1, 1]
        )
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["pooled_calls"], 3)
        self.assertEqual(snapshot["pooled_call_failures"], 0)
        self.assertEqual(snapshot["devices"]["1"]["max_in_use"], 1)
        self.assertEqual(snapshot["devices"]["0"]["max_in_use"], 0)

    def test_generic_stream_source_uses_original_path(self) -> None:
        self.install(pool_size=2)
        destination = FakeStream(0)

        destination.wait_stream(FakeGenericStream(0))

        self.assertIsInstance(destination.waited_events[0], FakeGenericEvent)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["non_cuda_source_fallbacks"], 1)
        self.assertEqual(snapshot["original_calls"], 1)
        self.assertEqual(snapshot["pooled_call_failures"], 0)
        self.assertEqual(snapshot["devices"]["0"]["available"], 2)
        self.assertEqual(snapshot["devices"]["0"]["quarantined"], 0)

    def test_exhaustion_falls_back_without_growing_pool(self) -> None:
        self.install(pool_size=2)
        internal_pool = cuda_event_pool._INSTALLATION.pools[0]
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

    def test_concurrent_callers_never_share_a_leased_event(self) -> None:
        num_threads, calls_per_thread = 8, 2000
        self.install(pool_size=16)
        lock = threading.Lock()
        leased: set[int] = set()
        double_leases: list[int] = []

        class LeaseCheckingStream(FakeStream):
            def record_event(self, event=None):
                event = super().record_event(event)
                with lock:
                    if id(event) in leased:
                        double_leases.append(id(event))
                    leased.add(id(event))
                time.sleep(0)  # Let other callers run while this lease is open.
                return event

            def wait_event(self, event) -> None:
                super().wait_event(event)
                with lock:
                    leased.discard(id(event))

        destinations = [LeaseCheckingStream(0) for _ in range(num_threads)]

        def worker(destination: LeaseCheckingStream) -> None:
            source = LeaseCheckingStream(0)
            for _ in range(calls_per_thread):
                destination.wait_stream(source)

        threads = [threading.Thread(target=worker, args=(d,)) for d in destinations]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(double_leases, [])
        self.assertEqual(
            sum(len(d.waited_events) for d in destinations),
            num_threads * calls_per_thread,
        )
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        device = snapshot["devices"]["0"]
        self.assertEqual(device["in_use"], 0)
        self.assertEqual(device["available"], 16)
        # Each caller holds at most one event, so the pool never runs dry.
        self.assertLessEqual(device["max_in_use"], num_threads)
        self.assertEqual(snapshot["pool_exhaustions"], 0)
        available = cuda_event_pool._INSTALLATION.pools[0]._available
        self.assertEqual(len({id(event) for event in available}), 16)

    def test_failed_event_is_quarantined(self) -> None:
        self.install(pool_size=2)
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

    def test_second_module_copy_does_not_patch_again(self) -> None:
        self.install(pool_size=2)
        spec = importlib.util.spec_from_file_location(
            "_cuda_event_pool_copy", cuda_event_pool.__file__
        )
        module_copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_copy)

        with self.assertRaisesRegex(RuntimeError, "already patched"):
            module_copy.install_cuda_event_pool(torch_module=self.torch, pool_size=2)

    def test_uninstall_restores_original_method(self) -> None:
        original = FakeStream.wait_stream
        self.install(pool_size=2)

        self.assertIsNot(FakeStream.wait_stream, original)
        self.assertTrue(cuda_event_pool.uninstall_cuda_event_pool())
        self.assertIs(FakeStream.wait_stream, original)

    def test_call_in_flight_after_uninstall_uses_original_path(self) -> None:
        self.install(pool_size=2)
        patched = FakeStream.wait_stream
        cuda_event_pool.uninstall_cuda_event_pool()
        destination = FakeStream(0)

        patched(destination, FakeStream(0))

        self.assertEqual(len(destination.waited_events), 1)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["unprewarmed_fallbacks"], 1)
        self.assertEqual(snapshot["pooled_calls"], 0)

    def test_reinstall_uses_new_pool_size(self) -> None:
        self.install(pool_size=2)
        self.assertTrue(cuda_event_pool.uninstall_cuda_event_pool())

        self.install(pool_size=3)

        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["pool_size"], 3)
        self.assertEqual(snapshot["devices"]["0"]["size"], 3)


class TestCudaEventPoolFlag(unittest.TestCase):
    def test_flag_off_leaves_wait_stream_untouched(self) -> None:
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.model_runner import ModelRunner

        cuda_event_pool._reset_cuda_event_pool_for_testing()
        original = torch.cuda.Stream.wait_stream
        platform = SimpleNamespace(is_cuda=lambda: True)
        with (
            envs.SGLANG_ENABLE_CUDA_EVENT_POOL.override(False),
            patch("sglang.srt.model_executor.model_runner.current_platform", platform),
        ):
            ModelRunner.init_cuda_event_pool(SimpleNamespace())

        self.assertIs(torch.cuda.Stream.wait_stream, original)
        self.assertFalse(cuda_event_pool.cuda_event_pool_stats()["installed"])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCudaEventPoolOnDevice(unittest.TestCase):
    def setUp(self) -> None:
        cuda_event_pool._reset_cuda_event_pool_for_testing()
        torch.cuda.synchronize()

    def tearDown(self) -> None:
        torch.cuda.synchronize()
        cuda_event_pool._reset_cuda_event_pool_for_testing()

    def install(self, pool_size: int, devices: tuple[int, ...] = (0,)) -> None:
        cuda_event_pool.install_cuda_event_pool(torch_module=torch, pool_size=pool_size)
        for device in devices:
            self.assertIsNotNone(cuda_event_pool.prewarm_cuda_event_pool(device))

    def assert_ordered(self, source, destination, value, observed) -> None:
        """Each copy on ``destination`` must see the write that ``source``
        makes after a delay, which only ``wait_stream`` enforces."""
        torch.cuda.synchronize()
        for i in range(observed.numel()):
            with torch.cuda.stream(source):
                torch.cuda._sleep(_SLEEP_CYCLES)
                value.fill_(i + 1)
            destination.wait_stream(source)
            with torch.cuda.stream(destination):
                observed[i : i + 1].copy_(value)
            source.wait_stream(destination)
        torch.cuda.synchronize()

        expected = torch.arange(1, observed.numel() + 1, dtype=observed.dtype)
        self.assertEqual(observed.cpu().tolist(), expected.tolist())

    def test_wait_stream_orders_delayed_writes(self) -> None:
        self.install(pool_size=8)
        value = torch.zeros(1, device="cuda")
        observed = torch.zeros(16, device="cuda")

        self.assert_ordered(torch.cuda.Stream(), torch.cuda.Stream(), value, observed)

        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["events_materialized"], 8)
        self.assertEqual(snapshot["pooled_calls"], 32)
        self.assertEqual(snapshot["original_calls"], 0)
        self.assertEqual(snapshot["devices"]["0"]["in_use"], 0)

    def test_rerecorded_event_keeps_enqueued_wait(self) -> None:
        self.install(pool_size=1)
        stream_a, stream_b, stream_c, stream_d = (torch.cuda.Stream() for _ in range(4))
        value = torch.zeros(1, device="cuda")
        observed = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

        with torch.cuda.stream(stream_a):
            torch.cuda._sleep(5 * _SLEEP_CYCLES)
            value.fill_(42)
        stream_b.wait_stream(stream_a)
        # The only event is re-recorded on idle C and completes at once; B's
        # wait was enqueued first and must still wait for A.
        stream_d.wait_stream(stream_c)
        with torch.cuda.stream(stream_b):
            observed.copy_(value)
        torch.cuda.synchronize()

        self.assertEqual(observed.item(), 42)
        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["pooled_calls"], 2)
        self.assertEqual(snapshot["devices"]["0"]["size"], 1)

    def test_graph_capture_uses_original_path_and_replays(self) -> None:
        self.install(pool_size=8)
        side = torch.cuda.Stream()
        x = torch.arange(4, device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            before = cuda_event_pool.cuda_event_pool_stats()
            x.add_(1)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                y.copy_(x * 2)
            torch.cuda.current_stream().wait_stream(side)
            after = cuda_event_pool.cuda_event_pool_stats()

        self.assertEqual(after["capture_bypasses"] - before["capture_bypasses"], 2)
        self.assertEqual(after["pooled_calls"], before["pooled_calls"])
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(y.cpu().tolist(), [4.0, 6.0, 8.0, 10.0])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 GPUs")
    def test_cross_device_wait_orders_delayed_writes(self) -> None:
        self.install(pool_size=8, devices=(0, 1))
        with torch.cuda.device(1):
            source = torch.cuda.Stream()
        destination = torch.cuda.Stream(device=0)
        value = torch.zeros(1, device="cuda:1")
        observed = torch.zeros(16, device="cuda:0")

        self.assert_ordered(source, destination, value, observed)

        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["pooled_calls"], 32)
        self.assertEqual(snapshot["pooled_call_failures"], 0)
        self.assertEqual(snapshot["devices"]["0"]["in_use"], 0)
        self.assertEqual(snapshot["devices"]["1"]["in_use"], 0)

    def test_generic_stream_source_orders_delayed_writes(self) -> None:
        self.install(pool_size=8)
        source = torch.Stream(device="cuda")
        self.assertNotIsInstance(source, torch.cuda.Stream)
        value = torch.zeros(1, device="cuda")
        observed = torch.zeros(16, device="cuda")

        self.assert_ordered(source, torch.cuda.Stream(), value, observed)

        snapshot = cuda_event_pool.cuda_event_pool_stats()
        self.assertEqual(snapshot["non_cuda_source_fallbacks"], 16)
        self.assertEqual(snapshot["pooled_call_failures"], 0)
        self.assertEqual(snapshot["devices"]["0"]["quarantined"], 0)


if __name__ == "__main__":
    unittest.main()

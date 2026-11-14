"""Unit tests for ``device_memory_monitor`` — CPU-only.

Covers the per-rank device-memory accounting and the opt-in all-rank
allocator-cache release added for sgl-project/sglang#46831:

  * ``DeviceMemorySample`` arithmetic: the allocator's cached bytes and the
    driver-allocated memory that the Torch allocator does not own.
  * ``observe()``: only decode passes count; the log cadence and the batch
    shape that is recorded alongside each reading.
  * The release policy: it fires only below the driver-reported free-memory
    floor, only on the configured iteration cadence, it measures the pause it
    causes, and -- the point of the whole exercise -- it fires
    **independently on each rank**, so a rank whose own free memory is low
    releases even while its peers are fine.
  * ``maybe_create()``: the monitor stays disabled unless the
    ``SGLANG_DEVICE_MEMORY_*`` knobs ask for it, and the environment defaults
    are asserted so a future change cannot silently turn the policy on.
  * Allocator snapshots: written on demand, the baseline only once, and
    carrying the driver reading next to the allocator's segment table.

The CUDA / HIP APIs are mocked; everything under test is pure Python.
"""

import inspect
import json
import logging
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.model_executor import device_memory_monitor as mod
from sglang.srt.model_executor.device_memory_monitor import (
    DeviceMemoryMonitor,
    DeviceMemorySample,
    read_device_memory,
    read_free_bytes,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MIB = 1024 * 1024
GIB = 1024 * MIB


class _CapturingLogger(logging.Logger):
    """Logger that keeps every JSON payload it was handed."""

    def __init__(self):
        super().__init__("device-memory-test", logging.INFO)
        self.records = []

    def info(self, msg, *args, **kwargs):
        self.records.append(json.loads(msg))

    def events(self):
        return [r["event"] for r in self.records]


def _sample(
    *,
    allocated=0,
    reserved=0,
    free=None,
    total=None,
    stats=None,
):
    return DeviceMemorySample(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        free_bytes=free,
        total_bytes=total,
        stats=stats or {},
    )


def _monitor(**kwargs):
    """A monitor with every device touch point faked out."""
    logger = _CapturingLogger()
    default_free = kwargs.pop("_free", 0)
    kwargs.setdefault("loggers", [logger])
    kwargs.setdefault("memory_reader", lambda: _sample(allocated=1, reserved=2))
    kwargs.setdefault("free_reader", lambda: default_free)
    kwargs.setdefault("rank_info", {"tp_rank": 0})
    monitor = DeviceMemoryMonitor(**kwargs)
    monitor.capturing_logger = logger
    return monitor


class TestDeviceMemorySample(CustomTestCase):
    def test_cached_bytes_is_reserved_minus_allocated(self):
        sample = _sample(allocated=int(0.8 * GIB), reserved=int(15.7 * GIB))
        self.assertAlmostEqual(sample.cached_bytes / GIB, 14.9, places=3)

    def test_cached_bytes_never_negative(self):
        sample = _sample(allocated=100, reserved=50)
        self.assertEqual(sample.cached_bytes, 0)

    def test_unaccounted_bytes_is_driver_memory_torch_does_not_own(self):
        # 64 GiB device, 0.92 GiB free -> 63.08 GiB handed out by the driver,
        # of which Torch reserved 15.7 GiB.
        sample = _sample(
            allocated=int(0.8 * GIB),
            reserved=int(15.7 * GIB),
            free=int(0.92 * GIB),
            total=64 * GIB,
        )
        expected = 64 * GIB - int(0.92 * GIB) - int(15.7 * GIB)
        self.assertEqual(sample.unaccounted_bytes, expected)

    def test_unaccounted_bytes_is_none_without_driver_readings(self):
        sample = _sample(allocated=1, reserved=2)
        self.assertIsNone(sample.unaccounted_bytes)

    def test_unaccounted_bytes_never_negative(self):
        # A driver that reports more free than "total - reserved" must not
        # produce a negative footprint.
        sample = _sample(allocated=1, reserved=10, free=1000, total=100)
        self.assertEqual(sample.unaccounted_bytes, 0)

    def test_to_json_omits_absent_driver_readings(self):
        data = _sample(allocated=1, reserved=2).to_json()
        self.assertIn("allocated_bytes", data)
        self.assertIn("reserved_bytes", data)
        self.assertIn("cached_bytes", data)
        self.assertNotIn("free_bytes", data)
        self.assertNotIn("total_bytes", data)
        self.assertNotIn("unaccounted_bytes", data)

    def test_to_json_includes_driver_readings_when_present(self):
        data = _sample(allocated=1, reserved=2, free=3, total=4).to_json()
        self.assertEqual(data["free_bytes"], 3)
        self.assertEqual(data["total_bytes"], 4)
        self.assertEqual(data["unaccounted_bytes"], 0)


class TestObserve(CustomTestCase):
    def test_non_decode_passes_are_ignored(self):
        monitor = _monitor(log_interval=1)
        monitor.observe(is_decode=False, batch_size=4, forward_mode="EXTEND")
        self.assertEqual(monitor.decode_iterations, 0)
        self.assertEqual(monitor.capturing_logger.records, [])

    def test_decode_passes_are_counted(self):
        monitor = _monitor(log_interval=1)
        monitor.observe(is_decode=True, batch_size=4, forward_mode="DECODE")
        monitor.observe(is_decode=True, batch_size=4, forward_mode="DECODE")
        self.assertEqual(monitor.decode_iterations, 2)

    def test_logging_honours_the_interval(self):
        monitor = _monitor(log_interval=3)
        for _ in range(7):
            monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        samples = [
            r
            for r in monitor.capturing_logger.records
            if r["event"] == "device_memory.sample"
        ]
        # Iterations 3 and 6.
        self.assertEqual([r["decode_iterations"] for r in samples], [3, 6])

    def test_logging_is_off_by_default(self):
        monitor = _monitor(log_interval=0)
        for _ in range(5):
            monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        self.assertEqual(monitor.capturing_logger.records, [])

    def test_sample_log_carries_the_batch_shape(self):
        monitor = _monitor(log_interval=1)
        monitor.observe(
            is_decode=True,
            batch_size=16,
            num_tokens=16,
            forward_mode="DECODE",
            can_run_graph=True,
        )
        record = monitor.capturing_logger.records[0]
        self.assertEqual(record["event"], "device_memory.sample")
        self.assertEqual(record["batch_size"], 16)
        self.assertEqual(record["num_tokens"], 16)
        self.assertEqual(record["forward_mode"], "DECODE")
        self.assertTrue(record["can_run_graph"])
        self.assertEqual(record["tp_rank"], 0)

    def test_sample_log_carries_the_memory_reading(self):
        monitor = _monitor(
            log_interval=1,
            memory_reader=lambda: _sample(
                allocated=int(0.8 * GIB),
                reserved=int(15.7 * GIB),
                free=int(0.92 * GIB),
                total=64 * GIB,
            ),
        )
        monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        record = monitor.capturing_logger.records[0]
        self.assertEqual(record["allocated_bytes"], int(0.8 * GIB))
        self.assertEqual(record["reserved_bytes"], int(15.7 * GIB))
        self.assertEqual(record["free_bytes"], int(0.92 * GIB))
        self.assertAlmostEqual(record["cached_bytes"] / GIB, 14.9, places=3)
        self.assertIn("unaccounted_bytes", record)


class TestReleasePolicy(CustomTestCase):
    def test_no_release_without_a_threshold(self):
        released = []
        monitor = _monitor(
            free_reader=lambda: 1, cache_release=lambda: released.append(1)
        )
        for _ in range(20):
            monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        self.assertEqual(released, [])
        self.assertEqual(monitor.release_count, 0)

    def test_release_fires_below_the_free_memory_floor(self):
        released = []
        monitor = _monitor(
            free_threshold_bytes=6 * 1024 * MIB,
            release_interval=1,
            free_reader=lambda: 5 * 1024 * MIB,
            cache_release=lambda: released.append(1),
        )
        monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        self.assertEqual(len(released), 1)
        self.assertEqual(monitor.release_count, 1)

    def test_release_does_not_fire_above_the_floor(self):
        released = []
        monitor = _monitor(
            free_threshold_bytes=6 * 1024 * MIB,
            release_interval=1,
            free_reader=lambda: 6 * 1024 * MIB,
            cache_release=lambda: released.append(1),
        )
        monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        self.assertEqual(released, [])

    def test_release_honours_the_iteration_cadence(self):
        released = []
        monitor = _monitor(
            free_threshold_bytes=6 * 1024 * MIB,
            release_interval=8,
            free_reader=lambda: 1,
            cache_release=lambda: released.append(1),
        )
        for _ in range(24):
            monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        # Iterations 8, 16 and 24.
        self.assertEqual(len(released), 3)
        logged = [
            r["decode_iterations"]
            for r in monitor.capturing_logger.records
            if r["event"] == "device_memory.release"
        ]
        self.assertEqual(logged, [8, 16, 24])

    def test_release_measures_the_pause_it_causes(self):
        ticks = iter([1.0, 1.25, 2.0, 2.5])
        released = []
        monitor = _monitor(
            free_threshold_bytes=6 * 1024 * MIB,
            release_interval=1,
            free_reader=lambda: 1,
            cache_release=lambda: released.append(1),
            clock=lambda: next(ticks),
        )
        monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        self.assertEqual(len(released), 1)
        # 0.25 s of clock spent inside the release.
        self.assertAlmostEqual(monitor.total_release_ms, 250.0, places=3)
        self.assertAlmostEqual(monitor.max_release_ms, 250.0, places=3)

    def test_release_log_carries_before_and_after(self):
        readings = iter(
            [
                _sample(allocated=1, reserved=2, free=10, total=100),
                _sample(allocated=1, reserved=20, free=90, total=100),
            ]
        )
        monitor = _monitor(
            free_threshold_bytes=6 * MIB,
            release_interval=1,
            free_reader=lambda: 1,
            cache_release=lambda: None,
            memory_reader=lambda: next(readings),
        )
        monitor.observe(is_decode=True, batch_size=8, forward_mode="DECODE")
        record = monitor.capturing_logger.records[0]
        self.assertEqual(record["event"], "device_memory.release")
        self.assertEqual(record["free_bytes_before"], 1)
        self.assertEqual(record["before"]["reserved_bytes"], 2)
        self.assertEqual(record["after"]["reserved_bytes"], 20)
        self.assertEqual(record["batch_size"], 8)
        self.assertIn("pause_ms", record)

    def test_each_rank_releases_on_its_own_free_memory(self):
        """The regression this whole mechanism exists for.

        #46831's control ``S2k`` released only on rank 0, which left the other
        decode ranks near full and aborting.  A rank whose own free memory is
        low must therefore release even while its peers are comfortable.
        """
        released = {"rank0": [], "rank3": []}

        def make(rank):
            return _monitor(
                rank_info={"tp_rank": rank},
                free_threshold_bytes=6 * 1024 * MIB,
                release_interval=1,
                free_reader=lambda: 10 * 1024 * MIB if rank == 0 else 2 * 1024 * MIB,
                cache_release=lambda: released[f"rank{rank}"].append(1),
            )

        rank0, rank3 = make(0), make(3)
        rank0.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        rank3.observe(is_decode=True, batch_size=1, forward_mode="DECODE")

        self.assertEqual(released["rank0"], [])
        self.assertEqual(len(released["rank3"]), 1)

    def test_release_is_skipped_when_the_driver_cannot_be_read(self):
        released = []
        monitor = _monitor(
            free_threshold_bytes=6 * MIB,
            release_interval=1,
            free_reader=lambda: None,
            cache_release=lambda: released.append(1),
        )
        monitor.observe(is_decode=True, batch_size=1, forward_mode="DECODE")
        self.assertEqual(released, [])


class TestMaybeCreate(CustomTestCase):
    def _clear_env(self):
        for name in (
            "SGLANG_DEVICE_MEMORY_LOG_INTERVAL",
            "SGLANG_DEVICE_MEMORY_LOG_TARGET",
            "SGLANG_DEVICE_MEMORY_FREE_THRESHOLD_MB",
            "SGLANG_DEVICE_MEMORY_RELEASE_INTERVAL",
            "SGLANG_DEVICE_MEMORY_SNAPSHOT_DIR",
        ):
            os.environ.pop(name, None)

    def setUp(self):
        self._clear_env()
        self.addCleanup(self._clear_env)

    def test_disabled_when_nothing_is_requested(self):
        self.assertIsNone(DeviceMemoryMonitor.maybe_create())

    def test_disabled_without_a_device(self):
        os.environ["SGLANG_DEVICE_MEMORY_LOG_INTERVAL"] = "1"
        with mock.patch.object(mod.torch.cuda, "is_available", return_value=False):
            self.assertIsNone(DeviceMemoryMonitor.maybe_create())

    def test_created_when_logging_is_requested(self):
        os.environ["SGLANG_DEVICE_MEMORY_LOG_INTERVAL"] = "5"
        with mock.patch.object(mod.torch.cuda, "is_available", return_value=True):
            monitor = DeviceMemoryMonitor.maybe_create()
        self.assertIsNotNone(monitor)
        self.assertEqual(monitor.log_interval, 5)
        self.assertIsNone(monitor.free_threshold_bytes)

    def test_created_when_the_free_memory_floor_is_requested(self):
        os.environ["SGLANG_DEVICE_MEMORY_FREE_THRESHOLD_MB"] = "6144"
        os.environ["SGLANG_DEVICE_MEMORY_RELEASE_INTERVAL"] = "8"
        with mock.patch.object(mod.torch.cuda, "is_available", return_value=True):
            monitor = DeviceMemoryMonitor.maybe_create()
        self.assertIsNotNone(monitor)
        self.assertEqual(monitor.free_threshold_bytes, 6144 * MIB)
        self.assertEqual(monitor.release_interval, 8)

    def test_release_interval_is_never_zero(self):
        os.environ["SGLANG_DEVICE_MEMORY_FREE_THRESHOLD_MB"] = "1024"
        os.environ["SGLANG_DEVICE_MEMORY_RELEASE_INTERVAL"] = "0"
        with mock.patch.object(mod.torch.cuda, "is_available", return_value=True):
            monitor = DeviceMemoryMonitor.maybe_create()
        self.assertEqual(monitor.release_interval, 1)

    def test_rank_info_is_merged(self):
        os.environ["SGLANG_DEVICE_MEMORY_LOG_INTERVAL"] = "1"
        with mock.patch.object(mod.torch.cuda, "is_available", return_value=True):
            monitor = DeviceMemoryMonitor.maybe_create(
                rank_info={"is_draft_worker": True}
            )
        self.assertTrue(monitor.rank_info["is_draft_worker"])

    def test_snapshot_dir_alone_enables_the_monitor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SGLANG_DEVICE_MEMORY_SNAPSHOT_DIR"] = tmpdir
            with mock.patch.object(mod.torch.cuda, "is_available", return_value=True):
                monitor = DeviceMemoryMonitor.maybe_create()
        self.assertIsNotNone(monitor)
        self.assertIsNotNone(monitor.snapshot_dir)


class TestEnvDefaults(CustomTestCase):
    """The defaults must keep the engine's current behavior."""

    def test_defaults_are_off(self):
        for name in (
            "SGLANG_DEVICE_MEMORY_LOG_INTERVAL",
            "SGLANG_DEVICE_MEMORY_LOG_TARGET",
            "SGLANG_DEVICE_MEMORY_FREE_THRESHOLD_MB",
            "SGLANG_DEVICE_MEMORY_SNAPSHOT_DIR",
        ):
            os.environ.pop(name, None)
        self.assertEqual(envs.SGLANG_DEVICE_MEMORY_LOG_INTERVAL.get(), 0)
        self.assertEqual(envs.SGLANG_DEVICE_MEMORY_LOG_TARGET.get(), "")
        self.assertEqual(envs.SGLANG_DEVICE_MEMORY_FREE_THRESHOLD_MB.get(), 0)
        self.assertEqual(envs.SGLANG_DEVICE_MEMORY_SNAPSHOT_DIR.get(), "")

    def test_release_interval_default_matches_the_reported_policy(self):
        # #46831's all-rank release ran "every eight decode iterations".
        os.environ.pop("SGLANG_DEVICE_MEMORY_RELEASE_INTERVAL", None)
        self.assertEqual(envs.SGLANG_DEVICE_MEMORY_RELEASE_INTERVAL.get(), 8)


class TestSnapshots(CustomTestCase):
    def test_snapshot_is_written_with_the_driver_reading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = _monitor(snapshot_dir=tmpdir)
            with mock.patch.object(
                mod.torch.cuda, "memory_snapshot", return_value={"segments": []}
            ):
                with mock.patch.object(mod, "read_free_bytes", return_value=4096):
                    path = monitor._write_snapshot("before_release")
            self.assertIsNotNone(path)
            with open(path) as f:
                payload = json.load(f)
            self.assertEqual(payload["reason"], "before_release")
            self.assertEqual(payload["free_bytes"], 4096)
            self.assertIn("allocator_snapshot", payload)
            self.assertEqual(payload["rank_info"], {"tp_rank": 0})

    def test_baseline_snapshot_is_written_only_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = _monitor(snapshot_dir=tmpdir)
            with mock.patch.object(
                mod.torch.cuda, "memory_snapshot", return_value={"segments": []}
            ):
                first = monitor.write_baseline_snapshot()
                second = monitor.write_baseline_snapshot()
            self.assertIsNotNone(first)
            self.assertIsNone(second)

    def test_no_snapshot_without_a_directory(self):
        monitor = _monitor()
        self.assertIsNone(monitor._write_snapshot("before_release"))

    def test_snapshot_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = _monitor(snapshot_dir=tmpdir)
            with mock.patch.object(
                mod.torch.cuda,
                "memory_snapshot",
                side_effect=RuntimeError("no cuda"),
            ):
                self.assertIsNone(monitor._write_snapshot("before_release"))


class TestReaders(CustomTestCase):
    def test_read_device_memory_survives_a_driver_failure(self):
        with mock.patch.object(mod.torch.cuda, "memory_allocated", return_value=10):
            with mock.patch.object(mod.torch.cuda, "memory_reserved", return_value=20):
                with mock.patch.object(
                    mod.torch.cuda, "mem_get_info", side_effect=RuntimeError("no cuda")
                ):
                    with mock.patch.object(
                        mod.torch.cuda,
                        "memory_stats",
                        side_effect=RuntimeError("no cuda"),
                    ):
                        sample = read_device_memory()
        self.assertEqual(sample.allocated_bytes, 10)
        self.assertEqual(sample.reserved_bytes, 20)
        self.assertIsNone(sample.free_bytes)
        self.assertEqual(sample.stats, {})

    def test_read_free_bytes_survives_a_driver_failure(self):
        with mock.patch.object(
            mod.torch.cuda, "mem_get_info", side_effect=RuntimeError("no cuda")
        ):
            self.assertIsNone(read_free_bytes())

    def test_read_device_memory_keeps_the_curated_stats(self):
        with mock.patch.object(mod.torch.cuda, "memory_allocated", return_value=1):
            with mock.patch.object(mod.torch.cuda, "memory_reserved", return_value=2):
                with mock.patch.object(
                    mod.torch.cuda, "mem_get_info", return_value=(3, 4)
                ):
                    with mock.patch.object(
                        mod.torch.cuda,
                        "memory_stats",
                        return_value={
                            "allocated_bytes.all.current": 1,
                            "num_device_alloc": 7,
                            "some_uninteresting_key": 9,
                        },
                    ):
                        sample = read_device_memory()
        self.assertEqual(sample.free_bytes, 3)
        self.assertEqual(sample.total_bytes, 4)
        self.assertEqual(sample.stats["num_device_alloc"], 7)
        self.assertNotIn("some_uninteresting_key", sample.stats)


class TestModelRunnerWiring(CustomTestCase):
    """The monitor has to be driven from the serving loop.

    Constructing a ``ModelRunner`` needs a model, a server and a device, so
    the wiring is checked on the source instead: the monitor must be created
    in ``__init__`` and observed from ``forward``.
    """

    def test_model_runner_creates_the_monitor(self):
        src = inspect.getsource(ModelRunner.__init__)
        self.assertIn("DeviceMemoryMonitor.maybe_create(", src)

    def test_model_runner_observes_the_monitor_in_forward(self):
        src = inspect.getsource(ModelRunner.forward)
        self.assertIn("self.device_memory_monitor.observe(", src)

    def test_model_runner_takes_the_baseline_snapshot_after_capture(self):
        src = inspect.getsource(ModelRunner.init_cuda_graphs)
        self.assertIn("write_baseline_snapshot()", src)


if __name__ == "__main__":
    unittest.main()

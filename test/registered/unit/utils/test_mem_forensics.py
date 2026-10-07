"""Unit tests for allocator-history forensics gating and dump lifecycle."""

import os
import pickle
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import mem_forensics
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

LOGGER = "sglang.srt.utils.mem_forensics"
WITH_HISTORY = {"segments": [], "device_traces": [[{"action": "alloc"}]]}
NO_HISTORY = {"segments": [], "device_traces": [[], []]}


class MemForensicsTest(unittest.TestCase):
    def setUp(self):
        mem_forensics._started = False
        mem_forensics._dumped_tags = set()

    def test_disabled_without_directory(self):
        with envs.SGLANG_MEM_FORENSICS_DIR.override(None):
            with mock.patch.object(
                torch.cuda.memory, "_record_memory_history"
            ) as record:
                mem_forensics.maybe_start_memory_forensics()
        self.assertFalse(record.called)
        self.assertFalse(mem_forensics._started)

    def test_start_records_once_with_configured_parameters(self):
        with envs.SGLANG_MEM_FORENSICS_DIR.override("/tmp/forensics"):
            with envs.SGLANG_MEM_FORENSICS_MAX_ENTRIES.override(1234):
                with (
                    mock.patch.object(torch.cuda, "is_available", return_value=True),
                    mock.patch.object(
                        torch.cuda.memory, "_record_memory_history"
                    ) as record,
                ):
                    mem_forensics.maybe_start_memory_forensics()
                    mem_forensics.maybe_start_memory_forensics()
        self.assertEqual(record.call_count, 1)
        self.assertEqual(record.call_args.kwargs["enabled"], "all")
        self.assertEqual(record.call_args.kwargs["context"], "all")
        self.assertEqual(record.call_args.kwargs["stacks"], "python")
        self.assertEqual(record.call_args.kwargs["max_entries"], 1234)
        self.assertTrue(mem_forensics._started)

    def test_start_failure_is_contained_and_not_marked_started(self):
        with envs.SGLANG_MEM_FORENSICS_DIR.override("/tmp/forensics"):
            with (
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(
                    torch.cuda.memory,
                    "_record_memory_history",
                    side_effect=RuntimeError("driver"),
                ),
            ):
                mem_forensics.maybe_start_memory_forensics()
        self.assertFalse(mem_forensics._started)

    def test_dump_writes_one_snapshot_per_tag(self):
        snapshot = WITH_HISTORY
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics._started = True
                with (
                    mock.patch.object(
                        torch.cuda.memory, "_snapshot", return_value=snapshot
                    ),
                    mock.patch.object(
                        torch.cuda.memory, "_record_memory_history"
                    ) as record,
                ):
                    mem_forensics.maybe_dump_memory_forensics("ready")
                    mem_forensics.maybe_dump_memory_forensics("ready")
                    mem_forensics.maybe_dump_memory_forensics("corruption")
            names = sorted(os.listdir(out_dir))
            self.assertEqual(len(names), 2)
            self.assertEqual(
                sorted(name.split("-")[2] for name in names),
                ["corruption", "ready"],
            )
            for name in names:
                self.assertIn(f"pid{os.getpid()}", name)
                self.assertTrue(name.endswith(".pickle"))
            with open(os.path.join(out_dir, names[0]), "rb") as file:
                self.assertEqual(pickle.load(file), snapshot)
        # History was present, so nothing was re-armed.
        self.assertFalse(record.called)

    def test_dump_is_noop_before_start(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics.maybe_dump_memory_forensics("ready")
            self.assertEqual(os.listdir(out_dir), [])

    def test_failed_dump_never_raises_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics._started = True
                with mock.patch.object(
                    torch.cuda.memory,
                    "_snapshot",
                    side_effect=RuntimeError("faulted context"),
                ):
                    mem_forensics.maybe_dump_memory_forensics("corruption")
                self.assertEqual(os.listdir(out_dir), [])
                with mock.patch.object(
                    torch.cuda.memory, "_snapshot", return_value=WITH_HISTORY
                ):
                    mem_forensics.maybe_dump_memory_forensics("corruption")
            names = os.listdir(out_dir)
            self.assertEqual(len(names), 1)

    def test_failed_write_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics._started = True
                with (
                    mock.patch.object(
                        torch.cuda.memory, "_snapshot", return_value=WITH_HISTORY
                    ),
                    mock.patch.object(pickle, "dump", side_effect=OSError("disk full")),
                ):
                    mem_forensics.maybe_dump_memory_forensics("ready")
            self.assertEqual(os.listdir(out_dir), [])
            self.assertNotIn("ready", mem_forensics._dumped_tags)

    def test_dump_without_history_writes_rearms_and_keeps_tag_open(self):
        # A MEM torch profile ran and stopped the process-wide recorder,
        # which drops the events; live blocks still carry their stacks.
        with tempfile.TemporaryDirectory() as out_dir:
            with (
                envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir),
                envs.SGLANG_MEM_FORENSICS_MAX_ENTRIES.override(4321),
            ):
                mem_forensics._started = True
                with (
                    mock.patch.object(
                        torch.cuda.memory, "_snapshot", return_value=NO_HISTORY
                    ),
                    mock.patch.object(
                        torch.cuda.memory, "_record_memory_history"
                    ) as record,
                    self.assertLogs(LOGGER, level="WARNING") as logs,
                ):
                    mem_forensics.maybe_dump_memory_forensics("retained-kda:extend")
                # Written, re-armed with the configured parameters, tag open.
                self.assertEqual(record.call_count, 1)
                self.assertEqual(record.call_args.kwargs["enabled"], "all")
                self.assertEqual(record.call_args.kwargs["stacks"], "python")
                self.assertEqual(record.call_args.kwargs["max_entries"], 4321)
                self.assertEqual(len(os.listdir(out_dir)), 1)
                self.assertNotIn("retained-kda:extend", mem_forensics._dumped_tags)
                self.assertTrue(any("re-arming" in line for line in logs.output))
                # The next request for the same tag finds history, writes a
                # second snapshot and consumes the tag.
                with (
                    mock.patch.object(
                        torch.cuda.memory, "_snapshot", return_value=WITH_HISTORY
                    ),
                    mock.patch.object(
                        torch.cuda.memory, "_record_memory_history"
                    ) as record,
                ):
                    mem_forensics.maybe_dump_memory_forensics("retained-kda:extend")
                    mem_forensics.maybe_dump_memory_forensics("retained-kda:extend")
                self.assertFalse(record.called)
                names = os.listdir(out_dir)
                self.assertEqual(len(names), 2)
                self.assertTrue(all("retained-kda:extend" in name for name in names))
            self.assertIn("retained-kda:extend", mem_forensics._dumped_tags)

    def test_rearm_failure_is_contained_and_still_writes(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics._started = True
                with (
                    mock.patch.object(
                        torch.cuda.memory, "_snapshot", return_value=NO_HISTORY
                    ),
                    mock.patch.object(
                        torch.cuda.memory,
                        "_record_memory_history",
                        side_effect=RuntimeError("driver"),
                    ),
                    self.assertLogs(LOGGER, level="ERROR"),
                ):
                    mem_forensics.maybe_dump_memory_forensics("ready")
            self.assertEqual(len(os.listdir(out_dir)), 1)
            self.assertNotIn("ready", mem_forensics._dumped_tags)

    def test_dump_after_directory_is_unset_is_noop(self):
        mem_forensics._started = True
        with envs.SGLANG_MEM_FORENSICS_DIR.override(None):
            with mock.patch.object(torch.cuda.memory, "_snapshot") as snapshot:
                mem_forensics.maybe_dump_memory_forensics("corruption")
        self.assertFalse(snapshot.called)
        self.assertNotIn("corruption", mem_forensics._dumped_tags)

    def test_stop_memory_history_stops_without_forensics(self):
        with mock.patch.object(torch.cuda.memory, "_record_memory_history") as record:
            mem_forensics.stop_memory_history()
        record.assert_called_once_with(enabled=None)

    def test_stop_memory_history_keeps_forensics_recording(self):
        mem_forensics._started = True
        with envs.SGLANG_MEM_FORENSICS_MAX_ENTRIES.override(4321):
            with mock.patch.object(
                torch.cuda.memory, "_record_memory_history"
            ) as record:
                mem_forensics.stop_memory_history()
        record.assert_called_once()
        self.assertEqual(record.call_args.kwargs["enabled"], "all")
        self.assertEqual(record.call_args.kwargs["stacks"], "python")
        self.assertEqual(record.call_args.kwargs["max_entries"], 4321)


class _FakeRecorder:
    """Process-wide allocator recorder: stopping it drops the recorded
    events, as the CUDA caching allocator does, while live blocks keep the
    stacks they were allocated with."""

    BLOCKS = [{"segments": [{"blocks": [{"frames": [{"name": "capture"}]}]}]}]

    def __init__(self):
        self.events = []

    def record(self, enabled="all", **kwargs):
        if enabled is None:
            self.events.clear()
        else:
            self.events.append({"action": "alloc"})

    def snapshot(self):
        return {"segments": self.BLOCKS, "device_traces": [list(self.events)]}


class CaptureProfilerReadySnapshotTest(unittest.TestCase):
    """Startup with ``--enable-profile-cuda-graph``: forensics starts, the
    capture profiler records and then stops, the scheduler asks for its one
    ``ready`` snapshot."""

    def setUp(self):
        mem_forensics._started = False
        mem_forensics._dumped_tags = set()

    def _startup(self, out_dir, stop_profiler):
        recorder = _FakeRecorder()
        with (
            envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda.memory, "_record_memory_history", recorder.record
            ),
            mock.patch.object(torch.cuda.memory, "_snapshot", recorder.snapshot),
        ):
            mem_forensics.maybe_start_memory_forensics()
            torch.cuda.memory._record_memory_history()  # capture profiler
            stop_profiler()
            mem_forensics.maybe_dump_memory_forensics("ready")
        names = os.listdir(out_dir)
        self.assertEqual(len(names), 1)
        with open(os.path.join(out_dir, names[0]), "rb") as file:
            return pickle.load(file)

    def test_capture_profiler_keeps_forensics_history(self):
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner

        runner = mock.Mock(spec=[])

        def post_process():
            with (
                mock.patch.object(torch.cuda.memory, "_dump_snapshot"),
                mock.patch.object(
                    decode_cuda_graph_runner, "export_cuda_graph_capture_trace"
                ),
            ):
                decode_cuda_graph_runner.DecodeCudaGraphRunner._post_process_after_profile(
                    runner, mock.MagicMock()
                )

        with tempfile.TemporaryDirectory() as out_dir:
            snapshot = self._startup(out_dir, post_process)
        self.assertTrue(snapshot["device_traces"][0])
        self.assertIn("ready", mem_forensics._dumped_tags)

    def test_other_profiler_stop_still_writes_ready(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertLogs(LOGGER, level="WARNING"):
                snapshot = self._startup(
                    out_dir,
                    lambda: torch.cuda.memory._record_memory_history(enabled=None),
                )
        self.assertEqual(snapshot["device_traces"], [[]])
        self.assertEqual(snapshot["segments"], _FakeRecorder.BLOCKS)


if __name__ == "__main__":
    unittest.main()

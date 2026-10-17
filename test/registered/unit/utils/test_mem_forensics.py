"""Unit tests for allocator-history forensics gating and dump lifecycle."""

import os
import pickle
import sys
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
        mem_forensics._max_entries = 0
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
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics._started = True
                mem_forensics._max_entries = 4321
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

    def test_start_memory_history_uses_defaults_without_forensics(self):
        with mock.patch.object(torch.cuda.memory, "_record_memory_history") as record:
            mem_forensics.start_memory_history()
            mem_forensics.stop_memory_history()
        self.assertEqual(record.call_args_list, [mock.call(), mock.call(enabled=None)])

    def test_profiler_hooks_keep_forensics_configuration(self):
        mem_forensics._started = True
        mem_forensics._max_entries = 4321
        with envs.SGLANG_MEM_FORENSICS_DIR.override("/tmp/forensics"):
            with mock.patch.object(
                torch.cuda.memory, "_record_memory_history"
            ) as record:
                mem_forensics.start_memory_history()
                mem_forensics.stop_memory_history()
        # Re-armed with the configuration recording started with, never
        # reconfigured to the profiler defaults, never stopped.
        record.assert_called_once()
        self.assertEqual(record.call_args.kwargs["enabled"], "all")
        self.assertEqual(record.call_args.kwargs["stacks"], "python")
        self.assertEqual(record.call_args.kwargs["max_entries"], 4321)

    def test_profiler_hooks_after_directory_is_unset(self):
        # Forensics no longer owns the recorder: the profiler's own default
        # start and stop apply, without re-enabling forensics recording.
        mem_forensics._started = True
        mem_forensics._max_entries = 4321
        with envs.SGLANG_MEM_FORENSICS_DIR.override(None):
            with mock.patch.object(
                torch.cuda.memory, "_record_memory_history"
            ) as record:
                mem_forensics.start_memory_history()
                mem_forensics.stop_memory_history()
        self.assertEqual(record.call_args_list, [mock.call(), mock.call(enabled=None)])


class _FakeRecorder:
    """Process-wide allocator recorder modeled on the CUDA caching
    allocator's ring buffer: reconfiguring changes the capacity without
    resizing the buffer or resetting its cursor, stopping drops the events,
    and live blocks keep the stacks they were allocated with."""

    BLOCKS = [{"blocks": [{"frames": [{"name": "capture"}]}]}]

    def __init__(self):
        self.enabled = False
        self.max_entries = 1
        self.trace = []
        self.next = 0

    def record(self, enabled="all", max_entries=sys.maxsize, **kwargs):
        self.enabled = enabled is not None
        self.max_entries = max(1, max_entries)
        if not self.enabled:
            self.trace.clear()
            self.next = 0

    def alloc(self, name):
        if not self.enabled:
            return
        if len(self.trace) < self.max_entries:
            self.trace.append(name)
        else:
            self.trace[self.next] = name
            self.next += 1
            if self.next == self.max_entries:
                self.next = 0

    def snapshot(self):
        events = self.trace[self.next :] + self.trace[: self.next]
        return {
            "segments": self.BLOCKS,
            "device_traces": [[{"action": "alloc", "name": n} for n in events]],
        }


class CaptureProfilerReadySnapshotTest(unittest.TestCase):
    """Startup with ``--enable-profile-cuda-graph``: forensics starts, the
    capture profiler records and then stops, serving allocates, and the
    scheduler asks for its one ``ready`` snapshot."""

    MAX_ENTRIES = 3

    def setUp(self):
        mem_forensics._started = False
        mem_forensics._max_entries = 0
        mem_forensics._dumped_tags = set()

    def _startup(self, out_dir, profile_capture):
        recorder = _FakeRecorder()
        with (
            envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir),
            envs.SGLANG_MEM_FORENSICS_MAX_ENTRIES.override(self.MAX_ENTRIES),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda.memory, "_record_memory_history", recorder.record
            ),
            mock.patch.object(torch.cuda.memory, "_snapshot", recorder.snapshot),
        ):
            mem_forensics.maybe_start_memory_forensics()
            recorder.alloc("init")
            profile_capture(recorder)
            for i in range(4):
                recorder.alloc(f"serve{i}")
            mem_forensics.maybe_dump_memory_forensics("ready")
        names = os.listdir(out_dir)
        self.assertEqual(len(names), 1)
        with open(os.path.join(out_dir, names[0]), "rb") as file:
            snapshot = pickle.load(file)
        return [event["name"] for event in snapshot["device_traces"][0]], snapshot

    def test_capture_profiler_keeps_forensics_history_ordered(self):
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner

        runner_cls = decode_cuda_graph_runner.DecodeCudaGraphRunner
        runner = mock.Mock(spec=[])
        runner._graph_batch_capture_active = lambda: False

        def profile_capture(recorder):
            with mock.patch.object(decode_cuda_graph_runner, "profile"):
                runner_cls._init_profile_context_and_memory_record(runner)
            for i in range(5):
                recorder.alloc(f"capture{i}")
            with (
                mock.patch.object(torch.cuda.memory, "_dump_snapshot"),
                mock.patch.object(
                    decode_cuda_graph_runner, "export_cuda_graph_capture_trace"
                ),
            ):
                runner_cls._post_process_after_profile(runner, mock.MagicMock())

        with tempfile.TemporaryDirectory() as out_dir:
            events, _ = self._startup(out_dir, profile_capture)
        # The configured capacity holds, and the newest events come last.
        self.assertEqual(events, ["serve1", "serve2", "serve3"])
        self.assertIn("ready", mem_forensics._dumped_tags)

    def test_other_profiler_stop_still_writes_ready(self):
        def profile_capture(recorder):
            torch.cuda.memory._record_memory_history()
            recorder.alloc("capture")
            torch.cuda.memory._record_memory_history(enabled=None)

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertLogs(LOGGER, level="WARNING"):
                events, snapshot = self._startup(out_dir, profile_capture)
        self.assertEqual(events, [])
        self.assertEqual(snapshot["segments"], _FakeRecorder.BLOCKS)


if __name__ == "__main__":
    unittest.main()

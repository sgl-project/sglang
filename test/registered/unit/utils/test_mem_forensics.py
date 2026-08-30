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
                with mock.patch.object(
                    torch.cuda, "is_available", return_value=True
                ), mock.patch.object(
                    torch.cuda.memory, "_record_memory_history"
                ) as record:
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
            with mock.patch.object(
                torch.cuda, "is_available", return_value=True
            ), mock.patch.object(
                torch.cuda.memory,
                "_record_memory_history",
                side_effect=RuntimeError("driver"),
            ):
                mem_forensics.maybe_start_memory_forensics()
        self.assertFalse(mem_forensics._started)

    def test_dump_writes_one_snapshot_per_tag(self):
        snapshot = WITH_HISTORY
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics._started = True
                with mock.patch.object(
                    torch.cuda.memory, "_snapshot", return_value=snapshot
                ), mock.patch.object(
                    torch.cuda.memory, "_record_memory_history"
                ) as record:
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
                with mock.patch.object(
                    torch.cuda.memory, "_snapshot", return_value=WITH_HISTORY
                ), mock.patch.object(pickle, "dump", side_effect=OSError("disk full")):
                    mem_forensics.maybe_dump_memory_forensics("ready")
            self.assertEqual(os.listdir(out_dir), [])
            self.assertNotIn("ready", mem_forensics._dumped_tags)

    def test_dump_without_history_rearms_and_defers_to_next_request(self):
        # Sequence from the review: a MEM torch profile ran and stopped the
        # process-wide recorder, then a retained phase asks for a snapshot.
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(
                out_dir
            ), envs.SGLANG_MEM_FORENSICS_MAX_ENTRIES.override(4321):
                mem_forensics._started = True
                with mock.patch.object(
                    torch.cuda.memory, "_snapshot", return_value=NO_HISTORY
                ), mock.patch.object(
                    torch.cuda.memory, "_record_memory_history"
                ) as record, self.assertLogs(
                    LOGGER, level="WARNING"
                ) as logs:
                    mem_forensics.maybe_dump_memory_forensics("retained-kda:extend")
                # Re-armed with the configured parameters, nothing written,
                # tag still open.
                self.assertEqual(record.call_count, 1)
                self.assertEqual(record.call_args.kwargs["enabled"], "all")
                self.assertEqual(record.call_args.kwargs["stacks"], "python")
                self.assertEqual(record.call_args.kwargs["max_entries"], 4321)
                self.assertEqual(os.listdir(out_dir), [])
                self.assertNotIn("retained-kda:extend", mem_forensics._dumped_tags)
                self.assertTrue(any("re-armed" in line for line in logs.output))
                # The next request for the same tag finds history and writes.
                with mock.patch.object(
                    torch.cuda.memory, "_snapshot", return_value=WITH_HISTORY
                ), mock.patch.object(
                    torch.cuda.memory, "_record_memory_history"
                ) as record:
                    mem_forensics.maybe_dump_memory_forensics("retained-kda:extend")
                self.assertFalse(record.called)
                names = os.listdir(out_dir)
                self.assertEqual(len(names), 1)
                self.assertIn("retained-kda:extend", names[0])
            self.assertIn("retained-kda:extend", mem_forensics._dumped_tags)

    def test_rearm_failure_is_contained_and_keeps_tag_open(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with envs.SGLANG_MEM_FORENSICS_DIR.override(out_dir):
                mem_forensics._started = True
                with mock.patch.object(
                    torch.cuda.memory, "_snapshot", return_value=NO_HISTORY
                ), mock.patch.object(
                    torch.cuda.memory,
                    "_record_memory_history",
                    side_effect=RuntimeError("driver"),
                ), self.assertLogs(
                    LOGGER, level="ERROR"
                ):
                    mem_forensics.maybe_dump_memory_forensics("ready")
            self.assertEqual(os.listdir(out_dir), [])
            self.assertNotIn("ready", mem_forensics._dumped_tags)


if __name__ == "__main__":
    unittest.main()

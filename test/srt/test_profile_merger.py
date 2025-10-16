"""
Unit tests for the ProfileMerger implementation.

Usage:
    python test_profile_merger.py
    python -m unittest test_profile_merger.py -v
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqInput, ProfileReqType
from sglang.srt.utils.profile_merger import ProfileMerger


class TestProfileMerger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_id = "test_profile_123"
        self.merger = ProfileMerger(self.temp_dir, self.profile_id)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rank_extraction_and_labeling(self):
        # Test TP-only
        filename = f"{self.profile_id}-TP-0.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)
        self.assertEqual(rank_info, {"tp_rank": 0})
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[TP00]")

        # Test all parallelism types
        filename = f"{self.profile_id}-TP-1-DP-2-PP-3-EP-4.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)
        self.assertEqual(
            rank_info, {"tp_rank": 1, "dp_rank": 2, "pp_rank": 3, "ep_rank": 4}
        )
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[TP01-DP02-PP03-EP04]")

        # Test partial ranks
        filename = f"{self.profile_id}-TP-0-DP-1.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)
        self.assertEqual(rank_info, {"tp_rank": 0, "dp_rank": 1})
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[TP00-DP01]")

        # Test no ranks
        filename = f"{self.profile_id}.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)
        self.assertEqual(rank_info, {})
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[Unknown]")

    def test_sort_index_calculation(self):
        # Single rank
        rank_info = {"tp_rank": 0}
        sort_idx = self.merger._calculate_sort_index(rank_info, 83)
        self.assertEqual(sort_idx, 83)

        # Multiple ranks
        rank_info = {"tp_rank": 1, "dp_rank": 2, "pp_rank": 3, "ep_rank": 4}
        sort_idx = self.merger._calculate_sort_index(rank_info, 83)
        self.assertNotEqual(sort_idx, 83)
        self.assertGreater(sort_idx, 1000000)

        # Empty ranks
        rank_info = {}
        sort_idx = self.merger._calculate_sort_index(rank_info, 83)
        self.assertEqual(sort_idx, 83)

    def test_rank_sort_key(self):
        # Full ranks: TP-1, DP-2, PP-3, EP-4 → sorted as (DP, EP, PP, TP)
        filename = f"{self.profile_id}-TP-1-DP-2-PP-3-EP-4.trace.json.gz"
        sort_key = self.merger._get_rank_sort_key(filename)
        self.assertEqual(sort_key, (2, 4, 3, 1))

        # Missing ranks: only TP-1 → sorted as (DP=0, EP=0, PP=0, TP=1)
        filename = f"{self.profile_id}-TP-1.trace.json.gz"
        sort_key = self.merger._get_rank_sort_key(filename)
        self.assertEqual(sort_key, (0, 0, 0, 1))

    def test_discover_trace_files(self):
        # Create mock trace files
        trace_files = [
            f"{self.profile_id}-TP-0.trace.json.gz",  # Old format
            f"{self.profile_id}-TP-1.trace.json.gz",  # Old format
            f"{self.profile_id}-TP-0-DP-1.trace.json.gz",  # New format
        ]

        for filename in trace_files:
            filepath = os.path.join(self.temp_dir, filename)
            with gzip.open(filepath, "wt") as f:
                json.dump({"traceEvents": []}, f)

        discovered = self.merger._discover_trace_files()
        self.assertEqual(len(discovered), 3)

        # Check that all expected files are discovered
        discovered_basenames = {os.path.basename(f) for f in discovered}
        expected_basenames = {
            f"{self.profile_id}-TP-0.trace.json.gz",
            f"{self.profile_id}-TP-1.trace.json.gz",
            f"{self.profile_id}-TP-0-DP-1.trace.json.gz",
        }
        self.assertEqual(discovered_basenames, expected_basenames)

        # Test no matches
        empty_merger = ProfileMerger(self.temp_dir, "nonexistent")
        discovered = empty_merger._discover_trace_files()
        self.assertEqual(len(discovered), 0)

    def test_merge_chrome_traces(self):
        # Create multiple trace files in random order
        trace_files = [
            {
                "filename": f"{self.profile_id}-TP-1-DP-1.trace.json.gz",
                "events": [
                    {"ph": "X", "name": "op1", "pid": 83, "ts": 1000.0, "dur": 10.0}
                ],
            },
            {
                "filename": f"{self.profile_id}-TP-0.trace.json.gz",
                "events": [
                    {"ph": "X", "name": "op2", "pid": 84, "ts": 2000.0, "dur": 15.0}
                ],
            },
            {
                "filename": f"{self.profile_id}-TP-0-DP-1.trace.json.gz",
                "events": [
                    {"ph": "X", "name": "op3", "pid": 85, "ts": 3000.0, "dur": 20.0}
                ],
            },
        ]

        for trace_data in trace_files:
            filepath = os.path.join(self.temp_dir, trace_data["filename"])
            trace_content = {
                "schemaVersion": 1,
                "deviceProperties": [{"device_id": 0, "name": "GPU-0"}],
                "traceEvents": trace_data["events"],
            }
            with gzip.open(filepath, "wt") as f:
                json.dump(trace_content, f)

        # Test file ordering by capturing log messages
        import logging

        logger = logging.getLogger("sglang.srt.utils.profile_merger")
        with self.assertLogs(logger, level="INFO") as log_capture:
            merged_path = self.merger.merge_chrome_traces()

        # Verify files were processed in rank order
        log_messages = [
            record.getMessage()
            for record in log_capture.records
            if "Processing file:" in record.getMessage()
        ]
        self.assertIn("TP-0.trace.json.gz", log_messages[0])  # (0,0,0,0) comes first
        self.assertIn(
            "TP-0-DP-1.trace.json.gz", log_messages[1]
        )  # (0,1,0,0) comes second
        self.assertIn(
            "TP-1-DP-1.trace.json.gz", log_messages[2]
        )  # (1,1,0,0) comes last

        # Verify merged content
        self.assertTrue(os.path.exists(merged_path))
        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)

        self.assertEqual(len(merged_data["traceEvents"]), 3)
        self.assertEqual(len(merged_data["deviceProperties"]), 3)

        # Check rank labels in events
        events = merged_data["traceEvents"]
        pids = [event["pid"] for event in events]
        self.assertIn("[TP00] 84", pids)
        self.assertIn("[TP00-DP01] 85", pids)
        self.assertIn("[TP01-DP01] 83", pids)

        # Test merge summary
        summary = self.merger.get_merge_summary()
        self.assertEqual(summary["total_files"], 3)
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["profile_id"], self.profile_id)

        # Test no files error
        empty_merger = ProfileMerger(self.temp_dir, "nonexistent")
        with self.assertRaises(ValueError):
            empty_merger.merge_chrome_traces()


class TestProfileMergerIntegration(unittest.TestCase):

    def test_data_structures_merge_profiles(self):
        # Test ProfileReqInput
        req_input = ProfileReqInput()
        self.assertFalse(req_input.merge_profiles)

        req_input = ProfileReqInput(merge_profiles=True)
        self.assertTrue(req_input.merge_profiles)

        # Test ProfileReq
        req = ProfileReq(type=ProfileReqType.START_PROFILE)
        self.assertFalse(req.merge_profiles)

        req = ProfileReq(type=ProfileReqType.START_PROFILE, merge_profiles=True)
        self.assertTrue(req.merge_profiles)

    def test_integration_parameters(self):
        import inspect

        # Test TokenizerManager
        from sglang.srt.managers.tokenizer_communicator_mixin import (
            TokenizerCommunicatorMixin,
        )

        sig = inspect.signature(TokenizerCommunicatorMixin.start_profile)
        self.assertIn("merge_profiles", sig.parameters)

        # Test SchedulerProfilerMixin
        from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin

        sig = inspect.signature(SchedulerProfilerMixin.init_profile)
        self.assertIn("merge_profiles", sig.parameters)

        # Test CLI profiler
        from sglang.profiler import run_profile

        sig = inspect.signature(run_profile)
        self.assertIn("merge_profiles", sig.parameters)


class TestProfileMergerEdgeCases(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_id = "test_edge_cases"
        self.merger = ProfileMerger(self.temp_dir, self.profile_id)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_error_handling_and_edge_cases(self):
        # Test malformed trace file
        filename = f"{self.profile_id}-TP-0.trace.json.gz"
        filepath = os.path.join(self.temp_dir, filename)
        with gzip.open(filepath, "wt") as f:
            f.write("invalid json content")

        merged_path = self.merger.merge_chrome_traces()
        self.assertTrue(os.path.exists(merged_path))

        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)
        self.assertEqual(len(merged_data["traceEvents"]), 0)

        # Test empty trace file
        with gzip.open(filepath, "wt") as f:
            json.dump({}, f)
        merged_path = self.merger.merge_chrome_traces()
        self.assertTrue(os.path.exists(merged_path))

        # Test missing device properties
        trace_data = {
            "schemaVersion": 1,
            "traceEvents": [
                {"ph": "X", "name": "test", "pid": 83, "ts": 1000.0, "dur": 10.0}
            ],
        }
        with gzip.open(filepath, "wt") as f:
            json.dump(trace_data, f)

        merged_path = self.merger.merge_chrome_traces()
        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)
        self.assertNotIn("deviceProperties", merged_data)

    def test_missing_ranks_and_none_handling(self):
        # Test rank extraction with missing ranks
        filename = f"{self.profile_id}-TP-0.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)
        self.assertEqual(rank_info, {"tp_rank": 0})

        # Test rank label creation with missing ranks
        label = self.merger._create_rank_label({"tp_rank": 0})
        self.assertEqual(label, "[TP00]")

        label = self.merger._create_rank_label({})
        self.assertEqual(label, "[Unknown]")

        # Test sort index calculation
        sort_idx = self.merger._calculate_sort_index({"tp_rank": 0}, 83)
        self.assertGreater(sort_idx, 0)

        sort_idx = self.merger._calculate_sort_index({}, 83)
        self.assertEqual(sort_idx, 83)

        # Test sort key generation
        sort_key = self.merger._get_rank_sort_key(filename)
        self.assertEqual(sort_key, (0, 0, 0, 0))

        # Test _maybe_cast_int with various inputs
        self.assertIsNone(self.merger._maybe_cast_int(None))
        self.assertIsNone(self.merger._maybe_cast_int("invalid"))
        self.assertEqual(self.merger._maybe_cast_int("123"), 123)
        self.assertEqual(self.merger._maybe_cast_int(456), 456)

    def test_mixed_rank_scenarios(self):
        trace_scenarios = [
            {
                "filename": f"{self.profile_id}-TP-0.trace.json.gz",
                "events": [
                    {"ph": "X", "name": "op1", "pid": 83, "ts": 1000.0, "dur": 10.0}
                ],
            },
            {
                "filename": f"{self.profile_id}-TP-1-DP-0.trace.json.gz",
                "events": [
                    {"ph": "X", "name": "op2", "pid": 84, "ts": 2000.0, "dur": 15.0}
                ],
            },
            {
                "filename": f"{self.profile_id}-TP-0-DP-1-PP-0.trace.json.gz",
                "events": [
                    {"ph": "X", "name": "op3", "pid": 85, "ts": 3000.0, "dur": 20.0}
                ],
            },
        ]

        for scenario in trace_scenarios:
            filepath = os.path.join(self.temp_dir, scenario["filename"])
            trace_data = {
                "schemaVersion": 1,
                "deviceProperties": [{"device_id": 0, "name": "GPU-0"}],
                "traceEvents": scenario["events"],
            }
            with gzip.open(filepath, "wt") as f:
                json.dump(trace_data, f)

        merged_path = self.merger.merge_chrome_traces()
        self.assertTrue(os.path.exists(merged_path))

        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)

        self.assertEqual(len(merged_data["traceEvents"]), 3)
        events = merged_data["traceEvents"]
        pids = [event["pid"] for event in events]
        self.assertIn("[TP00] 83", pids)
        self.assertIn("[TP01-DP00] 84", pids)
        self.assertIn("[TP00-DP01-PP00] 85", pids)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for the ProfileMerger implementation.

Usage:
    python test_profile_merger.py
    python -m unittest test_profile_merger.py -v
    python -m unittest test_profile_merger.py::TestProfileMerger::test_rank_extraction_tp_only
"""

import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqInput, ProfileReqType
from sglang.srt.utils.profile_merger import ProfileMerger


class TestProfileMerger(unittest.TestCase):
    """Test cases for ProfileMerger core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profile_id = "test_profile_123"
        self.merger = ProfileMerger(self.temp_dir, self.profile_id)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rank_extraction_tp_only(self):
        """Test rank extraction for TP-only traces."""
        filename = f"{self.profile_id}-TP-0.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)

        expected = {"tp_rank": 0}
        self.assertEqual(rank_info, expected)

    def test_rank_extraction_all_parallelism(self):
        """Test rank extraction for all parallelism types."""
        filename = f"{self.profile_id}-TP-1-DP-2-PP-3-EP-4.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)

        expected = {"tp_rank": 1, "dp_rank": 2, "pp_rank": 3, "ep_rank": 4}
        self.assertEqual(rank_info, expected)

    def test_rank_extraction_partial_ranks(self):
        """Test rank extraction with missing rank types."""
        filename = f"{self.profile_id}-TP-0-DP-1.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)

        expected = {"tp_rank": 0, "dp_rank": 1}
        self.assertEqual(rank_info, expected)

    def test_rank_extraction_no_ranks(self):
        """Test rank extraction with no rank information."""
        filename = f"{self.profile_id}.trace.json.gz"
        rank_info = self.merger._extract_rank_info(filename)

        self.assertEqual(rank_info, {})

    def test_create_rank_label_tp_only(self):
        """Test rank label creation for TP-only."""
        rank_info = {"tp_rank": 0}
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[TP00]")

    def test_create_rank_label_all_ranks(self):
        """Test rank label creation for all parallelism types."""
        rank_info = {"tp_rank": 1, "dp_rank": 2, "pp_rank": 3, "ep_rank": 4}
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[TP01-DP02-PP03-EP04]")

    def test_create_rank_label_partial_ranks(self):
        """Test rank label creation with partial rank information."""
        rank_info = {"tp_rank": 0, "dp_rank": 1}
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[TP00-DP01]")

    def test_create_rank_label_no_ranks(self):
        """Test rank label creation with no rank information."""
        rank_info = {}
        label = self.merger._create_rank_label(rank_info)
        self.assertEqual(label, "[Unknown]")

    def test_calculate_sort_index_single_rank(self):
        """Test sort index calculation for single rank."""
        rank_info = {"tp_rank": 0}
        sort_idx = self.merger._calculate_sort_index(rank_info, 83)
        self.assertEqual(sort_idx, 83)

    def test_calculate_sort_index_multiple_ranks(self):
        """Test sort index calculation for multiple ranks."""
        rank_info = {"tp_rank": 1, "dp_rank": 2, "pp_rank": 3, "ep_rank": 4}
        sort_idx = self.merger._calculate_sort_index(rank_info, 83)
        # Should be different from single rank
        self.assertNotEqual(sort_idx, 83)
        self.assertGreater(sort_idx, 1000000)  # Multi-dimensional should be large

    def test_get_rank_sort_key(self):
        """Test rank sort key generation."""
        filename = f"{self.profile_id}-TP-1-DP-2-PP-3-EP-4.trace.json.gz"
        sort_key = self.merger._get_rank_sort_key(filename)

        expected = (1, 2, 3, 4)
        self.assertEqual(sort_key, expected)

    def test_get_rank_sort_key_missing_ranks(self):
        """Test rank sort key with missing rank information."""
        filename = f"{self.profile_id}-TP-1.trace.json.gz"
        sort_key = self.merger._get_rank_sort_key(filename)

        expected = (1, 0, 0, 0)  # Missing ranks default to 0
        self.assertEqual(sort_key, expected)

    def test_discover_trace_files(self):
        """Test trace file discovery."""
        # Create mock trace files with backward-compatible naming
        trace_files = [
            f"{self.profile_id}-TP-0.trace.json.gz",  # Old format (TP only)
            f"{self.profile_id}-TP-1.trace.json.gz",  # Old format (TP only)
            f"{self.profile_id}-TP-0-DP-1.trace.json.gz",  # New format (TP + DP)
            f"{self.profile_id}-TP-1-DP-0-PP-1.trace.json.gz",  # New format (TP + PP)
        ]

        for filename in trace_files:
            filepath = os.path.join(self.temp_dir, filename)
            with gzip.open(filepath, "wt") as f:
                json.dump({"traceEvents": []}, f)

        discovered = self.merger._discover_trace_files()
        self.assertEqual(len(discovered), 4)

        # Check that files are sorted by rank (TP first, then DP, PP, EP)
        basenames = [os.path.basename(f) for f in discovered]
        expected_order = [
            f"{self.profile_id}-TP-0-DP-1.trace.json.gz",  # TP=0, DP=1, PP=0, EP=0
            f"{self.profile_id}-TP-0.trace.json.gz",  # TP=0, DP=0, PP=0, EP=0
            f"{self.profile_id}-TP-1-DP-0-PP-1.trace.json.gz",  # TP=1, DP=0, PP=1, EP=0
            f"{self.profile_id}-TP-1.trace.json.gz",  # TP=1, DP=0, PP=0, EP=0
        ]
        self.assertEqual(basenames, expected_order)

    def test_discover_trace_files_no_matches(self):
        """Test trace file discovery with no matching files."""
        discovered = self.merger._discover_trace_files()
        self.assertEqual(len(discovered), 0)

    def test_discover_trace_files_backward_compatibility(self):
        """Test that old TP-only format still works."""
        # Create old format files (TP only)
        old_format_files = [
            f"{self.profile_id}-TP-0.trace.json.gz",
            f"{self.profile_id}-TP-1.trace.json.gz",
        ]

        for filename in old_format_files:
            filepath = os.path.join(self.temp_dir, filename)
            with gzip.open(filepath, "wt") as f:
                json.dump({"traceEvents": []}, f)

        discovered = self.merger._discover_trace_files()
        self.assertEqual(len(discovered), 2)

        # Should be able to extract rank info from old format
        for filepath in discovered:
            rank_info = self.merger._extract_rank_info(filepath)
            self.assertIn("tp_rank", rank_info)
            # Other ranks should default to 0
            self.assertEqual(rank_info.get("dp_rank", 0), 0)
            self.assertEqual(rank_info.get("pp_rank", 0), 0)
            self.assertEqual(rank_info.get("ep_rank", 0), 0)

    def test_merge_chrome_traces_single_file(self):
        """Test merging a single trace file."""
        # Create a single trace file
        trace_data = {
            "schemaVersion": 1,
            "deviceProperties": [{"device_id": 0, "name": "GPU-0"}],
            "distributedInfo": {"tp_size": 1},
            "displayTimeUnit": "ms",
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "test_op",
                    "pid": 83,
                    "tid": 83,
                    "ts": 1000.0,
                    "dur": 10.0,
                    "args": {"sort_index": 0},
                }
            ],
        }

        filename = f"{self.profile_id}-TP-0-DP-0-PP-0-EP-0.trace.json.gz"
        filepath = os.path.join(self.temp_dir, filename)
        with gzip.open(filepath, "wt") as f:
            json.dump(trace_data, f)

        merged_path = self.merger.merge_chrome_traces()

        # Verify merged file exists
        self.assertTrue(os.path.exists(merged_path))

        # Verify merged content
        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)

        self.assertEqual(len(merged_data["traceEvents"]), 1)
        self.assertEqual(len(merged_data["deviceProperties"]), 1)

        # Check that pid was modified with rank label
        event = merged_data["traceEvents"][0]
        self.assertEqual(event["pid"], "[TP00-DP00-PP00-EP00] 83")

    def test_merge_chrome_traces_multiple_files(self):
        """Test merging multiple trace files."""
        # Create multiple trace files
        trace_files = [
            {
                "filename": f"{self.profile_id}-TP-0-DP-0-PP-0-EP-0.trace.json.gz",
                "rank_info": {"tp_rank": 0, "dp_rank": 0, "pp_rank": 0, "ep_rank": 0},
                "device_props": [{"device_id": 0, "name": "GPU-0"}],
                "events": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "op1",
                        "pid": 83,
                        "tid": 83,
                        "ts": 1000.0,
                        "dur": 10.0,
                        "args": {"sort_index": 0},
                    }
                ],
            },
            {
                "filename": f"{self.profile_id}-TP-1-DP-0-PP-0-EP-0.trace.json.gz",
                "rank_info": {"tp_rank": 1, "dp_rank": 0, "pp_rank": 0, "ep_rank": 0},
                "device_props": [{"device_id": 1, "name": "GPU-1"}],
                "events": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "op2",
                        "pid": 83,
                        "tid": 83,
                        "ts": 2000.0,
                        "dur": 15.0,
                        "args": {"sort_index": 0},
                    }
                ],
            },
        ]

        for trace_data in trace_files:
            filepath = os.path.join(self.temp_dir, trace_data["filename"])
            trace_content = {
                "schemaVersion": 1,
                "deviceProperties": trace_data["device_props"],
                "distributedInfo": {"tp_size": 2},
                "displayTimeUnit": "ms",
                "traceEvents": trace_data["events"],
            }
            with gzip.open(filepath, "wt") as f:
                json.dump(trace_content, f)

        merged_path = self.merger.merge_chrome_traces()

        # Verify merged file exists
        self.assertTrue(os.path.exists(merged_path))

        # Verify merged content
        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)

        # Should have 2 events and 2 device properties
        self.assertEqual(len(merged_data["traceEvents"]), 2)
        self.assertEqual(len(merged_data["deviceProperties"]), 2)

        # Check that events have correct rank labels
        events = merged_data["traceEvents"]
        pids = [event["pid"] for event in events]
        self.assertIn("[TP00-DP00-PP00-EP00] 83", pids)
        self.assertIn("[TP01-DP00-PP00-EP00] 83", pids)

    def test_merge_chrome_traces_no_files(self):
        """Test merging when no trace files exist."""
        with self.assertRaises(Exception):
            self.merger.merge_chrome_traces()

    def test_get_merge_summary(self):
        """Test merge summary generation."""
        # Create a trace file first
        trace_data = {
            "schemaVersion": 1,
            "deviceProperties": [{"device_id": 0, "name": "GPU-0"}],
            "traceEvents": [
                {"ph": "X", "name": "test", "pid": 83, "ts": 1000.0, "dur": 10.0}
            ],
        }

        filename = f"{self.profile_id}-TP-0-DP-0-PP-0-EP-0.trace.json.gz"
        filepath = os.path.join(self.temp_dir, filename)
        with gzip.open(filepath, "wt") as f:
            json.dump(trace_data, f)

        # Perform merge
        merged_path = self.merger.merge_chrome_traces()

        # Get summary
        summary = self.merger.get_merge_summary()

        self.assertEqual(summary["total_files"], 1)
        self.assertEqual(summary["total_events"], 1)
        self.assertEqual(summary["profile_id"], self.profile_id)
        self.assertEqual(summary["merged_file"], merged_path)


class TestProfileMergerDataStructures(unittest.TestCase):
    """Test cases for data structure modifications."""

    def test_profile_req_input_merge_profiles(self):
        """Test ProfileReqInput with merge_profiles parameter."""
        # Test default value
        req = ProfileReqInput()
        self.assertFalse(req.merge_profiles)

        # Test explicit value
        req = ProfileReqInput(merge_profiles=True)
        self.assertTrue(req.merge_profiles)

        # Test with other parameters
        req = ProfileReqInput(
            output_dir="/tmp/test", profile_by_stage=True, merge_profiles=True
        )
        self.assertTrue(req.merge_profiles)
        self.assertEqual(req.output_dir, "/tmp/test")
        self.assertTrue(req.profile_by_stage)

    def test_profile_req_merge_profiles(self):
        """Test ProfileReq with merge_profiles parameter."""
        # Test default value
        req = ProfileReq(type=ProfileReqType.START_PROFILE)
        self.assertFalse(req.merge_profiles)

        # Test explicit value
        req = ProfileReq(type=ProfileReqType.START_PROFILE, merge_profiles=True)
        self.assertTrue(req.merge_profiles)

        # Test with other parameters
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir="/tmp/test",
            profile_id="test123",
            merge_profiles=True,
        )
        self.assertTrue(req.merge_profiles)
        self.assertEqual(req.output_dir, "/tmp/test")
        self.assertEqual(req.profile_id, "test123")


class TestProfileMergerIntegration(unittest.TestCase):
    """Test cases for integration with SGLang components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_http_api_merge_profiles_parameter(self):
        """Test that HTTP API accepts merge_profiles parameter."""
        # This test verifies the parameter is accepted by the API structure
        # In a real test, this would make an actual HTTP request
        from sglang.srt.managers.io_struct import ProfileReqInput

        # Test that the parameter exists in the data structure
        req_input = ProfileReqInput(merge_profiles=True)
        self.assertTrue(req_input.merge_profiles)

    def test_tokenizer_manager_merge_profiles_parameter(self):
        """Test that TokenizerManager accepts merge_profiles parameter."""
        # This test verifies the parameter signature exists
        # In a real test, this would call the actual method
        # Check that the method signature includes merge_profiles
        import inspect

        from sglang.srt.managers.tokenizer_communicator_mixin import (
            TokenizerCommunicatorMixin,
        )

        sig = inspect.signature(TokenizerCommunicatorMixin.start_profile)
        self.assertIn("merge_profiles", sig.parameters)

    def test_scheduler_profiler_mixin_merge_profiles_parameter(self):
        """Test that SchedulerProfilerMixin accepts merge_profiles parameter."""
        # This test verifies the parameter signature exists
        # Check that the method signature includes merge_profiles
        import inspect

        from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin

        sig = inspect.signature(SchedulerProfilerMixin.init_profile)
        self.assertIn("merge_profiles", sig.parameters)

    def test_command_line_profiler_merge_profiles_parameter(self):
        """Test that command-line profiler accepts merge_profiles parameter."""
        # This test verifies the parameter exists in the profiler module
        # Check that the function signature includes merge_profiles
        import inspect

        from sglang.profiler import run_profile

        sig = inspect.signature(run_profile)
        self.assertIn("merge_profiles", sig.parameters)


class TestProfileMergerEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profile_id = "test_edge_cases"
        self.merger = ProfileMerger(self.temp_dir, self.profile_id)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_handle_malformed_trace_file(self):
        """Test handling of malformed trace files."""
        # Create a malformed trace file
        filename = f"{self.profile_id}-TP-0-DP-0-PP-0-EP-0.trace.json.gz"
        filepath = os.path.join(self.temp_dir, filename)

        # Write invalid JSON
        with gzip.open(filepath, "wt") as f:
            f.write("invalid json content")

        # Should handle gracefully (ProfileMerger logs error and continues)
        # The merge should still complete but with empty results
        merged_path = self.merger.merge_chrome_traces()
        self.assertTrue(os.path.exists(merged_path))

        # Verify merged content is empty due to malformed file
        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)

        # Should have empty traceEvents due to malformed file
        self.assertEqual(len(merged_data["traceEvents"]), 0)

    def test_handle_empty_trace_file(self):
        """Test handling of empty trace files."""
        # Create an empty trace file
        filename = f"{self.profile_id}-TP-0-DP-0-PP-0-EP-0.trace.json.gz"
        filepath = os.path.join(self.temp_dir, filename)

        with gzip.open(filepath, "wt") as f:
            json.dump({}, f)

        # Should handle gracefully
        merged_path = self.merger.merge_chrome_traces()
        self.assertTrue(os.path.exists(merged_path))

    def test_handle_missing_device_properties(self):
        """Test handling of trace files without deviceProperties."""
        # Create trace file without deviceProperties
        trace_data = {
            "schemaVersion": 1,
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "test",
                    "pid": 83,
                    "tid": 83,
                    "ts": 1000.0,
                    "dur": 10.0,
                }
            ],
        }

        filename = f"{self.profile_id}-TP-0-DP-0-PP-0-EP-0.trace.json.gz"
        filepath = os.path.join(self.temp_dir, filename)
        with gzip.open(filepath, "wt") as f:
            json.dump(trace_data, f)

        # Should handle gracefully
        merged_path = self.merger.merge_chrome_traces()
        self.assertTrue(os.path.exists(merged_path))

        # Verify merged content
        with gzip.open(merged_path, "rt") as f:
            merged_data = json.load(f)

        # Should not have deviceProperties if none were provided
        self.assertNotIn("deviceProperties", merged_data)

    def test_handle_duplicate_rank_files(self):
        """Test handling of duplicate rank files."""
        # Create multiple files with same rank
        trace_data = {
            "schemaVersion": 1,
            "deviceProperties": [{"device_id": 0, "name": "GPU-0"}],
            "traceEvents": [
                {"ph": "X", "name": "test", "pid": 83, "ts": 1000.0, "dur": 10.0}
            ],
        }

        # Create two files with same rank
        for i in range(2):
            filename = f"{self.profile_id}-TP-0-DP-0-PP-0-EP-0-{i}.trace.json.gz"
            filepath = os.path.join(self.temp_dir, filename)
            with gzip.open(filepath, "wt") as f:
                json.dump(trace_data, f)

        # Should handle gracefully (both files will be processed)
        merged_path = self.merger.merge_chrome_traces()
        self.assertTrue(os.path.exists(merged_path))


if __name__ == "__main__":
    unittest.main()

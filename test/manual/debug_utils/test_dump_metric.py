"""Unit tests for dump_metric() function."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from sglang.test.test_utils import dump_metric


class TestDumpMetric(unittest.TestCase):
    """Test suite for dump_metric() function."""

    _ENV_KEYS_TO_CLEAN = ["SGLANG_TEST_METRICS_OUTPUT", "PYTEST_CURRENT_TEST"]

    def setUp(self):
        """Clean up env vars before each test."""
        for key in self._ENV_KEYS_TO_CLEAN:
            os.environ.pop(key, None)

    def tearDown(self):
        """Clean up env vars after each test."""
        for key in self._ENV_KEYS_TO_CLEAN:
            os.environ.pop(key, None)

    def test_writes_valid_jsonl(self):
        """Test that dump_metric writes one valid JSON line when env is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "metrics")
            os.environ["SGLANG_TEST_METRICS_OUTPUT"] = base_path

            dump_metric("test_accuracy", 0.95, labels={"model": "llama"})

            # Check file exists with PID suffix
            pid = os.getpid()
            jsonl_path = f"{base_path}.{pid}.jsonl"
            self.assertTrue(os.path.exists(jsonl_path))

            # Read and validate
            with open(jsonl_path, encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])

            # Validate required fields
            self.assertIn("filename", record)
            self.assertIn("test_case", record)
            self.assertEqual(record["metric_name"], "test_accuracy")
            self.assertEqual(record["value"], 0.95)

            # Validate optional fields
            self.assertIn("ts", record)
            self.assertIsInstance(record["ts"], (int, float))
            self.assertEqual(record["labels"], {"model": "llama"})

    def test_no_env_no_file(self):
        """Test that dump_metric doesn't create file when env var not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't set env var
            dump_metric("test_metric", 42)

            # Verify no files created
            files = list(Path(tmpdir).glob("*.jsonl"))
            self.assertEqual(len(files), 0)

    def test_labels_not_serializable_stringified(self):
        """Test that non-serializable labels are stringified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "metrics")
            os.environ["SGLANG_TEST_METRICS_OUTPUT"] = base_path

            # Non-serializable label
            class NonSerializable:
                pass

            dump_metric("test_metric", 100, labels={"obj": NonSerializable()})

            pid = os.getpid()
            jsonl_path = f"{base_path}.{pid}.jsonl"
            with open(jsonl_path, encoding="utf-8") as f:
                lines = f.readlines()

            record = json.loads(lines[0])
            self.assertIn("labels", record)
            self.assertIsInstance(record["labels"], str)

    def test_bool_to_int(self):
        """Test that bool values are converted to int."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "metrics")
            os.environ["SGLANG_TEST_METRICS_OUTPUT"] = base_path

            dump_metric("bool_true", True)
            dump_metric("bool_false", False)

            pid = os.getpid()
            jsonl_path = f"{base_path}.{pid}.jsonl"
            with open(jsonl_path, encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 2)
            record1 = json.loads(lines[0])
            record2 = json.loads(lines[1])

            self.assertEqual(record1["value"], 1)  # True -> 1
            self.assertEqual(record2["value"], 0)  # False -> 0

    def test_pytest_current_test_parsing(self):
        """Test PYTEST_CURRENT_TEST parsing for test_case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "metrics")
            os.environ["SGLANG_TEST_METRICS_OUTPUT"] = base_path
            os.environ["PYTEST_CURRENT_TEST"] = (
                "test/srt/test_example.py::TestClass::test_method (call)"
            )

            dump_metric("pytest_metric", 123)

            pid = os.getpid()
            jsonl_path = f"{base_path}.{pid}.jsonl"
            with open(jsonl_path, encoding="utf-8") as f:
                lines = f.readlines()

            record = json.loads(lines[0])
            # Only assert test_case parsing, not filename
            self.assertEqual(record["test_case"], "TestClass.test_method")


if __name__ == "__main__":
    unittest.main()

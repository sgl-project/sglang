import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils.log_utils import create_log_targets, log_json


class TestLogUtils(unittest.TestCase):
    def test_create_log_targets_stdout(self):
        loggers = create_log_targets(["stdout"], "test_log_utils")
        self.assertEqual(len(loggers), 1)
        self.assertEqual(loggers[0].level, 20)  # INFO level

    def test_create_log_targets_default_stdout(self):
        loggers = create_log_targets(None, "test_log_utils_default")
        self.assertEqual(len(loggers), 1)
        self.assertEqual(loggers[0].level, 20)

    def test_create_log_targets_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets([temp_dir], "test_log_utils_file")
            self.assertEqual(len(loggers), 1)
            self.assertEqual(loggers[0].level, 20)

    def test_create_log_targets_multiple(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets(["stdout", temp_dir], "test_log_utils_multi")
            self.assertEqual(len(loggers), 2)

    def test_log_json_single_logger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets([temp_dir], "test_log_json_single")
            log_json(loggers[0], "test.event", {"key": "value", "number": 42})
            loggers[0].handlers[0].flush()
            log_files = list(Path(temp_dir).glob("*.log"))
            self.assertEqual(len(log_files), 1)
            content = log_files[0].read_text().strip()
            data = json.loads(content)
            self.assertIn("timestamp", data)
            self.assertEqual(data["event"], "test.event")
            self.assertEqual(data["key"], "value")
            self.assertEqual(data["number"], 42)

    def test_log_json_multiple_loggers(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir1,
            tempfile.TemporaryDirectory() as temp_dir2,
        ):
            loggers = create_log_targets(
                [temp_dir1, temp_dir2], "test_log_json_multi"
            )
            log_json(loggers, "test.event", {"key": "value"})
            for lg in loggers:
                lg.handlers[0].flush()

            log_files1 = list(Path(temp_dir1).glob("*.log"))
            log_files2 = list(Path(temp_dir2).glob("*.log"))
            self.assertEqual(len(log_files1), 1)
            self.assertEqual(len(log_files2), 1)

            content1 = log_files1[0].read_text().strip()
            content2 = log_files2[0].read_text().strip()
            self.assertEqual(content1, content2)


if __name__ == "__main__":
    unittest.main()

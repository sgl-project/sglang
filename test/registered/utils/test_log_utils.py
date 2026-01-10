import io
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils.log_utils import create_log_targets, log_json



class TestLogUtils(unittest.TestCase):
    def test_stdout(self):
        for targets in [["stdout"], None]:
            with self.subTest(targets=targets):
                loggers = create_log_targets(targets=targets, name_prefix=f"test_{targets}")
                self.assertEqual(len(loggers), 1)
                data = _log_and_capture_stdout(loggers[0], "test.event", {"key": "value"})
                self.assertIn("timestamp", data)
                self.assertEqual(data["event"], "test.event")
                self.assertEqual(data["key"], "value")

    def test_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets(targets=[temp_dir], name_prefix="test_file")
            self.assertEqual(len(loggers), 1)
            log_json(loggers[0], "file.event", {"data": 123})
            loggers[0].handlers[0].flush()
            data = _read_log_file(temp_dir)
            self.assertIn("timestamp", data)
            self.assertEqual(data["event"], "file.event")
            self.assertEqual(data["data"], 123)

    def test_multiple_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets(
                targets=["stdout", temp_dir], name_prefix="test_multi"
            )
            self.assertEqual(len(loggers), 2)
            stdout_stream = io.StringIO()
            loggers[0].handlers[0].stream = stdout_stream
            log_json(loggers, "multi.event", {"x": 1})
            loggers[1].handlers[0].flush()
            stdout_data = json.loads(stdout_stream.getvalue().strip())
            file_data = _read_log_file(temp_dir)
            self.assertEqual(stdout_data["event"], "multi.event")
            self.assertEqual(file_data["event"], "multi.event")
            self.assertEqual(stdout_data["x"], file_data["x"])


def _read_log_file(temp_dir: str) -> dict:
    log_files = list(Path(temp_dir).glob("*.log"))
    assert len(log_files) == 1
    return json.loads(log_files[0].read_text().strip())


def _log_and_capture_stdout(logger, event: str, data: dict) -> dict:
    stream = io.StringIO()
    logger.handlers[0].stream = stream
    log_json(logger, event, data)
    return json.loads(stream.getvalue().strip())


if __name__ == "__main__":
    unittest.main()

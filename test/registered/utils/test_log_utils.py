import io
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils.log_utils import create_log_targets, log_json


class TestLogUtils(unittest.TestCase):
    def test_create_log_targets_stdout(self):
        loggers = create_log_targets(targets=["stdout"], name_prefix="test_stdout")
        self.assertEqual(len(loggers), 1)
        stream = io.StringIO()
        loggers[0].handlers[0].stream = stream
        log_json(loggers[0], "test.event", {"key": "value"})
        output = stream.getvalue().strip()
        data = json.loads(output)
        self.assertEqual(data["event"], "test.event")
        self.assertEqual(data["key"], "value")

    def test_create_log_targets_default_stdout(self):
        loggers = create_log_targets(targets=None, name_prefix="test_default_stdout")
        self.assertEqual(len(loggers), 1)
        stream = io.StringIO()
        loggers[0].handlers[0].stream = stream
        log_json(loggers[0], "default.event", {"foo": "bar"})
        output = stream.getvalue().strip()
        data = json.loads(output)
        self.assertEqual(data["event"], "default.event")
        self.assertEqual(data["foo"], "bar")

    def test_create_log_targets_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets(targets=[temp_dir], name_prefix="test_file")
            self.assertEqual(len(loggers), 1)
            log_json(loggers[0], "file.event", {"data": 123})
            loggers[0].handlers[0].flush()
            log_files = list(Path(temp_dir).glob("*.log"))
            self.assertEqual(len(log_files), 1)
            data = json.loads(log_files[0].read_text().strip())
            self.assertEqual(data["event"], "file.event")
            self.assertEqual(data["data"], 123)

    def test_create_log_targets_multiple(self):
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
            self.assertEqual(stdout_data["event"], "multi.event")
            log_files = list(Path(temp_dir).glob("*.log"))
            file_data = json.loads(log_files[0].read_text().strip())
            self.assertEqual(file_data["event"], "multi.event")

    def test_log_json_single_logger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets(targets=[temp_dir], name_prefix="test_log_json_single")
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
                targets=[temp_dir1, temp_dir2], name_prefix="test_log_json_multi"
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

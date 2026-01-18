import io
import json
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from sglang.srt.utils.log_utils import create_log_targets, log_json
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="default")


class TestLogUtils(unittest.TestCase):
    def test_stdout(self):
        for targets in [["stdout"], None]:
            with self.subTest(targets=targets):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    loggers = create_log_targets(
                        targets=targets, name_prefix=f"test_stdout_{uuid.uuid4()}"
                    )
                    self.assertEqual(len(loggers), 1)
                    log_json(loggers[0], "test.event", {"key": "value"})
                data = json.loads(buf.getvalue().strip())
                self.assertIn("timestamp", data)
                self.assertEqual(data["event"], "test.event")
                self.assertEqual(data["key"], "value")

    def test_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loggers = create_log_targets(
                targets=[temp_dir], name_prefix=f"test_file_{uuid.uuid4()}"
            )
            self.assertEqual(len(loggers), 1)
            log_json(loggers, "file.event", {"data": 123})
            _flush_all(loggers)
            data = _read_log_file(temp_dir)
            self.assertIn("timestamp", data)
            self.assertEqual(data["event"], "file.event")
            self.assertEqual(data["data"], 123)

    def test_multiple_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.StringIO()
            with redirect_stdout(buf):
                loggers = create_log_targets(
                    targets=["stdout", temp_dir],
                    name_prefix=f"test_multi_{uuid.uuid4()}",
                )
                self.assertEqual(len(loggers), 2)
                log_json(loggers, "multi.event", {"x": 1})
            _flush_all(loggers)
            stdout_data = json.loads(buf.getvalue().strip())
            file_data = _read_log_file(temp_dir)
            self.assertEqual(stdout_data["event"], "multi.event")
            self.assertEqual(file_data["event"], "multi.event")
            self.assertEqual(stdout_data["x"], file_data["x"])


def _flush_all(loggers: list) -> None:
    for logger in loggers:
        for handler in logger.handlers:
            handler.flush()


def _read_log_file(temp_dir: str) -> dict:
    log_files = list(Path(temp_dir).glob("*.log"))
    assert len(log_files) == 1
    return json.loads(log_files[0].read_text().strip())


if __name__ == "__main__":
    unittest.main()

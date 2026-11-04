import argparse
import io
import json
import os
import re
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sglang.srt.arg_groups.fields.observability import _positive_int
from sglang.srt.utils.log_utils import (
    _cleanup_rotated_log_files,
    _start_log_retention_cleanup,
    create_log_targets,
    log_json,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=4, suite="stage-b-test-cpu-intel")

_LOG_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")


class TestLogUtils(CustomTestCase):
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
                data = _parse_log_json(buf.getvalue().strip())
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
            stdout_data = _parse_log_json(buf.getvalue().strip())
            file_data = _read_log_file(temp_dir)
            self.assertEqual(stdout_data["event"], "multi.event")
            self.assertEqual(file_data["event"], "multi.event")
            self.assertEqual(stdout_data["x"], file_data["x"])

    def test_log_retention_deletes_only_expired_rotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "host_0.log")
            now = 10 * 24 * 60 * 60
            expired = Path(f"{filename}.2025-01-01_00")
            recent = Path(f"{filename}.2025-01-09_00")
            current = Path(filename)
            similarly_prefixed = Path(f"{filename}extra.2025-01-01_00")
            unrelated = Path(f"{temp_dir}/other.log.2025-01-01_00")
            for path in (expired, recent, current, similarly_prefixed, unrelated):
                path.write_text("log")
            os.utime(expired, (now - 3 * 24 * 60 * 60,) * 2)
            os.utime(recent, (now - 24 * 60 * 60,) * 2)

            with patch("sglang.srt.utils.log_utils.time.time", return_value=now):
                _cleanup_rotated_log_files(filename, retention_days=2)

            self.assertFalse(expired.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(current.exists())
            self.assertTrue(similarly_prefixed.exists())
            self.assertTrue(unrelated.exists())

    def test_unset_log_retention_keeps_rotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "sglang.srt.utils.log_utils.socket.gethostname", return_value="host"
            ):
                rotated = Path(temp_dir) / "host_0.log.2025-01-01_00"
                rotated.write_text("old")
                _start_log_retention_cleanup(str(Path(temp_dir) / "host_0.log"), None)
                self.assertTrue(rotated.exists())

    def test_log_retention_repeats_without_new_log_records(self):
        class FakeTimer:
            instances = []

            def __init__(self, interval, function):
                self.interval = interval
                self.function = function
                self.daemon = False
                self.started = False
                self.instances.append(self)

            def start(self):
                self.started = True

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = str(Path(temp_dir) / "host_0.log")
            now = 10 * 24 * 60 * 60
            old_from_startup = Path(f"{filename}.2025-01-01_00")
            old_before_periodic = Path(f"{filename}.2025-01-02_00")
            old_from_startup.write_text("old")
            os.utime(old_from_startup, (now - 3 * 24 * 60 * 60,) * 2)

            with (
                patch("sglang.srt.utils.log_utils.time.time", return_value=now),
                patch("sglang.srt.utils.log_utils.threading.Timer", FakeTimer),
            ):
                timer = _start_log_retention_cleanup(filename, retention_days=2)
                duplicate_timer = _start_log_retention_cleanup(
                    filename, retention_days=2
                )
                self.assertFalse(old_from_startup.exists())
                self.assertTrue(timer.daemon)
                self.assertTrue(timer.started)
                self.assertIsNone(duplicate_timer)

                old_before_periodic.write_text("old")
                os.utime(old_before_periodic, (now - 3 * 24 * 60 * 60,) * 2)
                timer.function()

            self.assertFalse(old_before_periodic.exists())
            self.assertEqual(len(FakeTimer.instances), 2)
            self.assertTrue(FakeTimer.instances[1].started)

    def test_log_retention_days_must_be_positive(self):
        self.assertEqual(_positive_int("3"), 3)
        for value in ("0", "-1"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                _positive_int(value)


def _parse_log_json(line: str) -> dict:
    """Strip the ``[YYYY-MM-DD HH:MM:SS] `` prefix added by the formatter."""
    return json.loads(_LOG_PREFIX_RE.sub("", line))


def _flush_all(loggers: list) -> None:
    for logger in loggers:
        for handler in logger.handlers:
            handler.flush()


def _read_log_file(temp_dir: str) -> dict:
    log_files = list(Path(temp_dir).glob("*.log"))
    assert len(log_files) == 1
    return _parse_log_json(log_files[0].read_text().strip())


if __name__ == "__main__":
    unittest.main()

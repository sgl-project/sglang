"""CPU-only contract tests for shared logging targets and JSON events."""

import io
import json
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from sglang.srt.utils import log_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logging.Logger.manager.loggerDict.pop(logger.name, None)


class TestLogTargets(CustomTestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)

    def _track_logger(self, logger: logging.Logger) -> logging.Logger:
        self.addCleanup(_close_logger, logger)
        return logger

    def test_empty_targets_default_to_stdout(self):
        for targets in (None, []):
            with self.subTest(targets=targets):
                stream = io.StringIO()
                prefix = f"{__name__}.{self._testMethodName}.{targets!r}"
                with patch.object(log_utils.sys, "stdout", stream):
                    logger = self._track_logger(
                        log_utils.create_log_targets(
                            targets=targets, name_prefix=prefix
                        )[0]
                    )
                    logger.info("stdout-record")

                self.assertIs(logger.handlers[0].stream, stream)
                self.assertIn("stdout-record", stream.getvalue())

    def test_stdout_target_is_case_insensitive(self):
        stream = io.StringIO()
        with patch.object(log_utils.sys, "stdout", stream):
            logger = self._track_logger(
                log_utils.create_log_targets(
                    targets=["StDoUt"], name_prefix=self._testMethodName
                )[0]
            )
            logger.info("case-insensitive")

        self.assertIn("case-insensitive", stream.getvalue())

    def test_file_target_uses_distributed_rank_in_filename(self):
        directory = Path(self._temp_dir.name)
        with (
            patch.object(log_utils.socket, "gethostname", return_value="worker-a"),
            patch.object(log_utils.dist, "is_initialized", return_value=True),
            patch.object(log_utils.dist, "get_rank", return_value=7),
        ):
            logger = self._track_logger(
                log_utils.create_log_targets(
                    targets=[str(directory)], name_prefix=self._testMethodName
                )[0]
            )
            logger.info("file-record")
            logger.handlers[0].flush()

        log_path = directory / "worker-a_7.log"
        self.assertTrue(log_path.is_file())
        self.assertIn("file-record", log_path.read_text(encoding="utf-8"))

    def test_repeated_target_creation_reuses_one_handler(self):
        stream = io.StringIO()
        prefix = f"{__name__}.{self._testMethodName}"
        with patch.object(log_utils.sys, "stdout", stream):
            first = self._track_logger(
                log_utils.create_log_targets(targets=None, name_prefix=prefix)[0]
            )
            second = log_utils.create_log_targets(targets=None, name_prefix=prefix)[0]
            first.info("one-record")

        self.assertIs(first, second)
        self.assertEqual(len(first.handlers), 1)
        self.assertEqual(stream.getvalue().count("one-record"), 1)


class TestLogJson(CustomTestCase):
    def _capture_logger(self, name: str) -> tuple[logging.Logger, io.StringIO]:
        stream = io.StringIO()
        logger = logging.Logger(name)
        logger.addHandler(logging.StreamHandler(stream))
        self.addCleanup(_close_logger, logger)
        return logger, stream

    def test_log_json_accepts_one_logger(self):
        logger, stream = self._capture_logger("single")

        log_utils.log_json(logger, "scheduler.status", {"rank": 3})

        record = json.loads(stream.getvalue())
        self.assertEqual(record["event"], "scheduler.status")
        self.assertEqual(record["rank"], 3)
        datetime.fromisoformat(record["timestamp"])

    def test_log_json_fans_out_unicode_with_one_timestamp(self):
        first, first_stream = self._capture_logger("first")
        second, second_stream = self._capture_logger("second")
        now = datetime(2026, 8, 16, 12, 34, 56, 123456)

        with patch.object(log_utils, "datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            log_utils.log_json(
                [first, second],
                "request.finished",
                {"message": "你好", "count": 2},
            )

        expected = {
            "timestamp": now.isoformat(),
            "event": "request.finished",
            "message": "你好",
            "count": 2,
        }
        self.assertEqual(json.loads(first_stream.getvalue()), expected)
        self.assertEqual(json.loads(second_stream.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()

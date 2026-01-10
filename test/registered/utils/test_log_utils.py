import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils.log_utils import create_log_target, log_json
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=60, suite="default", nightly=True)


class TestLogUtils(unittest.TestCase):
    def test_create_log_target_stdout(self):
        logger = create_log_target("stdout")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, 20)  # INFO level

    def test_create_log_target_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_log_target(temp_dir)
            self.assertIsNotNone(logger)
            self.assertEqual(logger.level, 20)

    def test_log_json_single_logger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_log_target(temp_dir)
            log_json(logger, "test.event", {"key": "value", "number": 42})
            logger.handlers[0].flush()
            log_files = list(Path(temp_dir).glob("*.log"))
            self.assertEqual(len(log_files), 1)
            content = log_files[0].read_text().strip()
            data = json.loads(content)
            self.assertIn("timestamp", data)
            self.assertEqual(data["event"], "test.event")
            self.assertEqual(data["key"], "value")
            self.assertEqual(data["number"], 42)

    def test_log_json_multiple_loggers(self):
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                logger1 = create_log_target(temp_dir1)
                logger2 = create_log_target(temp_dir2)
                log_json([logger1, logger2], "test.event", {"key": "value"})
                logger1.handlers[0].flush()
                logger2.handlers[0].flush()

                log_files1 = list(Path(temp_dir1).glob("*.log"))
                log_files2 = list(Path(temp_dir2).glob("*.log"))
                self.assertEqual(len(log_files1), 1)
                self.assertEqual(len(log_files2), 1)

                content1 = log_files1[0].read_text().strip()
                content2 = log_files2[0].read_text().strip()
                self.assertEqual(content1, content2)


if __name__ == "__main__":
    unittest.main()

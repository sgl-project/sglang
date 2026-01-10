import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger


class TestSchedulerStatusLogger(unittest.TestCase):
    def test_maybe_dump_respects_interval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SchedulerStatusLogger([temp_dir])
            logger.DUMP_INTERVAL_S = 0.1

            running_rids = ["rid1", "rid2"]
            queued_rids = ["rid3"]

            logger.maybe_dump(running_rids, queued_rids)
            log_files = list(Path(temp_dir).glob("*.log"))
            self.assertEqual(len(log_files), 1)
            content1 = log_files[0].read_text()
            lines1 = [l for l in content1.strip().split("\n") if l]
            self.assertEqual(len(lines1), 1)

            logger.maybe_dump(running_rids, queued_rids)
            content2 = log_files[0].read_text()
            lines2 = [l for l in content2.strip().split("\n") if l]
            self.assertEqual(len(lines2), 1)

            time.sleep(0.15)
            logger.maybe_dump(running_rids, queued_rids)
            logger.loggers[0].handlers[0].flush()
            content3 = log_files[0].read_text()
            lines3 = [l for l in content3.strip().split("\n") if l]
            self.assertEqual(len(lines3), 2)

    def test_dump_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SchedulerStatusLogger([temp_dir])
            running_rids = ["running-1", "running-2"]
            queued_rids = ["queued-1", "queued-2", "queued-3"]

            logger.maybe_dump(running_rids, queued_rids)
            logger.loggers[0].handlers[0].flush()

            log_files = list(Path(temp_dir).glob("*.log"))
            content = log_files[0].read_text().strip()
            data = json.loads(content)

            self.assertIn("timestamp", data)
            self.assertEqual(data["event"], "scheduler.status")
            self.assertEqual(data["running_rids"], running_rids)
            self.assertEqual(data["queued_rids"], queued_rids)

    def test_multiple_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                logger = SchedulerStatusLogger([temp_dir1, temp_dir2])
                logger.maybe_dump(["rid1"], ["rid2"])
                for lg in logger.loggers:
                    lg.handlers[0].flush()

                log_files1 = list(Path(temp_dir1).glob("*.log"))
                log_files2 = list(Path(temp_dir2).glob("*.log"))
                self.assertEqual(len(log_files1), 1)
                self.assertEqual(len(log_files2), 1)

                content1 = log_files1[0].read_text().strip()
                content2 = log_files2[0].read_text().strip()
                self.assertEqual(content1, content2)

    def test_maybe_create_disabled(self):
        with patch.dict("os.environ", {"SGLANG_LOG_SCHEDULER_STATUS_TARGET": ""}):
            from sglang.srt.environ import envs

            envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.clear()
            logger = SchedulerStatusLogger.maybe_create()
            self.assertIsNone(logger)

    def test_maybe_create_stdout(self):
        with patch.dict("os.environ", {"SGLANG_LOG_SCHEDULER_STATUS_TARGET": "stdout"}):
            from sglang.srt.environ import envs

            envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.set("stdout")
            logger = SchedulerStatusLogger.maybe_create()
            self.assertIsNotNone(logger)
            envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.clear()


if __name__ == "__main__":
    unittest.main()

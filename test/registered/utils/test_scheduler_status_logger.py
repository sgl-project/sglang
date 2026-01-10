import json
import tempfile
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)


class TestSchedulerStatusLogger(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name
        cls.process = popen_launch_server(
            "Qwen/Qwen3-0.6B",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--skip-server-warmup"],
            env={
                "SGLANG_LOG_SCHEDULER_STATUS_TARGET": cls.temp_dir,
                "SGLANG_LOG_SCHEDULER_STATUS_INTERVAL_S": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._temp_dir_obj.cleanup()

    def test_scheduler_status_dump(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(2)

        log_files = list(Path(self.temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0, "No log files found in temp directory")

        file_content = "".join(f.read_text() for f in log_files)
        status_found = False
        for line in file_content.splitlines():
            if not line.strip() or not line.startswith("{"):
                continue
            data = json.loads(line)
            if data.get("event") == "scheduler.status":
                self.assertIn("timestamp", data)
                self.assertIn("running_rids", data)
                self.assertIn("queued_rids", data)
                self.assertIsInstance(data["running_rids"], list)
                self.assertIsInstance(data["queued_rids"], list)
                status_found = True
                break

        self.assertTrue(status_found, "scheduler.status event not found in log files")


if __name__ == "__main__":
    unittest.main()

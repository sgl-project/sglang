import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=120, suite="nightly-amd-1-gpu", nightly=True)


class TestSchedulerStatusLogger(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.temp_dir)
        env = os.environ.copy()
        env["SGLANG_LOG_SCHEDULER_STATUS_TARGET"] = cls.temp_dir
        env["SGLANG_LOG_SCHEDULER_STATUS_INTERVAL"] = "1"
        cls.process = popen_launch_server(
            "Qwen/Qwen3-0.6B",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--skip-server-warmup"],
            env=env,
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

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

        events = list(_find_log_events(self.temp_dir, "scheduler.status"))
        print(f"{events=}")
        self.assertGreater(len(events), 0, "scheduler.status event not found")
        data = events[0]
        for field in ["timestamp", "rank", "running_rids", "queued_rids"]:
            self.assertIn(field, data)
        self.assertIsInstance(data["running_rids"], list)
        self.assertIsInstance(data["queued_rids"], list)


def _find_log_events(log_dir: str, event_name: str):
    for f in Path(log_dir).glob("*.log"):
        for line in f.read_text().splitlines():
            if line.startswith("{"):
                data = json.loads(line)
                if data.get("event") == event_name:
                    yield data


if __name__ == "__main__":
    unittest.main()

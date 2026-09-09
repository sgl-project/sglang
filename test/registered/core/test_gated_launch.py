import os
import subprocess
import time
import unittest

import psutil
import requests

from sglang.srt.utils.common import kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(
    est_time=180, stage="nightly", runner_config="1-gpu-large", nightly=True
)

MEM_FRACTION_STATIC = 0.6
GATED_MEMORY_CEILING_MB = 8 * 1024
SERVING_MEMORY_FLOOR_MB = 8 * 1024


class TestGatedLaunch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        _, host, port = cls.base_url.split(":")
        cls.gate_port = get_open_port()
        cls.gate_url = f"http:{host}:{cls.gate_port}"

        command = [
            "sglang",
            "serve",
            "--model-path",
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "--host",
            host[2:],
            "--port",
            port,
            "--gated-launch-port",
            str(cls.gate_port),
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
        ]
        cls.process = subprocess.Popen(command, env=os.environ.copy())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gated_launch_defers_startup_until_activated(self):
        """The engine holds off every sizable allocation until it is activated."""
        self._wait_for_health(self.gate_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

        with self.assertRaises(requests.exceptions.RequestException):
            requests.get(f"{self.base_url}/health", timeout=5)

        gated_memory_mb = self._device_memory_mb()
        self.assertLess(gated_memory_mb, GATED_MEMORY_CEILING_MB)

        for _ in range(2):
            response = requests.post(f"{self.gate_url}/gate/activate", timeout=5)
            self.assertEqual(response.status_code, 200)

        self._wait_for_health(self.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["text"])

        self.assertGreater(self._device_memory_mb(), SERVING_MEMORY_FLOOR_MB)

    def _wait_for_health(self, url: str, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self.assertIsNone(
                self.process.poll(), msg=f"server died while waiting for {url}"
            )
            try:
                if requests.get(f"{url}/health", timeout=5).status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        self.fail(f"{url} did not become healthy within {timeout}s")

    def _device_memory_mb(self) -> int:
        parent = psutil.Process(self.process.pid)
        pids = {parent.pid} | {child.pid for child in parent.children(recursive=True)}

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )

        total_mb = 0
        for line in output.splitlines():
            if not line.strip():
                continue
            pid, used_mb = (field.strip() for field in line.split(","))
            if int(pid) in pids:
                total_mb += int(used_mb)
        return total_mb


if __name__ == "__main__":
    unittest.main()

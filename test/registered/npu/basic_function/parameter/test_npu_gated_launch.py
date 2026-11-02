"""Verify gated startup with Llama-3.2-1B-Instruct on Ascend.

Run from the repository root:
    ASCEND_RT_VISIBLE_DEVICES=0 python \
        test/registered/npu/basic_function/parameter/test_npu_gated_launch.py -v

Optional overrides: SGLANG_TEST_MODEL_PATH, SGLANG_TEST_TP_SIZE (default: 1),
and SGLANG_TEST_LOG_DIR. For TP=2, expose two otherwise idle NPUs. Memory
measurements are device-wide deltas, as in test_npu_memory_consumption.py.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, CustomTestCase

register_npu_ci(est_time=180, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Requires Ascend NPU")
class TestNpuGatedLaunch(CustomTestCase):
    """[Test Category] Parameter; [Test Target] --gated-launch-port."""

    def setUp(self):
        self.tp_size = int(os.environ.get("SGLANG_TEST_TP_SIZE", "1"))
        self.assertIn(self.tp_size, (1, 2))
        self.assertGreaterEqual(torch.npu.device_count(), self.tp_size)
        model = os.environ.get(
            "SGLANG_TEST_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        )
        port = get_open_port()
        gate_port = get_open_port()
        while gate_port == port:
            gate_port = get_open_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.gate_url = f"http://127.0.0.1:{gate_port}"
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

        log_dir = os.environ.get("SGLANG_TEST_LOG_DIR")
        if log_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="sglang-npu-gate-")
            self.addCleanup(temp_dir.cleanup)
            log_dir = temp_dir.name
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / f"gated_launch_tp{self.tp_size}.log"
        log_file = self.log_path.open("w", encoding="utf-8")
        self.addCleanup(log_file.close)

        self.baseline_memory = self._device_memory_mb()
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gated-launch-port",
            str(gate_port),
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--tp-size",
            str(self.tp_size),
            "--mem-fraction-static",
            "0.4",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--disable-cuda-graph",
        ]
        # popen_launch_server waits for serving health, which cannot succeed
        # until this test explicitly opens the gate.
        self.process = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        kill_process_tree(self.process.pid)
        self.process.wait(timeout=30)

    def _device_memory_mb(self):
        memory = []
        for device in range(self.tp_size):
            free, total = torch.npu.mem_get_info(device)
            memory.append((total - free) / (1 << 20))
        return memory

    def _wait_for_health(self, url):
        deadline = time.monotonic() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while time.monotonic() < deadline:
            self.assertIsNone(self.process.poll(), f"Server exited waiting for {url}")
            try:
                if self.session.get(f"{url}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.fail(f"Server did not become healthy: {url}")

    def test_gated_launch_defers_startup_until_activated(self):
        try:
            self._check_gated_launch()
        except Exception:
            print(self.log_path.read_text(encoding="utf-8", errors="replace")[-16000:])
            raise

    def _check_gated_launch(self):
        self._wait_for_health(self.gate_url)
        # Observe multiple gate polling intervals to catch premature startup.
        for _ in range(3):
            time.sleep(1)
            self.assertIsNone(self.process.poll())
            response = self.session.get(f"{self.gate_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "OK")
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=2)
            except requests.RequestException:
                pass
            else:
                self.assertNotEqual(response.status_code, 200)

        log = self.log_path.read_text(encoding="utf-8")
        self.assertEqual(
            log.count("Gated launch waiting for activation."), self.tp_size
        )
        self.assertNotIn("Load weight begin.", log)
        gated_memory = self._device_memory_mb()
        for device, (baseline, gated) in enumerate(
            zip(self.baseline_memory, gated_memory)
        ):
            self.assertLess(gated - baseline, 4096, f"NPU {device}: gated allocation")

        # Retrying activation is idempotent; it must never close the gate again.
        for _ in range(2):
            response = self.session.post(f"{self.gate_url}/gate/activate", timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "OK")
        self._wait_for_health(self.base_url)

        response = self.session.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 16, "temperature": 0},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn("Paris", result["text"])
        self.assertGreater(result["meta_info"]["completion_tokens"], 0)
        log = self.log_path.read_text(encoding="utf-8")
        self.assertEqual(log.count("Gated launch activated."), self.tp_size)
        self.assertEqual(log.count("Load weight begin."), self.tp_size)

        serving_memory = self._device_memory_mb()
        for device, (gated, serving) in enumerate(zip(gated_memory, serving_memory)):
            self.assertGreater(
                serving - gated, 512, f"NPU {device}: activation allocation"
            )
        print(
            f"TP={self.tp_size}, NPU memory MiB: baseline={self.baseline_memory}, "
            f"gated={gated_memory}, serving={serving_memory}; text={result['text']!r}"
        )


if __name__ == "__main__":
    unittest.main()

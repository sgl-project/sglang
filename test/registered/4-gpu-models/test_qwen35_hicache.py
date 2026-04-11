import shutil
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci

# This eval harness applies the chat_template, which is critical for qwen3.5
# to get good accuracy on gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=620, suite="stage-c-test-4-gpu-h100")

QWEN35_27B_MODEL = "Qwen/Qwen3.5-27B"
ACC_THRESHOLDS = {QWEN35_27B_MODEL: {"gsm8k": 0.8}}


class TestQwen35WithHiCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_27B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="qwen35-hicache-")
        env = {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--tp-size",
                "4",
                "--max-mamba-cache-size",
                "500",
                "--max-total-tokens",
                "120000",
                "--chunked-prefill-size",
                "2048",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "128",
                "--mamba-ssm-dtype",
                "bfloat16",
                "--max-running-requests",
                "128",
                "--reasoning-parser",
                "qwen3",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
                "--hicache-mem-layout",
                "page_first_direct",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-size",
                "0",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=100,
            max_tokens=16000,
            num_threads=50,
            repeat=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        return run_eval(args)

    def test_gsm8k(self):
        first_metrics = self._run_gsm8k()
        print(f"first_metrics={first_metrics}")
        self.assertGreaterEqual(
            first_metrics["score"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )

        print(f"flush cache")
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

        second_metrics = self._run_gsm8k()
        print(f"second_metrics={second_metrics}")
        self.assertGreaterEqual(
            second_metrics["score"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )
        self.assertLessEqual(
            abs(second_metrics["score"] - first_metrics["score"]),
            0.05,
            f"HiCache prefetch accuracy drift too large: "
            f"first={first_metrics['score']}, second={second_metrics['score']}",
        )


if __name__ == "__main__":
    unittest.main()

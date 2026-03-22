import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, suite="stage-b-test-large-1-gpu")

QWEN35_9B_MODEL = "Qwen/Qwen3.5-9B"


class TestMambaMemoryLeak(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:30000"
        with (
            envs.SGLANG_REQ_WAITING_TIMEOUT.override(1),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True),
            envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False),
        ):
            cls.process = popen_launch_server(
                QWEN35_9B_MODEL,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--tp-size",
                    "1",
                    "--trust-remote-code",
                    "--kv-cache-dtype",
                    "fp8_e4m3",
                    "--prefill-max-requests",
                    "1",
                    "--mem-fraction-static",
                    "0.7",
                    "--enable-multimodal",
                    "--tool-call-parser",
                    "qwen3_coder",
                    "--reasoning-parser",
                    "qwen3",
                    "--disable-cuda-graph",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_abort(self):
        prefix = "t0 t1 t2 t3 t4 t5 t6 t7\n" * 4096

        requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prefix,
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=300,
        )

        def run(i):
            payload = {
                "text": prefix + f"\nTask {i}",
                "sampling_params": {
                    "temperature": 0.2,
                    "max_new_tokens": 2048,
                    "ignore_eos": True,
                },
            }
            r = requests.post(f"{self.base_url}/generate", json=payload, timeout=600)
            try:
                fr = r.json().get("meta_info", {}).get("finish_reason", {})
                print(i, r.status_code, fr.get("type") if isinstance(fr, dict) else fr)
            except Exception:
                print(i, r.status_code, "non-json")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for request_id in range(20):
                futures.append(executor.submit(run, request_id))
            for future in futures:
                future.result()

        # Let idle self-check run; leak crashes usually happen after requests finish.
        for _ in range(20):
            if self.process.poll() is not None:
                break
            time.sleep(1)

        self.assertIsNone(self.process.poll(), "Server crashed during stress run")


if __name__ == "__main__":
    unittest.main()

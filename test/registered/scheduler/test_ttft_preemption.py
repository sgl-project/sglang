import threading
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")


class TestTTFTPreemption(CustomTestCase):
    """python -m unittest test_ttft_preemption.TestTTFTPreemption"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-0.5B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--ttft-preemption-threshold", "50",
            "--max-prefill-tokens", "2048",
            "--chunked-prefill-size", "2048",
            "--enable-mixed-chunk",
            "--enable-priority-scheduling"
        ]
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ttft_preemption(self):
        # 1. Warm up
        requests.post(f"{self.base_url}/generate", json={"text": "hello", "sampling_params": {"max_new_tokens": 8}})

        large_prompts = []
        for _ in range(4):
            large_prompts.append([233] * 12000)

        small_prompts = []
        for _ in range(4):
            small_prompts.append([200] * 32)

        ttft_results = []

        def send_request(input_ids, track_ttft=False):
            start = time.perf_counter()
            try:
                res = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "input_ids": input_ids,
                        "sampling_params": {"max_new_tokens": 8},
                        "stream": True,
                    },
                    stream=True,
                    timeout=20
                )
                res.raise_for_status()
                for chunk in res.iter_lines():
                    if track_ttft and chunk:
                        ttft = time.perf_counter() - start
                        ttft_results.append(ttft)
                        break
            except Exception as e:
                test_exceptions.append(e)

        test_exceptions = []
        threads = []

        # Fire massive chunk queries to lock the scheduler
        for prompt in large_prompts:
            t = threading.Thread(target=send_request, args=(prompt, False))
            threads.append(t)
            t.start()

        # Give the large queries a moment to firmly establish themselves in the priority queue
        # To avoid arbitrary `time.sleep` race, we wait for the server /health_generate endpoint to show traffic
        for _ in range(50):
            try:
                metrics = requests.get(f"{self.base_url}/get_server_info").json()
                if metrics.get("num_running_requests", 0) > 0 or metrics.get("num_waiting_requests", 0) > 0:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        # Fire tiny queries requiring TTFT
        for prompt in small_prompts:
            t = threading.Thread(target=send_request, args=(prompt, True))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert self.process.poll() is None, "Server crashed during test"
        if test_exceptions:
            raise Exception(f"Requests failed: {test_exceptions}")

        # Ensure all small requests got a response within the timeout
        self.assertTrue(len(ttft_results) == 4, f"Not all small requests returned a TTFT! Results: {ttft_results}")
        max_small_ttft = max(ttft_results)

        print(f"Max small TTFT observed: {max_small_ttft:.2f}s")
        self.assertLess(max_small_ttft, 10.0)

if __name__ == "__main__":
    unittest.main()

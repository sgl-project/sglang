import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)


class TestEnableMixedChunk(CustomTestCase):
    """Testcase: Verify that enabling --enable-mixed-chunk accelerates mixed long and short text requests.

    [Test Category] Parameter Validation
    [Test Target] --enable-mixed-chunk
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Very long prompt (~16k tokens), short prompt (32 tokens)
    LONG_PROMPT = "Hello " * 3000
    SHORT_PROMPT = "The capital of France is"

    @classmethod
    def setUpClass(cls):
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _start_server(self, enable_mixed_chunk: bool):
        other_args = [
            "--attention-backend",
            "ascend",
            "--chunked-prefill-size",
            "4096",
        ]
        if enable_mixed_chunk:
            other_args.append("--enable-mixed-chunk")

        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def _send_single_request(self, prompt, max_new_tokens):
        """Send a single inference request"""
        requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )

    def _benchmark_mixed_load(self):
        """Concurrent benchmark: 50 long prompts + 50 short prompts"""
        threads = []
        # Launch 50 long text requests
        for _ in range(50):
            t = threading.Thread(
                target=self._send_single_request, args=(self.LONG_PROMPT, 32)
            )
            threads.append(t)

        # Launch 50 short text requests
        for _ in range(50):
            t = threading.Thread(
                target=self._send_single_request, args=(self.SHORT_PROMPT, 32)
            )
            threads.append(t)

        # Run concurrently and measure total time
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        end = time.time()
        return end - start

    def test_mixed_chunk_performance(self):
        """Compare total time with mixed-chunk disabled/enabled, verify faster processing when enabled"""
        # Disable mixed-chunk
        self.process = self._start_server(enable_mixed_chunk=False)
        time_off = self._benchmark_mixed_load()
        kill_process_tree(self.process.pid)

        # Enable mixed-chunk
        self.process = self._start_server(enable_mixed_chunk=True)
        time_on = self._benchmark_mixed_load()
        kill_process_tree(self.process.pid)

        # Assert: faster when enabled
        self.assertLess(time_on, time_off)


if __name__ == "__main__":
    unittest.main()

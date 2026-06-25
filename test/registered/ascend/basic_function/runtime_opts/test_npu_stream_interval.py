import json
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


class TestStreamInterval(CustomTestCase):
    """Testcase: Verify --stream-interval controls the stream output chunk size correctly.

    [Test Category] Parameter
    [Test Target] --stream-interval, --stream-output
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    prompt = "The capital of France is"
    total_tokens = 32

    @classmethod
    def setUpClass(cls):
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _start_server(self, interval: int):
        other_args = [
            "--attention-backend",
            "ascend",
            "--stream-output",
            "--stream-interval",
            str(interval),
        ]
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def _run_stream_request(self):
        req = {
            "text": self.prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": self.total_tokens,
            },
            "stream": True,
        }

        chunks = []
        resp = requests.post(self.base_url + "/generate", json=req, stream=True)

        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                body = line[6:]
                if body == b"[DONE]":
                    break
                chunks.append(json.loads(body))
        return chunks

    def test_stream_interval_1_vs_4(self):
        """Test interval=1 produces more chunks than interval=4"""
        # Start server with interval 1
        self.process = self._start_server(interval=1)
        chunks1 = self._run_stream_request()
        kill_process_tree(self.process.pid)

        # Start server with interval 4
        self.process = self._start_server(interval=4)
        chunks4 = self._run_stream_request()
        kill_process_tree(self.process.pid)

        # interval=1 should have significantly more chunks
        self.assertGreater(len(chunks1), len(chunks4))

        # Final output must be the same
        output1 = [t for chunk in chunks1 for t in chunk["output_ids"]]
        output4 = [t for chunk in chunks4 for t in chunk["output_ids"]]
        self.assertEqual(output1, output4)

        # Verify interval works as expected (rough range)
        self.assertAlmostEqual(len(chunks1), self.total_tokens, delta=3)
        self.assertAlmostEqual(len(chunks4), self.total_tokens // 4, delta=2)


if __name__ == "__main__":
    unittest.main()

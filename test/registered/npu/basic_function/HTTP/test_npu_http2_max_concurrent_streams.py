import os
import subprocess
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=50, suite="full-1-npu-a3", nightly=True)


class TestHttp2MaxConcurrentStreams(CustomTestCase):
    """
    Testcase：Verify that --http2-max-concurrent-streams parameter takes effect
              by checking MAX_CONCURRENT_STREAMS == 2 in HTTP/2 SETTINGS frame.

    [Test Category] Parameter
    [Test Target] --http2-max-concurrent-streams
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.port = cls.base_url.split(":")[-1]
        other_args = [
            "--enable-http2",
            "--http2-max-concurrent-streams",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**os.environ, "ASCEND_USE_FIA": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_http2_max_concurrent_streams(self):
        # Verify server is healthy first
        response = requests.get(f"{self.base_url}/health_generate")
        self.assertEqual(response.status_code, 200)

        # Use curl with --http2-prior-knowledge to send HTTP/2 request
        # The -v flag outputs HTTP/2 SETTINGS frame containing MAX_CONCURRENT_STREAMS
        result = subprocess.run(
            [
                "curl",
                "-v",
                "--http2-prior-knowledge",
                f"http://127.0.0.1:{self.port}/get_model_info",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # curl outputs verbose info to stderr, including HTTP/2 SETTINGS frames
        self.assertIn("MAX_CONCURRENT_STREAMS == 2", result.stderr)


if __name__ == "__main__":
    unittest.main()

"""GLM-5.2 INT8 W8A8 with Pipeline Parallelism (PP=2) + NSA attention.

Launches a GLM-5.2 server with PP=2, TP=8, W8A8-INT8 quantization,
NSA attention backend (flashmla_sparse prefill + flashmla_kv decode),
and context parallelism (attn-cp-size=8) to verify that the EPLB-Async
prepare_async_layers() fix correctly skips PPMissingLayer placeholders
under pipeline parallelism — preventing the LazyValue infinite recursion
that previously crashed the server at startup.

A simple generation request validates that the server can serve inference
after successful startup.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "modelscope.cn/organization/T-HEAD/GLM-5.2-W8A8-INT8"
SERVER_LAUNCH_TIMEOUT = 3600

SERVER_ENV = {
    "SGLANG_DSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0",
    "SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM": "1",
}


class TestGLM52PPNsaSparsePrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--pp-size",
                "2",
                "--tp-size",
                "8",
                "--disable-custom-all-reduce",
                "--attention-backend",
                "nsa",
                "--nsa-prefill-backend",
                "flashmla_sparse",
                "--nsa-decode-backend",
                "flashmla_kv",
                "--mem-fraction-static",
                "0.9",
                "--quantization",
                "w8a8_int8",
                "--watchdog-timeout",
                "24000",
                "--dist-timeout",
                "24000",
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                "32",
                "--num-continuous-decode-steps",
                "1",
                "--disable-radix-cache",
                "--trust-remote-code",
                "--disable-shared-experts-fusion",
                "--enable-metrics",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--enforce-disable-flashinfer-allreduce-fusion",
                "--enable-prefill-cp",
                "--cp-strategy",
                "interleave",
                "--attn-cp-size",
                "8",
            ],
            env=SERVER_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generate_simple_request(self):
        """Send a simple generation request to verify server can serve inference."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 256,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        ret = response.json()
        self.assertIn("text", ret)
        output_text = ret["text"]
        print(f"Server response: {output_text}")
        self.assertIn(
            "Paris",
            output_text,
            f"Expected 'Paris' in output, but got: {output_text}",
        )


if __name__ == "__main__":
    unittest.main()

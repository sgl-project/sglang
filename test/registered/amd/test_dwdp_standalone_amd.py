import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_amd_ci(
    est_time=1200,
    suite="stage-c-test-large-8-gpu-amd-mi35x",
)

MODEL_PATH = os.environ.get(
    "SGLANG_DWDP_TEST_MODEL",
    "/models/DeepSeek-R1-0528-MXFP4-th",
)
DWDP_BACKEND = os.environ.get("SGLANG_DWDP_TEST_BACKEND") or "ipc"


class TestDwdpStandaloneAmd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"DWDP test model is unavailable: {MODEL_PATH}")
        os.environ.setdefault("SGLANG_USE_AITER", "1")
        os.environ.setdefault("GPU_ARCHS", "gfx950")
        cls.process = popen_launch_server(
            model=MODEL_PATH,
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--dwdp-size",
                "4",
                "--dwdp-weight-backend",
                DWDP_BACKEND,
                "--trust-remote-code",
                "--attention-backend",
                "aiter",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.80",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_deterministic_completion(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": "The capital of France is",
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=300,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["text"].strip()
        self.assertIn("paris", text.lower(), msg=f"unexpected completion: {text!r}")


if __name__ == "__main__":
    unittest.main()

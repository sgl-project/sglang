"""GLM-5 MXFP4 tests (4-GPU, MI35x)

Tests two MXFP4 quantization formats for GLM-5:
- Quark format (amd/GLM-5-MXFP4): quant_method=quark, per-expert FP4
- Standard format (usableteapot/glm5-mxfp4): quant_method=mxfp4, per-expert FP4

Both use GlmMoeDsaForCausalLM with NSA attention and 256 routed experts.
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x")

SERVER_LAUNCH_TIMEOUT = 3600

GLM5_SERVE_ARGS = [
    "--tp",
    "4",
    "--nsa-prefill-backend",
    "tilelang",
    "--nsa-decode-backend",
    "aiter",
    "--cuda-graph-max-bs",
    "64",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.85",
    "--trust-remote-code",
    "--tool-call-parser",
    "glm47",
    "--reasoning-parser",
    "glm45",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 8}',
]

GLM5_SERVE_ENV = {
    "SGLANG_USE_AITER": "1",
    "SAFETENSORS_FAST_GPU": "1",
}


class _GLM5MXFP4Base(CustomTestCase):
    """Base class for GLM-5 MXFP4 tests."""

    model: str = ""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env.update(GLM5_SERVE_ENV)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=GLM5_SERVE_ARGS,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_server_healthy(self):
        """Verify server started with CUDA graph."""
        response = requests.get(self.base_url + "/health")
        self.assertEqual(response.status_code, 200)

    def test_bs_1_speed(self):
        """Benchmark single-request decode speed."""
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=128)
        _, speed = send_one_prompt(args)

        label = self.model.split("/")[-1]
        print(f"{label} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed ({label})\n" f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(speed, 10)


class TestGLM5QuarkMXFP4(_GLM5MXFP4Base):
    """GLM-5 with Quark MXFP4 quantization (amd/GLM-5-MXFP4)."""

    model = "amd/GLM-5-MXFP4"


class TestGLM5StandardMXFP4(_GLM5MXFP4Base):
    """GLM-5 with standard MXFP4 quantization (usableteapot/glm5-mxfp4)."""

    model = "usableteapot/glm5-mxfp4"


if __name__ == "__main__":
    unittest.main()

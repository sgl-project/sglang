import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-h200")

GLM52_FP8_MODEL = "zai-org/GLM-5.2-FP8"


class TestBCGGlm52Fp8TP8(CustomTestCase):
    """Breakable CUDA graph prefill on GLM-5.2-FP8 (DSA model, TP=8, H200)."""

    @classmethod
    def setUpClass(cls):
        cls.model = GLM52_FP8_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "8",
                "--trust-remote-code",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--mem-fraction-static",
                "0.8",
                "--disable-flashinfer-autotune",
                "--cuda-graph-backend-prefill=breakable",
                # Small chunks => many prefill iterations, each <= the 2048
                # capture max, so every prefill batch replays the BCG graph and
                # exercises the DSA split-op / dual-stream / MLA-fusion paths.
                "--chunked-prefill-size",
                "512",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
            ],
            env={
                "SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=200,
            max_tokens=4096,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()

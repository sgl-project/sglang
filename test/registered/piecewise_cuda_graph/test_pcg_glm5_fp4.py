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

register_cuda_ci(est_time=900, suite="stage-c-test-4-gpu-b200")

GLM5_FP4_MODEL = "nvidia/GLM-5-NVFP4"


class TestPCGGlm5Fp4(CustomTestCase):
    """PCG prefill on GLM-5-NVFP4 (NSA model, TP=4, B200).

    GLM-5 uses GlmMoeDsaForCausalLM (NSA attention). This test verifies that
    piecewise CUDA graph works correctly after the NSA indexer was updated to
    cache k_fp8/k_scale for PCG-compatible prefill.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = GLM5_FP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--trust-remote-code",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--quantization",
                "modelopt_fp4",
                "--enforce-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=200,
            max_tokens=65536,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()

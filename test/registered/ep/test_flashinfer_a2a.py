import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, suite="nightly-4-gpu-b200", nightly=True)

DEEPSEEK_V3_FP4_MODEL = "nvidia/DeepSeek-V3-0324-FP4"
QWEN3_FP8_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
SERVER_LAUNCH_TIMEOUT = 1000


class TestFlashinferA2ATrtllmRoutedFP4(CustomTestCase):
    """flashinfer A2A + flashinfer_trtllm_routed with modelopt_fp4 (DeepSeek V3)."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_FP4_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tp",
                "4",
                "--ep",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "flashinfer",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
                "--quantization",
                "modelopt_fp4",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
            ],
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
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.90)


class TestFlashinferA2ATrtllmRoutedFP8(CustomTestCase):
    """flashinfer A2A + flashinfer_trtllm_routed with fp8 (Qwen3-Next)."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_FP8_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tp",
                "4",
                "--ep",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "flashinfer",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
                "--attention-backend",
                "triton",
                "--mem-fraction-static",
                "0.7",
                "--mamba-ssm-dtype",
                "bfloat16",
            ],
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
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.93)


if __name__ == "__main__":
    unittest.main()

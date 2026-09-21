import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST,
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=359, stage="base-c", runner_config="4-gpu-gb300")

# Keep rendezvous ports below the ephemeral range on the 4-GPU GB300 runner.
NCCL_PORT_BASE = DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 100


class TestDeepseekR1Nvfp4CuteDSLDeepEP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.8",
            "--max-prefill-tokens",
            "16384",
            "--max-running-requests",
            "256",
            "--chunked-prefill-size",
            "1024",
            "--tp",
            "4",
            "--dp",
            "4",
            "--ep",
            "4",
            "--moe-dense-tp-size",
            "1",
            "--enable-dp-attention",
            "--nccl-port",
            str(NCCL_PORT_BASE),
            "--quantization",
            "modelopt_fp4",
            "--attention-backend",
            "trtllm_mla",
            "--moe-runner-backend",
            "flashinfer_cutedsl",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "low_latency",
            "--deepep-dispatcher-output-dtype",
            "bf16",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                **os.environ,
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
                "SGLANG_MOE_NVFP4_DISPATCH": "0",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=512,
            num_threads=512,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()

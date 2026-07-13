"""Test CuteDSL FP4 MoE + FlashInfer alltoall on B200 with DP attention.

Config: Qwen3.5-397B-A17B-NVFP4, B200x4, EP=4 DP=4, cutedsl + flashinfer a2a.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=549, stage="extra-b", runner_config="4-gpu-b200")

MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"

SKIP_TEST = torch.cuda.get_device_capability() < (10, 0)
SKIP_REASON = "Requires Blackwell (B200, sm_100a) or above."


@unittest.skipIf(SKIP_TEST, SKIP_REASON)
class TestCuteDslFlashinferA2A(CustomTestCase):
    """CuteDSL FP4 MoE + FlashInfer one-sided alltoall + DP4 EP4 on B200."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=[
                "--trust-remote-code",
                "--quantization",
                "modelopt_fp4",
                "--tp",
                "4",
                "--ep-size",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--moe-runner-backend",
                "flashinfer_cutedsl",
                "--moe-a2a-backend",
                "flashinfer",
                "--max-prefill-tokens",
                "4096",
                "--disable-radix-cache",
                "--disable-flashinfer-autotune",
                "--watchdog-timeout",
                "900",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            num_examples=1319,
            max_tokens=10240,
            repeat=1,
            num_threads=1319,
            num_shots=8,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.90)


if __name__ == "__main__":
    unittest.main()

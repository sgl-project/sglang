import unittest

import torch

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import (
    CustomTestCase,
    ModelLaunchSettings,
)

register_cuda_ci(est_time=720, suite="stage-c-test-4-gpu-b200")

QWEN35_FP4_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"
ACC_THRESHOLDS = {QWEN35_FP4_MODEL: {"gsm8k": 0.95}}

_is_sm100_cuda13 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 10
    and int(torch.version.cuda.split(".")[0]) >= 13
)


@unittest.skipUnless(_is_sm100_cuda13, "requires SM100+ GPU and CUDA 13+")
class TestQwen35FP4FlashInfer(CustomTestCase):
    def test_gsm8k(self):
        base_args = [
            "--tp-size",
            "4",
            "--chunked-prefill-size",
            "2048",
            "--mamba-scheduler-strategy",
            "extra_buffer",
            "--mamba-track-interval",
            "128",
            "--mamba-ssm-dtype",
            "bfloat16",
            "--max-running-requests",
            "128",
            "--reasoning-parser",
            "qwen3",
            "--attention-backend",
            "trtllm_mha",
            "--quantization",
            "modelopt_fp4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
            "--linear-attn-decode-backend",
            "flashinfer",
            "--linear-attn-prefill-backend",
            "flashinfer",
        ]

        variants = [
            ModelLaunchSettings(
                QWEN35_FP4_MODEL,
                extra_args=base_args,
                variant="FlashInfer",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-397B-A17B-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=ACC_THRESHOLDS[QWEN35_FP4_MODEL]["gsm8k"],
                num_examples=200,
                num_threads=128,
                max_tokens=16000,
                thinking_mode="qwen3",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            ),
        )


if __name__ == "__main__":
    unittest.main()

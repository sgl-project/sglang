import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci

# This eval harness applies the chat_template, which is critical for qwen3.5
# to get good accuracy on gsm8k
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import (
    CustomTestCase,
    ModelLaunchSettings,
)

register_cuda_ci(est_time=563, suite="stage-c-test-4-gpu-b200-small")

QWEN35_FP4_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"
ACC_THRESHOLDS = {QWEN35_FP4_MODEL: {"gsm8k": 0.95}}


class TestQwen35FP4(CustomTestCase):
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
        ]

        variants = [
            ModelLaunchSettings(
                QWEN35_FP4_MODEL,
                extra_args=base_args,
                variant="Triton",
            ),
            # TODO: Fix this and re-enable it
            # ModelLaunchSettings(
            #     QWEN35_FP4_MODEL,
            #     extra_args=base_args + ["--linear-attn-decode-backend", "flashinfer"],
            #     variant="FlashInfer",
            # ),
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

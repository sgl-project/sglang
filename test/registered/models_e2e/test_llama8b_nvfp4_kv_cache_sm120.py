import unittest

from sglang.srt.utils.common import is_sm120_supported
from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-small")

LLAMA8B_NVFP4_MODEL = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
TP_SIZE = 1


@unittest.skipUnless(
    is_sm120_supported(), "requires at least 1 SM120 GPU with CUDA 12.8+"
)
class TestLlama8BNVFP4KVCacheSM120(CustomTestCase):
    """Llama-3.1-8B-Instruct-NVFP4 with NVFP4 KV cache on SM120."""

    def test_gsm8k(self):
        variants = [
            ModelLaunchSettings(
                LLAMA8B_NVFP4_MODEL,
                tp_size=TP_SIZE,
                extra_args=[
                    "--quantization",
                    "modelopt_fp4",
                    "--fp4-gemm-backend",
                    "auto",
                    "--kv-cache-dtype",
                    "nvfp4",
                    "--prefill-attention-backend",
                    "flashinfer",
                    "--decode-attention-backend",
                    "trtllm_mha",
                    "--page-size",
                    "64",
                    "--mem-fraction-static",
                    "0.87",
                    "--cuda-graph-backend-prefill=disabled",
                ],
                variant="NVFP4-GEMM+NVFP4-KV+SM120-XQA",
            )
        ]

        run_combined_tests(
            models=variants,
            test_name="Llama-3.1-8B-Instruct-NVFP4-KV-SM120",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                # Full GSM8K measured locally with 1319 requested / 1314 scored:
                # - FP8 KV: 0.6461187214611872
                # - NVFP4 KV: 0.632420091324201
                # Keep the threshold 0.015 below the NVFP4 KV score.
                baseline_accuracy=0.632420091324201 - 0.015,
                num_examples=1319,
                num_threads=200,
                max_tokens=512,
                api="completion",
            ),
        )


if __name__ == "__main__":
    unittest.main()

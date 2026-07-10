import unittest

import torch

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-small")

LLAMA8B_NVFP4_MODEL = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
TP_SIZE = 1


def _has_sm120_devices(num_devices: int) -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_devices:
        return False
    if torch.version.cuda is None:
        return False
    cuda_version = tuple(map(int, torch.version.cuda.split(".")[:2]))
    if cuda_version < (12, 8):
        return False
    return all(torch.cuda.get_device_capability(i)[0] == 12 for i in range(num_devices))


@unittest.skipUnless(
    _has_sm120_devices(TP_SIZE), "requires at least 1 SM120 GPU with CUDA 12.8+"
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
                # Measured full GSM8K score with this config is ~0.638.
                baseline_accuracy=0.625,
                num_examples=1319,
                num_threads=200,
                max_tokens=512,
                api="completion",
            ),
        )


if __name__ == "__main__":
    unittest.main()

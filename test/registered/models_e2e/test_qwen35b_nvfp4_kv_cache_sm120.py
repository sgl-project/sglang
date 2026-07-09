import unittest

import torch

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=900, stage="extra-a", runner_config="2-gpu-large")

QWEN35B_FP8_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
TP_SIZE = 2


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
    _has_sm120_devices(TP_SIZE), "requires at least 2 SM120 GPUs with CUDA 12.8+"
)
class TestQwen35BNVFP4KVCacheSM120(CustomTestCase):
    """Qwen3.5-35B-A3B-FP8 with NVFP4 KV cache on SM120."""

    def test_gsm8k(self):
        variants = [
            ModelLaunchSettings(
                QWEN35B_FP8_MODEL,
                tp_size=TP_SIZE,
                extra_args=[
                    "--kv-cache-dtype",
                    "nvfp4",
                    "--prefill-attention-backend",
                    "flashinfer",
                    "--decode-attention-backend",
                    "trtllm_mha",
                    "--page-size",
                    "64",
                    "--mamba-scheduler-strategy",
                    "extra_buffer",
                    "--mamba-track-interval",
                    "128",
                    "--mamba-ssm-dtype",
                    "bfloat16",
                    "--max-total-tokens",
                    "10240",
                    "--max-running-requests",
                    "128",
                    "--reasoning-parser",
                    "qwen3",
                    "--linear-attn-decode-backend",
                    "flashinfer",
                    "--linear-attn-prefill-backend",
                    "triton",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true,"num_threads": 64}',
                    "--mem-fraction-static",
                    "0.8",
                    "--disable-radix-cache",
                ],
                variant="TP2+NVFP4-KV+SM120-XQA",
            )
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-35B-A3B-FP8-NVFP4-KV-SM120",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.96,
                num_examples=200,
                num_threads=128,
                max_tokens=10240,
                thinking_mode="qwen3",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            ),
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

# Runs on both Hopper and Blackwell via nightly-8-gpu-common suite
register_cuda_ci(est_time=5400, suite="nightly-8-gpu-common", nightly=True)

NEMOTRON_3_SUPER_BF16_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
NEMOTRON_3_SUPER_NVFP4_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

BASE_ARGS = [
    "--tp=8",
    "--trust-remote-code",
    "--reasoning-parser",
    "nemotron_3",
    "--tool-call-parser",
    "qwen3_coder",
    "--disable-radix-cache",
]

BF16_LOADER_ARGS = [
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 50}',
]

NVFP4_LOADER_ARGS = [
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 17}',
]

MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
    "--max-running-requests=512",
    "--mem-fraction-static=0.75",
]

# Accuracy threshold
GSM8K_BASELINE = 0.935


class TestNvidiaNemotron3SuperNightly(unittest.TestCase):
    """Unified nightly test class for Nemotron 3 Super 120B.

    BF16 variants (Hopper + Blackwell):
    - TP8, TP8+MTP

    NVFP4 variants (Blackwell only):
    - TP8, TP8+MTP

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    def test_nemotron_3_super_bf16(self):
        """Run performance and accuracy for all Nemotron 3 Super BF16 variants."""
        variants = [
            ModelLaunchSettings(
                NEMOTRON_3_SUPER_BF16_MODEL,
                tp_size=8,
                extra_args=BASE_ARGS + BF16_LOADER_ARGS,
                variant="TP8",
            ),
            ModelLaunchSettings(
                NEMOTRON_3_SUPER_BF16_MODEL,
                tp_size=8,
                extra_args=BASE_ARGS + BF16_LOADER_ARGS + MTP_ARGS,
                variant="TP8+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Nemotron-3-Super-120B-BF16",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=GSM8K_BASELINE,
                num_examples=1314,
                num_threads=512,
                max_tokens=16000,
                temperature=1.0,
                top_p=0.95,
                repeat=1,
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_nemotron_3_super_bf16",
            ),
        )

    @unittest.skipIf(not is_blackwell_system(), "NVFP4 requires Blackwell")
    def test_nemotron_3_super_nvfp4(self):
        """Run performance and accuracy for all Nemotron 3 Super NVFP4 variants (Blackwell only)."""
        variants = [
            ModelLaunchSettings(
                NEMOTRON_3_SUPER_NVFP4_MODEL,
                tp_size=8,
                extra_args=BASE_ARGS + NVFP4_LOADER_ARGS,
                variant="TP8",
            ),
            ModelLaunchSettings(
                NEMOTRON_3_SUPER_NVFP4_MODEL,
                tp_size=8,
                extra_args=BASE_ARGS + NVFP4_LOADER_ARGS + MTP_ARGS,
                variant="TP8+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Nemotron-3-Super-120B-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=GSM8K_BASELINE,
                num_examples=1314,
                num_threads=512,
                max_tokens=16000,
                temperature=1.0,
                top_p=0.95,
                repeat=1,
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_nemotron_3_super_nvfp4",
            ),
        )


if __name__ == "__main__":
    unittest.main()

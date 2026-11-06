import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=3600, suite="nightly-8-gpu-b200", nightly=True)

DEEPSEEK_V4_PRO_NVFP4_MODEL_PATH = "nvidia/DeepSeek-V4-Pro-NVFP4"

GSM8K_BASELINE = 0.935

BASE_ARGS = [
    "--trust-remote-code",
    "--tp=8",
    "--chunked-prefill-size=8192",
    "--swa-full-tokens-ratio=0.1",
    "--disable-flashinfer-autotune",
]

DP_ARGS = [
    "--ep=8",
    "--dp=8",
    "--enable-dp-attention",
    "--moe-runner-backend=flashinfer_cutedsl",
    "--moe-a2a-backend=flashinfer",
]

MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
]


class TestDeepseekV4ProNvfp4(unittest.TestCase):

    def test_deepseek_v4_pro_nvfp4_variants(self):
        variants = [
            ModelLaunchSettings(
                DEEPSEEK_V4_PRO_NVFP4_MODEL_PATH,
                extra_args=BASE_ARGS + MTP_ARGS,
                variant="TP8+MTP",
            ),
            ModelLaunchSettings(
                DEEPSEEK_V4_PRO_NVFP4_MODEL_PATH,
                extra_args=BASE_ARGS + DP_ARGS,
                env={"SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "2048"},
                variant="DEP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V4-Pro-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=GSM8K_BASELINE
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 8, 16, 64],
                profile_dir="performance_profiles_deepseek_v4_pro_nvfp4",
            ),
        )


if __name__ == "__main__":
    unittest.main()

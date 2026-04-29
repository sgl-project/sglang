import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

GLM_51_FP8_MODEL_PATH = "zai-org/GLM-5.1-FP8"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
    "--mem-fraction-static=0.9",
    "--enable-metrics",
]

MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
]


class TestGlm51Fp8(unittest.TestCase):
    """GLM-5.1 FP8 on H200/B200 (8-GPU, tp=8)."""

    def test_glm51_fp8(self):
        dp_args = ["--dp=8", "--enable-dp-attention"]

        variants = [
            ModelLaunchSettings(
                GLM_51_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS,
                variant="TP8",
            ),
            ModelLaunchSettings(
                GLM_51_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS + dp_args,
                variant="TP8+DP8",
            ),
            ModelLaunchSettings(
                GLM_51_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS + dp_args + MTP_ARGS,
                variant="TP8+DP8+MTP",
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-5.1-FP8",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.92),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_glm_51_fp8",
            ),
        )


if __name__ == "__main__":
    unittest.main()

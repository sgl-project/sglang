import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(
    est_time=7200,
    suite="nightly-4-gpu-gb300-glm5-nvfp4",
    nightly=True,
    disabled="temporarily disabled while debugging GB300 DSV4 Pro FP4 nightly",
)

MODEL_PATH = "nvidia/GLM-5-NVFP4"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
    "--quantization=modelopt_fp4",
    "--moe-runner-backend=flashinfer_trtllm",
    "--mem-fraction-static=0.9",
    "--enable-metrics",
]

TP_MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
]

DP_MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=1",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=2",
]


class TestGlm5Nvfp4(unittest.TestCase):
    """GLM-5 NVFP4 on GB300 (4x GB300 NVL4, tp=4)."""

    def test_glm5_nvfp4(self):
        variants = [
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS + TP_MTP_ARGS,
                variant="TP4+MTP",
            ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS
                + ["--dp-size=4", "--enable-dp-attention"]
                + DP_MTP_ARGS,
                variant="TP4+DP4+DPA+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-5-NVFP4",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.92),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gb300_utils import GB300_NCCL_PORT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=2280, stage="nightly", runner_config="4-gpu-gb300")

MODEL_PATH = "nvidia/GLM-5.2-NVFP4"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
    "--quantization=modelopt_fp4",
    "--moe-runner-backend=flashinfer_trtllm",
    "--mem-fraction-static=0.9",
    "--enable-metrics",
    "--nccl-port",
    GB300_NCCL_PORT,
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


class TestGlm52Nvfp4(CustomTestCase):
    """GLM-5.2 NVFP4 on GB300 (4x GB300 NVL4, tp=4)."""

    def test_glm52_nvfp4(self):
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
            test_name="GLM-5.2-NVFP4",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.92),
            performance_params=PerformanceTestParams(
                result_dir="performance_results_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()

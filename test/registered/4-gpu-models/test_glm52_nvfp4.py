import os
import unittest
from unittest.mock import patch

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=1800, suite="nightly-4-gpu-b200", nightly=True)

GLM_52_NVFP4_MODEL_PATH = os.getenv(
    "SGLANG_TEST_GLM52_NVFP4_MODEL_PATH", "nvidia/GLM-5.2-NVFP4"
)

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
    "--quantization=modelopt_fp4",
    "--chunked-prefill-size=4096",
    "--mem-fraction-static=0.78",
    "--cuda-graph-max-bs-decode=16",
    "--disable-flashinfer-autotune",
    "--enable-metrics",
]

DEP4_ARGS = [
    "--dp=4",
    "--ep=4",
    "--enable-dp-attention",
]

FLASHINFER_A2A_ENV = {
    "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "4096",
}


class TestGlm52Nvfp4(CustomTestCase):
    """GLM-5.2 NVFP4 DEP4 configurations on four B200 GPUs."""

    def test_glm52_nvfp4(self):
        variants = [
            ModelLaunchSettings(
                GLM_52_NVFP4_MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS
                + DEP4_ARGS
                + [
                    "--moe-runner-backend=flashinfer_cutedsl",
                    "--moe-a2a-backend=flashinfer",
                    "--ep-dispatch-algorithm=static",
                ],
                env=FLASHINFER_A2A_ENV,
                variant="DEP4+FlashInferA2A+StaticEP",
            ),
            ModelLaunchSettings(
                GLM_52_NVFP4_MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS + DEP4_ARGS,
                variant="DEP4-default",
            ),
        ]

        # The benchmark subprocess also validates the server arguments, so the
        # FlashInfer token budget must be inherited by both client and server.
        with patch.dict(os.environ, FLASHINFER_A2A_ENV, clear=False):
            run_combined_tests(
                models=variants,
                test_name="GLM-5.2-NVFP4",
                accuracy_params=AccuracyTestParams(
                    dataset="gsm8k", baseline_accuracy=0.92
                ),
                performance_params=PerformanceTestParams(
                    profile_dir="performance_profiles_glm_52_nvfp4",
                ),
            )


if __name__ == "__main__":
    unittest.main()

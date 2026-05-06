"""MI35x GLM-4.7-FP8 GSM8K Accuracy Evaluation Test (8-GPU)

Tests GLM-4.7-FP8 accuracy using GSM8K benchmark on MI35x.

Registry: nightly-amd-8-gpu-mi35x-glm47-fp8 suite
"""

import os

# Set HF cache for MI35x
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Register for AMD CI - MI35x GLM-4.7-FP8 accuracy test (~30 min)
register_amd_ci(
    est_time=1800,
    suite="nightly-amd-8-gpu-mi35x-glm47-fp8",
    nightly=True,
)

GLM_4_7_FP8_MODEL_PATH = "zai-org/GLM-4.7-FP8"


class TestGLM47FP8EvalMI35x(unittest.TestCase):
    """GLM-4.7-FP8 GSM8K Accuracy Evaluation Test for MI35x."""

    def test_glm_47_fp8(self):
        """Run accuracy test for GLM-4.7-FP8."""
        base_args = [
            "--trust-remote-code",
            "--tool-call-parser=glm47",
            "--reasoning-parser=glm45",
        ]

        variants = [
            ModelLaunchSettings(
                GLM_4_7_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-4.7-FP8",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.92),
        )


if __name__ == "__main__":
    unittest.main()

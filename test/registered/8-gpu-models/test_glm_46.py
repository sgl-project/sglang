"""Unified GLM-4.6 performance and accuracy tests using nightly_metrics.

This file replaces test_glm_4_6_perf.py and adds accuracy testing.
Simple configuration: TP=8 only.
GLM-4.6 is a 357B MoE model.
"""

import sys
import unittest
from pathlib import Path

# Add nightly directory to path for run_combined_tests import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nightly"))

from run_combined_tests import run_metrics

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Registered to nightly-8-gpu-h200-basic suite
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-h200-basic", nightly=True)
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-b200-basic", nightly=True)

GLM_4_6_MODEL_PATH = "zai-org/GLM-4.6"


class TestGLM46Unified(unittest.TestCase):
    """Unified test class for GLM-4.6 performance and accuracy.

    Single variant with simple TP=8 configuration.
    GLM-4.6 is a 357B MoE model.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_glm_46(self):
        """Run performance and accuracy for GLM-4.6."""
        print("\n" + "=" * 80)
        print("RUNNING: TestGLM46Unified.test_glm_46")
        print("=" * 80)

        variants = [
            ModelLaunchSettings(
                GLM_4_6_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp",
                    "8",
                ],
            ),
        ]

        # Run both performance and accuracy
        # run_metrics() handles summary printing and raises AssertionError on failure
        run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_glm_4_6",
            test_name="GLM-4.6 Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )


if __name__ == "__main__":
    unittest.main()

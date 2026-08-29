import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

register_cuda_ci(est_time=1800, stage="nightly", runner_config="4-gpu-b200")

LAGUNA_XS_NVFP4_MODEL = "poolside/Laguna-XS-2.1-NVFP4"
LAGUNA_S_NVFP4_MODEL = "poolside/Laguna-S-2.1-NVFP4"

# Measured 0.935 (temp=1.0, 200 examples); floored with margin for
# FP4-kernel variance across Blackwell parts.
LAGUNA_XS_GSM8K_BASELINE = 0.87
# Measured 0.95 (temp=1.0, 200 examples); floored with margin.
LAGUNA_S_GSM8K_BASELINE = 0.89


class TestLagunaNVFP4Nightly(unittest.TestCase):
    """Nightly test for Laguna-XS-2.1 / S-2.1 NVFP4, TP=1, Blackwell only.

    Each model runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)

    The models carry different gsm8k baselines, so each gets its own
    run_combined_tests call (the runner takes one baseline per call).
    """

    def _run_model(self, model_path: str, test_name: str, baseline: float) -> None:
        run_combined_tests(
            models=[
                ModelLaunchSettings(
                    model_path,
                    tp_size=1,
                    variant="TP1",
                )
            ],
            test_name=test_name,
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=baseline,
                num_examples=200,
                num_shots=5,
                num_threads=128,
                max_tokens=4096,
                temperature=1.0,
                top_p=0.95,
                repeat=1,
            ),
            performance_params=PerformanceTestParams(
                result_dir="performance_results_laguna_nvfp4",
            ),
        )

    @unittest.skipIf(not is_blackwell_system(), "NVFP4 requires Blackwell")
    def test_laguna_xs_nvfp4(self):
        """Run performance and accuracy for Laguna-XS-2.1-NVFP4 (TP1)."""
        self._run_model(
            model_path=LAGUNA_XS_NVFP4_MODEL,
            test_name="Laguna-XS-2.1-NVFP4",
            baseline=LAGUNA_XS_GSM8K_BASELINE,
        )

    @unittest.skipIf(not is_blackwell_system(), "NVFP4 requires Blackwell")
    def test_laguna_s_nvfp4(self):
        """Run performance and accuracy for Laguna-S-2.1-NVFP4 (TP1)."""
        self._run_model(
            model_path=LAGUNA_S_NVFP4_MODEL,
            test_name="Laguna-S-2.1-NVFP4",
            baseline=LAGUNA_S_GSM8K_BASELINE,
        )


if __name__ == "__main__":
    unittest.main()

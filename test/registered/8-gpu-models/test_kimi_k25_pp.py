import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

KIMI_K25_MODEL_PATH = "moonshotai/Kimi-K2.5"


class TestKimiK25PP(unittest.TestCase):
    """Test Kimi-K2.5 with Pipeline Parallelism (TP4 PP2) on 8 GPUs."""

    def test_kimi_k25_pp(self):
        base_args = [
            "--tp-size=4",
            "--pp-size=2",
            "--chunked-prefill-size=8192",
            "--tool-call-parser=kimi_k2",
            "--reasoning-parser=kimi_k2",
        ]

        variants = [
            ModelLaunchSettings(
                KIMI_K25_MODEL_PATH,
                tp_size=4,
                extra_args=base_args,
                variant="TP4_PP2",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.5-PP",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.90),
        )


if __name__ == "__main__":
    unittest.main()

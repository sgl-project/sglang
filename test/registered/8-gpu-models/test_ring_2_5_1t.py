import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# register_cuda_ci(est_time=1000, suite="nightly-8-gpu-common", nightly=True)
register_cuda_ci(est_time=1000, suite="stage-c-test-8-gpu-h200")

RING_2_5_1T_MODEL_PATH = "inclusionAI/Ring-2.5-1T"


class TestRing2_5_1T(unittest.TestCase):
    """Accuracy test for Ring-2.5-1T.

    Ring-2.5-1T is a ~1T MoE model with linear attention layers.
    Uses TP=8 for GSM8K evaluation.
    """

    def test_ring_2_5_1t(self):
        base_args = [
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]

        variants = [
            ModelLaunchSettings(
                RING_2_5_1T_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Ring-2.5-1T",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                num_examples=200,
                baseline_accuracy=0.88,
                temperature=1.2,
                top_p=0.8,
                max_tokens=4096,
            ),
        )


if __name__ == "__main__":
    unittest.main()

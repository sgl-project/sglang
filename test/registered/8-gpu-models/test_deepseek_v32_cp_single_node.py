import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

register_cuda_ci(est_time=5400, suite="nightly-8-gpu-common", nightly=True)

DEEPSEEK_V32_EXP_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"

BASE_ARGS = [
    "--trust-remote-code",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 64}',
]

DP_ARGS = [
    "--tp=8",
    "--dp=2",
    "--enable-dp-attention",
]

MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
    "--mem-frac=0.7",
    "--cuda-graph-max-bs=32",
    "--max-running-requests=32",
]

# Accuracy thresholds
GSM8K_BASELINE = 0.935

# CP mode arguments
CP_IN_SEQ_SPLIT_ARGS = [
    "--enable-nsa-prefill-context-parallel",
    "--nsa-prefill-cp-mode=in-seq-split",
]

CP_ROUND_ROBIN_ARGS = [
    "--enable-nsa-prefill-context-parallel",
    "--nsa-prefill-cp-mode=round-robin-split",
]


class TestDeepseekV32CPSingleNode(unittest.TestCase):
    """Test class for DeepSeek V3.2 with NSA context parallelism.

    Tests context parallelism modes with DP+MTP:
    - in-seq-split: In-sequence split CP mode
    - round-robin-split: Round-robin split CP mode
    """

    @unittest.skipIf(is_blackwell_system(), "Skip on B200 systems")
    def test_deepseek_v32_cp_variants(self):
        """Run accuracy tests for DeepSeek V3.2 CP variants."""
        variants = [
            # Variant: in-seq-split CP mode with DP+MTP
            ModelLaunchSettings(
                DEEPSEEK_V32_EXP_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + MTP_ARGS + CP_IN_SEQ_SPLIT_ARGS,
                variant="CP-in-seq-split",
            ),
            # Variant: round-robin-split CP mode (TP only, no DP)
            ModelLaunchSettings(
                DEEPSEEK_V32_EXP_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + ["--tp=8"] + MTP_ARGS + CP_ROUND_ROBIN_ARGS,
                variant="CP-round-robin-split",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2-Exp CP Single Node",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=GSM8K_BASELINE
            ),
            performance_params=None,
        )


if __name__ == "__main__":
    unittest.main()

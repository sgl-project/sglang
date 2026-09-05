import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via the nightly-8-gpu-common suite.
register_cuda_ci(est_time=900, suite="nightly-8-gpu-common", nightly=True)

LONGCAT_FLASH_LITE_SPARSE_MODEL_PATH = "meituan-longcat/LongCat-Flash-Lite-Sparse"

COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static=0.85",
    "--chunked-prefill-size=2048",
    "--nsa-prefill-backend=fa3",
    "--kv-cache-dtype=bfloat16",
]


class TestLongCatFlashLiteSparse(unittest.TestCase):
    """LongCat-Flash-Lite-Sparse on H200/B200 (8-GPU), plain TP8.

    Guards the zero-expert contribution / TP all-reduce ordering fix
    (LongcatFlashMoE.forward): if the zero-expert term were (re-)added
    before the all-reduce, it would be summed across all 8 ranks and
    inflate final_hidden_states by ~8x, collapsing gsm8k accuracy far
    below the floor below.
    """

    def test_longcat_flash_lite_sparse(self):
        variants = [
            ModelLaunchSettings(
                LONGCAT_FLASH_LITE_SPARSE_MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS,
                variant="TP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="LongCat-Flash-Lite-Sparse",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                # Measured 2026-08-06 on 8xH20-140GB (TP8):
                #   zero-expert added before all-reduce (buggy):  0.674
                #   zero-expert added after all-reduce (fixed): 0.849
                baseline_accuracy=0.8,
                num_threads=64,
            ),
        )


if __name__ == "__main__":
    unittest.main()

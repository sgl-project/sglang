import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via the nightly-8-gpu-common suite.
register_cuda_ci(est_time=1200, suite="nightly-8-gpu-common", nightly=True)

# LongCat-Flash-Lite-FP8 is the smallest member of the LongCat family
# (~138 GB FP8 weights, hidden=3072, 14 layers, 256 routed + 128 zero
# experts, moe_topk=12). It fits comfortably on a single 8-GPU node
# (~17 GB/GPU of weights at tp8), so it is the CI regression proxy for the
# much larger LongCat-2.0-FP8 (2 TB, needs >=16 GPUs and cannot run on any
# current single-node CUDA runner).
#
# It shares the SAME sglang model file (models/longcat_flash.py, arch name
# LongcatFlashNgramForCausalLM is remapped to model_type "longcat_flash")
# with LongCat-Flash-Chat-FP8 and LongCat-2.0-FP8, so it guards the shared
# code paths that recent LongCat EP fixes touched:
#   - the scheduler moe-topk gate for --moe-a2a-backend (PR #30975)
#   - ScMoE dense-branch gather (RoPE) + the MoE-vs-DeepEPMoE double
#     all_reduce fix (PR #31311)
#   - the zero-expert (identity) compute path (zero_expert_num=128) and
#     NgramEmbedding (ngram_vocab_size_ratio) — 2.0-specific features that
#     Flash-Chat-FP8 does NOT have, but Lite DOES.
# It does NOT cover DSA (LongCat Sparse Attention) or MTP/NextN, which are
# unique to LongCat-2.0 and require a >=16-GPU e2e run.
LONGCAT_FLASH_LITE_FP8_MODEL_PATH = "meituan-longcat/LongCat-Flash-Lite-FP8"

COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static=0.85",
]


class TestLongCatFlashLiteFp8(unittest.TestCase):
    """LongCat-Flash-Lite-FP8 on H200/B200 (8-GPU).

    Two variants exercise the two MoE all-to-all backends that the LongCat
    EP fixes gate on:
      - EP8 + deepep : real expert parallelism (the path #30975/#31311 fix)
      - EP8 + none   : EP-over-TP baseline (all_reduce / gather correctness)
    """

    def test_longcat_flash_lite_fp8(self):
        variants = [
            ModelLaunchSettings(
                LONGCAT_FLASH_LITE_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS + ["--ep=8"],
                variant="TP8+EP8+none",
            ),
            ModelLaunchSettings(
                LONGCAT_FLASH_LITE_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS + ["--ep=8", "--moe-a2a-backend=deepep"],
                variant="TP8+EP8+deepep",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="LongCat-Flash-Lite-FP8",
            # Measured 2026-07-22 on 8xH100-80GB, gsm8k 200q, 5-shot, greedy:
            #   TP8+EP8+none   -> 0.840
            #   TP8+EP8+deepep -> 0.820
            # Floor set to 0.78 to absorb 200-sample noise.
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.78,
                num_examples=200,
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_longcat_flash_lite_fp8",
            ),
        )


if __name__ == "__main__":
    unittest.main()

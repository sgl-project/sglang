"""Unified Mistral-Large-3 performance and accuracy tests using nightly_metrics.

This file replaces test_mistral_large3_perf.py which already had both perf + accuracy.
Two variants: basic (TP=8 + trtllm_mla) and eagle (basic + EAGLE speculative decoding).
Requires SGLANG_ENABLE_JIT_DEEPGEMM=0 environment variable.
"""

import os
import sys
import unittest
from pathlib import Path

# Add nightly directory to path for run_combined_tests import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nightly"))

from run_combined_tests import run_metrics

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Registered to nightly-8-gpu-b200-basic suite (B200 only - requires trtllm_mla backend)
# This suite should be run with --timeout-per-file=12000 (200 minutes)
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-b200-basic", nightly=True)

MISTRAL_LARGE3_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"
MISTRAL_LARGE3_EAGLE_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle"


class TestMistralLarge3Unified(unittest.TestCase):
    """Unified test class for Mistral-Large-3 performance and accuracy.

    Two variants:
    - basic: TP=8 + trtllm_mla backend
    - eagle: basic + EAGLE speculative decoding with draft model

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    @classmethod
    def setUpClass(cls):
        # Set environment variable to disable JIT DeepGemm
        os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

    @classmethod
    def tearDownClass(cls):
        # Clean up environment variable
        if "SGLANG_ENABLE_JIT_DEEPGEMM" in os.environ:
            del os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"]

    def test_mistral_large3_all_variants(self):
        """Run performance and accuracy for all Mistral-Large-3 variants."""
        print("\n" + "=" * 80)
        print("RUNNING: TestMistralLarge3Unified.test_mistral_large3_all_variants")
        print("=" * 80)

        variants = [
            # Variant: "basic" (from test_mistral_large3_perf.py)
            # TP=8 + trtllm_mla backend
            ModelLaunchSettings(
                MISTRAL_LARGE3_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--tp",
                    "8",
                    "--attention-backend",
                    "trtllm_mla",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                    "--chat-template",
                    "mistral",
                ],
            ),
            # Variant: "eagle" (from test_mistral_large3_perf.py)
            # TP=8 + trtllm_mla + EAGLE speculative decoding with draft model
            ModelLaunchSettings(
                MISTRAL_LARGE3_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--tp",
                    "8",
                    "--attention-backend",
                    "trtllm_mla",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    MISTRAL_LARGE3_EAGLE_MODEL_PATH,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--kv-cache-dtype",
                    "auto",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                    "--chat-template",
                    "mistral",
                ],
            ),
        ]

        # Run both performance and accuracy for all variants
        # run_metrics() handles summary printing and raises AssertionError on failure
        run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_mistral_large3",
            test_name="Mistral-Large-3 Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )


if __name__ == "__main__":
    unittest.main()

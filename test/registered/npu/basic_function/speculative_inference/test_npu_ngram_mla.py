"""Run the NGRAM corpus-option matrix on a real dense MLA target model."""

import os
import unittest
from unittest.mock import patch

import test_npu_ngram as ngram_checks

from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=800, suite="nightly-1-npu-a3", nightly=True)


class TestNpuMlaNgram(ngram_checks.TestNpuNgram):
    # Keep all 48 tokens inside a specified continuation. After the "first ten
    # integers" task ends, BF16 kernel shapes can flip near-tied free-form choices.
    prompts = ngram_checks.PROMPTS[:2] + [
        "Continue the sequence: 101, 102, 103, 104, 105,"
    ]

    @unittest.skip("NGRAM target-verification graphs are currently MHA/GQA-only.")
    def test_graph_matches_eager_with_long_prefix(self):
        pass

    def test_eight_options_match_greedy_baseline(self):
        # Compare the same FIA attention family on both sides. The legacy
        # paged MLA decoder has different BF16 rounding from target verification.
        with patch.dict(
            os.environ,
            {
                "ASCEND_USE_FIA": "true",
                "SGLANG_TEST_NGRAM_MODEL": os.environ.get(
                    "SGLANG_TEST_NGRAM_MLA_MODEL", DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
                ),
            },
        ):
            super().test_eight_options_match_greedy_baseline()


if __name__ == "__main__":
    unittest.main()

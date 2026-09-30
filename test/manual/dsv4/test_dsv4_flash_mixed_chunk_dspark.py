"""B200 x DeepSeek-V4-Flash: --enable-mixed-chunk with DSPARK speculative decoding.

With SGLANG_DSPARK_VERIFY_MERGED_MIXED=1, running requests verify inside prefill
steps (verify-merged mixed step). Runs the shared GSM8K sanity and AIME25 checks
with the mixed step enabled, then asserts that verify-merged mixed steps ran.
"""

import os
import sys
import unittest

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DSV4FlashAime25TestBase

MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class TestB200FlashMixedChunkDspark(DSV4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "2",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--speculative-algorithm",
        "DSPARK",
        "--chunked-prefill-size",
        "4096",
        "--enable-mixed-chunk",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--enable-metrics",
    ]
    EXTRA_ENV = {
        "SGLANG_ENABLE_PREFILL_WAR_READ_DONE": "1",
        "SGLANG_DSPARK_VERIFY_MERGED_MIXED": "1",
    }

    def test_zz_merged_step_observed(self):
        """Runs last: the evals above sent concurrent requests, so some prefill
        steps must have carried running (verifying) requests."""
        metrics = requests.get(f"{self.base_url}/metrics", timeout=30).text
        self.assertRegex(
            metrics,
            r"sglang:mixed_verify_steps_total(\{[^}]*\})? [1-9]",
            "no verify-merged mixed step ran",
        )


if __name__ == "__main__":
    unittest.main()

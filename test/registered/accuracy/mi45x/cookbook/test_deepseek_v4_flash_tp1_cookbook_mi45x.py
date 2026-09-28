"""MI455x cookbook entry 3: DeepSeek-V4-Flash GSM8K at TP=1.

Reproduces the cookbook's measured 0.932. The TP=4 half is a separate file and
suite: they share a 149 GB checkpoint, so keeping them apart avoids paying for
a second cold load when only one is under investigation, and lets this half run
on a single-card runner.

Registry: nightly-amd-1-gpu-mi45x-cookbook-dsv4-flash suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    DEEPSEEK_V4_FLASH_TP1,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 124.4s of eval, on top of a cold load of a 149 GB checkpoint.
register_amd_ci(
    est_time=1800,
    suite="nightly-amd-1-gpu-mi45x-cookbook-dsv4-flash",
    nightly=True,
)


class TestDeepSeekV4FlashTp1CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, DEEPSEEK_V4_FLASH_TP1)


if __name__ == "__main__":
    unittest.main()

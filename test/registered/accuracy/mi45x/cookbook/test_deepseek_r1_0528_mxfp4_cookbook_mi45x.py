"""MI455x cookbook entry 2: DeepSeek-R1-0528-MXFP4 GSM8K (1-GPU).

Reproduces the cookbook's measured 0.948 on MI455 A0. This entry carries the
longest env list in the cookbook -- fifteen AITER/SGLANG toggles -- and they are
transcribed verbatim in DEEPSEEK_R1_0528_MXFP4; none of them has been
independently re-derived, so treat the set as a unit.

Registry: nightly-amd-1-gpu-mi45x-cookbook-dsr1 suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    DEEPSEEK_R1_0528_MXFP4,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 471.6s of eval; the slowest of the four accuracy entries.
register_amd_ci(
    est_time=2400,
    suite="nightly-amd-1-gpu-mi45x-cookbook-dsr1",
    nightly=True,
)


class TestDeepSeekR10528Mxfp4CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, DEEPSEEK_R1_0528_MXFP4)


if __name__ == "__main__":
    unittest.main()

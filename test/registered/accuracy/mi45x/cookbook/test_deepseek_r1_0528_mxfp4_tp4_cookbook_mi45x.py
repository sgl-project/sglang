"""MI455x cookbook entry 2: DeepSeek-R1-0528-MXFP4 GSM8K at TP=4.

The cookbook ran this model only at TP=1, so neither this entry nor the TP=2
one has a same-config measurement behind it; both gate against that 0.948.
DEEPSEEK_R1_0528_MXFP4_TP4 records the origin via `accuracy_measured_at_tp` so
a delta is not read as a regression of a configuration that was never
measured.

Registry: nightly-amd-4-gpu-mi45x-cookbook-dsr1 suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    DEEPSEEK_R1_0528_MXFP4_TP4,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 471.6s of eval at TP=1; the slowest of the accuracy entries.
register_amd_ci(
    est_time=2400,
    suite="nightly-amd-4-gpu-mi45x-cookbook-dsr1",
    nightly=True,
)


class TestDeepSeekR10528Mxfp4Tp4CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, DEEPSEEK_R1_0528_MXFP4_TP4)


if __name__ == "__main__":
    unittest.main()

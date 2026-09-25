"""MI455x cookbook entry 2: DeepSeek-R1-0528-MXFP4 GSM8K at TP=2.

This entry carries the longest env list in the cookbook -- fifteen
AITER/SGLANG toggles -- and they are transcribed verbatim in
DEEPSEEK_R1_0528_MXFP4_TP2; none of them has been independently re-derived, so
treat the set as a unit. The NCCL and HSA_HOTSWAP_DISABLE vars every TP>1 run
needs are not in that list because `full_env()` adds them for any tp_size > 1.

The cookbook ran this model only at TP=1 (0.948), which is the number gated on
here; see the TP=4 file for the same caveat.

Registry: nightly-amd-2-gpu-mi45x-cookbook-dsr1 suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    DEEPSEEK_R1_0528_MXFP4_TP2,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 471.6s of eval at TP=1; the slowest of the accuracy entries.
register_amd_ci(
    est_time=2400,
    suite="nightly-amd-2-gpu-mi45x-cookbook-dsr1",
    nightly=True,
)


class TestDeepSeekR10528Mxfp4Tp2CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, DEEPSEEK_R1_0528_MXFP4_TP2)


if __name__ == "__main__":
    unittest.main()

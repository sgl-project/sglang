"""MI455x cookbook entry 6: Qwen3.5-397B-A17B-MXFP4 GSM8K at TP=4.

The cookbook ran this model only at TP=1, so the 0.970 gate is carried over
from that run; QWEN35_397B_MXFP4_TP4 records this via `accuracy_measured_at_tp`.
That TP=1 run was also on a different image
(rocm/sgl-dev:v0.5.18-rocm10-mi45x-dev-20260828), so this entry is two steps
removed from a same-config measurement.

Registry: nightly-amd-4-gpu-mi45x-cookbook-qwen35 suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    QWEN35_397B_MXFP4_TP4,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 216.5s of eval at TP=1, but a 397B checkpoint to page in first.
register_amd_ci(
    est_time=2400,
    suite="nightly-amd-4-gpu-mi45x-cookbook-qwen35",
    nightly=True,
)


class TestQwen35397bMxfp4Tp4CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, QWEN35_397B_MXFP4_TP4)


if __name__ == "__main__":
    unittest.main()

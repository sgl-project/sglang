"""MI455x cookbook entry 6: Qwen3.5-397B-A17B-MXFP4 GSM8K at TP=1.

Reproduces the cookbook's measured 0.970. Unlike the other entries this one was
measured on rocm/sgl-dev:v0.5.18-rocm10-mi45x-dev-20260828, so a delta here may
be an image difference rather than a code regression.

This model reaches GDN linear attention through Triton kernels that still call
`tl.make_block_ptr`. Images whose triton build has removed block pointers in
favour of the tensor descriptor API fail on the first prefill rather than
producing a number, which is a launch failure, not an accuracy regression.

Registry: nightly-amd-1-gpu-mi45x-cookbook-qwen35 suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    QWEN35_397B_MXFP4_TP1,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 216.5s of eval, but a 397B checkpoint to page in first.
register_amd_ci(
    est_time=2400,
    suite="nightly-amd-1-gpu-mi45x-cookbook-qwen35",
    nightly=True,
)


class TestQwen35397bMxfp4Tp1CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, QWEN35_397B_MXFP4_TP1)


if __name__ == "__main__":
    unittest.main()

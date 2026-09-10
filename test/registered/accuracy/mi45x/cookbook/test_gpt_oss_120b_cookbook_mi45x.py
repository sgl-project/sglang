"""MI455x cookbook entry 1: GPT-OSS-120B MXFP4 GSM8K (1-GPU).

Reproduces the cookbook's measured 0.845 on MI455 A0. The configuration lives
in `sglang.test.ci.amd_mi45x_cookbook`; see GPT_OSS_120B there for why this runs
a single Triton attention backend with CUDA graphs off rather than the
prefill-triton/decode-aiter split used by
`test/registered/amd/accuracy/mi45x/test_gpt_oss_w4a8_mxfp4_eval_mi45x.py`.

Registry: nightly-amd-1-gpu-mi45x-cookbook-gpt-oss suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import GPT_OSS_120B, run_cookbook_case
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 332.6s of eval on top of a cold load of a 120B MXFP4 checkpoint.
register_amd_ci(
    est_time=1800,
    suite="nightly-amd-1-gpu-mi45x-cookbook-gpt-oss",
    nightly=True,
)


class TestGptOss120bCookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, GPT_OSS_120B)


if __name__ == "__main__":
    unittest.main()

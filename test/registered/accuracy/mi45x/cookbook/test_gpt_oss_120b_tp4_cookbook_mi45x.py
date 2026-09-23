"""MI455x cookbook entry 1: GPT-OSS-120B MXFP4 GSM8K at TP=4.

The cookbook only ran this model at TP=1, so the 0.845 gate here is carried
over from that run rather than measured at TP=4; GPT_OSS_120B_TP4 records this
via `accuracy_measured_at_tp` and the step summary says so. Accuracy is close
to TP-invariant at temperature 0 -- the cookbook's own DSV4 entries differ by
0.006 between TP=1 and TP=2, well inside ACCURACY_TOLERANCE -- so the gate is
still worth having, but a delta here is weaker evidence than one at TP=1.

Registry: nightly-amd-4-gpu-mi45x-cookbook-gpt-oss suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import GPT_OSS_120B_TP4, run_cookbook_case
from sglang.test.ci.ci_register import register_amd_ci

# Same cold load as the TP=1 half; the eval itself should be shorter.
register_amd_ci(
    est_time=1800,
    suite="nightly-amd-4-gpu-mi45x-cookbook-gpt-oss",
    nightly=True,
)


class TestGptOss120bTp4CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, GPT_OSS_120B_TP4)


if __name__ == "__main__":
    unittest.main()

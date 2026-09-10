"""MI455x cookbook entry 3: DeepSeek-V4-Flash GSM8K (TP=1 and TP=2).

Reproduces the cookbook's measured 0.932 at TP=1 and 0.926 at TP=2. Both live in
one file because they share a checkpoint and differ only in TP plus the NCCL env
the cookbook adds for the multi-GPU case, and because running them back to back
in one process keeps the pair comparable.

Note the cookbook measured TP=2 on heliosr-1b114-a04-2, not the A0 box the other
entries used, so a TP=2 delta on this hardware is not automatically a
regression.

Registry: nightly-amd-2-gpu-mi45x-cookbook-dsv4-flash suite
"""

import os
import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    DEEPSEEK_V4_FLASH_TP1,
    DEEPSEEK_V4_FLASH_TP2,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 124.4s + 177.0s of eval, but two cold loads of the same checkpoint.
# Registered as a 2-GPU suite because the TP=2 half needs two cards.
register_amd_ci(
    est_time=2400,
    suite="nightly-amd-2-gpu-mi45x-cookbook-dsv4-flash",
    nightly=True,
)


# Each half costs a cold load of the same 149 GB checkpoint, so allow running
# one without the other when only one of them is under investigation.
_SKIP_TP1 = os.environ.get("SGLANG_MI45X_SKIP_DSV4_TP1") == "1"
_SKIP_TP2 = os.environ.get("SGLANG_MI45X_SKIP_DSV4_TP2") == "1"


class TestDeepSeekV4FlashCookbookMI45x(unittest.TestCase):
    @unittest.skipIf(_SKIP_TP1, "SGLANG_MI45X_SKIP_DSV4_TP1=1")
    def test_gsm8k_accuracy_tp1(self):
        run_cookbook_case(self, DEEPSEEK_V4_FLASH_TP1)

    @unittest.skipIf(_SKIP_TP2, "SGLANG_MI45X_SKIP_DSV4_TP2=1")
    def test_gsm8k_accuracy_tp2(self):
        run_cookbook_case(self, DEEPSEEK_V4_FLASH_TP2)


if __name__ == "__main__":
    unittest.main()

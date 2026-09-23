"""MI455x cookbook entry 3: DeepSeek-V4-Flash GSM8K at TP=4.

Gates against the cookbook's TP=2 number (0.926) rather than its TP=1 (0.932),
that being the nearer multi-GPU measurement; TP=4 itself was never run.
DEEPSEEK_V4_FLASH_TP4 records this via `accuracy_measured_at_tp`.

Two further reasons a delta here is weak evidence on its own: the cookbook
measured TP=2 on heliosr-1b114-a04-2 rather than the A0 box the rest of the
entries used, and this is the entry whose multi-GPU launch hangs in CUDA-graph
capture if the image's NCCL_MIN_NCHANNELS leaks through -- `full_env()` drops
it for every tp_size > 1 config, so a hang here means that path broke.

Registry: nightly-amd-4-gpu-mi45x-cookbook-dsv4-flash suite
"""

import unittest

from sglang.test.ci.amd_mi45x_cookbook import (
    DEEPSEEK_V4_FLASH_TP4,
    run_cookbook_case,
)
from sglang.test.ci.ci_register import register_amd_ci

# Cookbook: 177.0s of eval at TP=2, plus a cold load of a 149 GB checkpoint.
register_amd_ci(
    est_time=1800,
    suite="nightly-amd-4-gpu-mi45x-cookbook-dsv4-flash",
    nightly=True,
)


class TestDeepSeekV4FlashTp4CookbookMI45x(unittest.TestCase):
    def test_gsm8k_accuracy(self):
        run_cookbook_case(self, DEEPSEEK_V4_FLASH_TP4)


if __name__ == "__main__":
    unittest.main()

import unittest

import test_unified_radix_cache_kl_dsv4 as dsv4_kl

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="extra-b", runner_config="8-gpu-h200")


class TestUnifiedDeepSeekV4FlashHiCachePP4TP2(
    dsv4_kl.TestUnifiedDeepSeekV4FlashHiCache
):
    """DeepSeek V4 Flash FP8 + HiCache + UnifiedRadixCache under PP4 TP2."""

    pp_size = 4
    tp_size = 2

    @unittest.skip("PP4TP2 coverage uses accuracy and cache-hit KL cases.")
    def test_multiturn_logprobs_match(self):
        pass


if __name__ == "__main__":
    unittest.main()

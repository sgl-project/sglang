"""Verify routed-expert gathering across attention TP ranks on Ascend."""

import unittest

import numpy as np
from sglang.test.ascend import test_npu_rl_weights as weights
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase
from test_npu_return_routed_experts import (
    ATTN_DP_SIZE,
    CASE,
    LONG_PROMPT,
    RoutedExpertsTestMixin,
    generate,
    seqlen,
)

register_npu_ci(est_time=600, suite="extra-b-test-4-npu-a3")
register_npu_ci(est_time=600, suite="nightly-test-4-npu-a3", nightly=True)


@unittest.skipUnless(weights.case_available(CASE), f"model not found: {CASE.model}")
class TestNpuRoutedExpertsAttentionTP(RoutedExpertsTestMixin, CustomTestCase):
    tp_size = 4
    # Tokens are scattered across the two attention TP ranks before R3 capture.
    moe_a2a_backend = "deepep"
    extra_server_args = ("--deepep-mode", "low_latency", "--disable-radix-cache")

    def test_gathered_experts_preserve_request_rows(self):
        start_len = 2
        for dp_rank in range(ATTN_DP_SIZE):
            with self.subTest(dp_rank=dp_rank):
                prompt = LONG_PROMPT * (dp_rank + 3)
                full = generate(self.base_url, prompt, routed_dp_rank=dp_rank)
                cropped = generate(
                    self.base_url,
                    prompt,
                    routed_dp_rank=dp_rank,
                    routed_experts_start_len=start_len,
                )
                experts = self._experts(full)
                cropped_experts = self._experts(cropped)
                self._check_shape_and_ids(experts)
                self.assertEqual(full["output_ids"], cropped["output_ids"])
                self.assertEqual(experts.shape[0], seqlen(full) - 1)
                np.testing.assert_array_equal(
                    cropped_experts[:, self.moe_start :, :],
                    experts[start_len:, self.moe_start :, :],
                    err_msg=f"incorrect routed-expert crop on DP rank {dp_rank}",
                )


if __name__ == "__main__":
    unittest.main()

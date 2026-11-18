"""CPU guard for the DSV4 clamped-SwiGLU activation quant's scale-layout checks in the DeepGEMM masked runner."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner import deep_gemm as runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _run(*, packed_ue8m0, hidden=2304, experts=8, tokens=16):
    gateup = torch.zeros((experts, tokens, 2 * hidden), dtype=torch.bfloat16)
    masked_m = torch.full((experts,), tokens, dtype=torch.int32)
    with (
        patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", packed_ue8m0),
        patch.object(runner, "silu_and_mul_masked_post_quant") as kernel,
    ):
        _, scale = runner._varlen_deep_gemm_silu_mul_quant(
            gateup, masked_m, group_size=128, topk=6, swiglu_limit=10.0
        )
    return kernel, scale


class TestDsv4SiluMulQuantGroups(CustomTestCase):
    def test_fp32_scales_take_a_group_count_not_divisible_by_four(self):
        """FP32 scales [E, N, G] take any group count, e.g. 18 for 2304-wide experts;
        only the packed UE8M0 layout [E, G // 4, N] needs G % 4 == 0."""
        kernel, scale = _run(packed_ue8m0=False)
        kernel.assert_called_once()
        self.assertEqual(tuple(scale.shape), (8, 16, 18))
        self.assertEqual(scale.dtype, torch.float32)

    def test_packed_ue8m0_scales_still_require_groups_of_four(self):
        """The relaxed check must not admit a packed UE8M0 layout it cannot index."""
        with self.assertRaisesRegex(AssertionError, "G % 4 == 0"):
            _run(packed_ue8m0=True)


if __name__ == "__main__":
    unittest.main(verbosity=3)

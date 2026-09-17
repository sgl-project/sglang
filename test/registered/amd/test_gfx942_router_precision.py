"""Router logits must be repeatable before discrete expert selection."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(is_gfx942_supported(), "requires gfx942")
class TestGfx942RouterPrecision(unittest.TestCase):
    """The defect is that AITER's tuned router GEMM may accumulate split-K partials in
    BF16, so identical forwards can disagree. That non-determinism cannot be triggered on
    demand, so these pin the contract that removes it instead: on gfx942 the router must
    not reach the tuned GEMM at all, and must produce FP32.
    """

    def test_gfx942_does_not_reach_the_tuned_gemm(self):
        """The tuned GEMM is the source of the BF16 split-K accumulation being avoided."""
        import sglang.srt.layers.rocm_linear_utils as utils

        x = torch.randn(7, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16) * 0.01

        with patch.object(utils, "_IS_GFX942", True), patch.object(
            utils.tgemm, "mm", side_effect=AssertionError("tuned GEMM must not be used on gfx942")
        ) as gemm:
            out = utils.aiter_dsv3_router_gemm(x, weight)

        gemm.assert_not_called()
        self.assertEqual(out.dtype, torch.float32)

    def test_gfx942_matches_an_fp32_matmul(self):
        """Independent reference: matmul against the transposed weight, not the same call."""
        import sglang.srt.layers.rocm_linear_utils as utils

        torch.manual_seed(123)
        weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
        for rows in (1, 7, 16, 32, 256):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                reference = torch.matmul(x.float(), weight.float().t())
                with patch.object(utils, "_IS_GFX942", True):
                    actual = utils.aiter_dsv3_router_gemm(x, weight)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_other_architectures_retain_tuned_dispatch(self):
        """Only gfx942 changes; every other ROCm target keeps the tuned dispatcher."""
        import sglang.srt.layers.rocm_linear_utils as utils

        x = torch.ones(1, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(8, 128, device="cuda", dtype=torch.bfloat16)
        with patch.object(utils, "_IS_GFX942", False), patch.object(
            utils.tgemm, "mm", return_value=x
        ) as gemm:
            self.assertIs(utils.aiter_dsv3_router_gemm(x, weight), x)

        self.assertIs(gemm.call_args.args[0], x)
        self.assertEqual(gemm.call_args.kwargs["otype"], x.dtype)


if __name__ == "__main__":
    unittest.main()

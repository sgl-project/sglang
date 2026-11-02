import unittest

import torch
import torch.nn.functional as F
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestJetGatedNorm(unittest.TestCase):
    def test_matches_fp32_reference_for_jet_head_shape(self):
        from sglang.srt.models.jet_nemotron import JetRMSNormGated

        torch.manual_seed(42)
        norm = JetRMSNormGated(256, eps=1e-5).to(device="npu", dtype=torch.bfloat16)
        weight = torch.randn(256, dtype=torch.bfloat16)
        norm.weight.data.copy_(weight)
        for tokens in (1, 4, 37):
            for gated in (False, True):
                with self.subTest(tokens=tokens, gated=gated):
                    x = torch.randn(tokens, 12, 256, dtype=torch.bfloat16)
                    z = torch.randn_like(x) if gated else None
                    expected = x.float() * torch.rsqrt(
                        x.float().square().mean(dim=-1, keepdim=True) + norm.eps
                    )
                    expected = expected * weight.float()
                    if z is not None:
                        expected = expected * F.silu(z.float())
                    with torch.no_grad():
                        actual = norm(x.npu(), z.npu() if z is not None else None)
                    self.assertEqual(actual.dtype, x.dtype)
                    self.assertEqual(actual.shape, x.shape)
                    torch.testing.assert_close(
                        actual.cpu(), expected.to(x.dtype), rtol=1e-2, atol=1e-3
                    )


if __name__ == "__main__":
    unittest.main()

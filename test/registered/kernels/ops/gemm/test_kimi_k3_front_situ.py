"""The gfx950 Kimi-K3 MoE front GEMM with a SiTU epilogue against GEMM then SiTU."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the fused front is tuned for and selected on gfx950 only.
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8 merged MoE front rows: shared [gate 768 | up 768], router 896, latent 3584.
GATE_UP, EXPERTS, LATENT, HIDDEN = 1536, 896, 3584, 7168
BETA, LINEAR_BETA = 4.0, 25.0


def _situ(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    return gate * LINEAR_BETA * torch.tanh(up / LINEAR_BETA)


@unittest.skipUnless(is_gfx95_supported(), "gfx950 fused MoE front")
class TestKimiK3FrontSitu(CustomTestCase):
    @torch.inference_mode()
    def test_matches_gemm_then_situ(self):
        """The fused front pairs gate row j with up row 768 + j and splits the
        rest into router logits and latent: within one bf16 rounding of an fp32
        GEMM followed by SiTU on its bf16 output."""
        from sglang.kernels.ops.gemm.kimi_k3_front_situ import front_situ

        g = torch.Generator(device="cuda").manual_seed(0)
        weight = (
            torch.randn(GATE_UP + EXPERTS + LATENT, HIDDEN, generator=g, device="cuda")
            * 0.02
        ).bfloat16()
        for rows in (144, 300, 1200):
            x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
            full = (x.float() @ weight.float().t()).bfloat16()
            act, rest = front_situ(x, weight, GATE_UP, BETA, LINEAR_BETA)
            for name, got, want in (
                ("situ", act, _situ(full[:, :GATE_UP])),
                ("router|latent", rest, full[:, GATE_UP:]),
            ):
                with self.subTest(rows=rows, output=name):
                    want = want.float()
                    beyond_ulp = (got.float() - want).abs() > want.abs() * 2**-7 + 1e-3
                    self.assertLessEqual(beyond_ulp.sum().item(), got.numel() // 10000)


if __name__ == "__main__":
    unittest.main()

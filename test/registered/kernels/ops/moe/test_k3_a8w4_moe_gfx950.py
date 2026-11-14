"""Kimi-K3's gfx950 A8W4 MoE must match AITER's A8W4 fused MoE."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=25, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950():
    return (
        bool(torch.version.hip)
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestK3A8W4MoeGfx950(CustomTestCase):
    def test_matches_aiter_a8w4(self):
        # K3.1's TP8 shard (hidden 3584, expert width 384, top-16) at its own expert
        # count: AITER only has tuned A8W4 kernels for it (other shapes fall back to a
        # CK path this image cannot build). Random bytes in the preshuffled layout
        # serve both sides alike.
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        from sglang.kernels.ops.moe.k3_a8w4_gfx950 import k3_a8w4_moe
        from sglang.srt.utils.common import temp_set_env

        E, TOPK, K, I, N, T = 896, 16, 3584, 384, 3584, 128
        BETA, LBETA = 4.0, 25.0
        g = torch.Generator(device="cuda").manual_seed(2)
        u8 = dict(dtype=torch.uint8, device="cuda", generator=g)
        w13 = torch.randint(0, 256, (E, 2 * I, K // 2), **u8)
        w13s = torch.randint(118, 124, (E * 2 * I, K // 32), **u8)
        w2 = torch.randint(0, 256, (E, N, I // 2), **u8)
        w2s = torch.randint(118, 124, (E * N, (I // 32 + 7) // 8 * 8), **u8)
        x = torch.randn(T, K, device="cuda", generator=g).to(torch.bfloat16)
        wt, ids = torch.rand(T, E, device="cuda", generator=g).topk(TOPK, dim=1)
        ids, wt = (
            ids.to(torch.int32).contiguous(),
            (wt / wt.sum(1, keepdim=True)).contiguous(),
        )

        a13, a2 = w13.view(torch.float4_e2m1fn_x2), w2.view(torch.float4_e2m1fn_x2)
        a13.is_shuffled = True
        a2.is_shuffled = True
        # AITER picks MXFP8 activations for SiTUv2 from this env, read per call; bf16 stage-2 partials
        with temp_set_env(AITER_SITUV2_A8W4="1", AITER_OPUS_STAGE2_FP8="0"):
            ref = fused_moe(
                x,
                a13,
                a2,
                wt,
                ids,
                quant_type=QuantType.per_1x32,
                activation=ActivationType.Situv2,
                w1_scale=w13s,
                w2_scale=w2s,
                gate_mode=GateMode.INTERLEAVE.value,  # w13 is GU-interleaved (#49152)
                beta=BETA,
                linear_beta=LBETA,
            )
        # the output is a row-strided view, as the latent half of K3's [latent | shared]
        buf = torch.zeros(T, N + 64, dtype=torch.bfloat16, device="cuda")
        out = k3_a8w4_moe(
            x, ids, wt, w13, w13s, w2, w2s, beta=BETA, linear_beta=LBETA, out=buf[:, :N]
        )
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        # measured 3.2e-3 vs AITER's bf16-partials path on MI350X
        self.assertLess(rel, 2e-2, f"vs AITER A8W4: rel_l2={rel:.3e}")
        self.assertEqual(buf[:, N:].abs().max().item(), 0.0, "wrote past the view")


if __name__ == "__main__":
    unittest.main()

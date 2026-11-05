"""Kimi-K3's gfx950 A8W4 MoE must match dense and AITER references."""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

_FP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_LUT = _FP4_MAGNITUDES + [-v for v in _FP4_MAGNITUDES]  # e2m1 code -> value
LOG2E = 1.0 / math.log(2.0)


def _gfx950():
    return (
        bool(torch.version.hip)
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


def _fp4_dequant(raw, scale):
    lut = torch.tensor(FP4_LUT, device=raw.device)
    codes = torch.stack([raw & 15, raw >> 4], -1).flatten(-2)
    return lut[codes.long()] * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def _fp8_dequant(q_u8, s_u8):
    return q_u8.view(torch.float8_e4m3fn).float() * torch.exp2(
        s_u8.float() - 127
    ).repeat_interleave(32, -1)


def _situ_v2(g, u, beta, lbeta):
    def tanh(x):
        e = torch.exp2(x.abs() * (-2.0 * LOG2E))
        t = (1 - e) / (1 + e)
        return torch.where(x > 0, t, -t)

    a = (beta * tanh(g * (1.0 / beta))) / (1.0 + torch.exp2(g * -LOG2E))
    return (a * lbeta) * tanh(u * (1.0 / lbeta))


def _mxfp8_round_trip(h):
    """AITER's stage-1 output quant: ceil-pow2 scale of amax / 448 per 32, RNE e4m3."""
    hb = h.view(h.shape[0], -1, 32)
    v = hb.abs().amax(-1).clamp_min(1e-10) * torch.tensor(1.0 / 448.0)
    bits = v.view(torch.int32)
    code = (
        (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(1, 254).float()
    )
    q = (
        (hb * torch.exp2(127.0 - code)[..., None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    return (q.float() * torch.exp2(code - 127)[..., None]).view_as(h)


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestK3A8W4MoeGfx950(CustomTestCase):
    # K3's TP8 shard (hidden 3584, expert width 384, top-16) over fewer experts
    E, TOPK, K, I, N = 32, 16, 3584, 384, 3584
    BETA, LBETA = 4.0, 25.0

    @classmethod
    def setUpClass(cls):
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        from sglang.kernels.ops.moe import k3_a8w4_gfx950 as M

        cls.M = M
        g = torch.Generator(device="cuda").manual_seed(0)
        E, K, I, N = cls.E, cls.K, cls.I, cls.N
        u8 = dict(dtype=torch.uint8, device="cuda", generator=g)
        # w13 rows: [gate; up], the layout K3 loads (gate_up_interleaved=False)
        cls.w13_raw = torch.randint(0, 256, (E, 2 * I, K // 2), **u8)
        cls.w13s_raw = torch.randint(118, 124, (E, 2 * I, K // 32), **u8)
        cls.w2_raw = torch.randint(0, 256, (E, N, I // 2), **u8)
        cls.w2s_raw = torch.randint(118, 124, (E, N, I // 32), **u8)
        cls.w13 = shuffle_weight_a16w4(cls.w13_raw, 16, True)
        cls.w13s = shuffle_scale_a16w4(cls.w13s_raw.view(-1, K // 32), E, True)
        cls.w2 = shuffle_weight_a16w4(cls.w2_raw, 16, False)
        cls.w2s = shuffle_scale_a16w4(cls.w2s_raw.view(-1, I // 32), E, False)

    def _inputs(self, T, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(T, self.K, device="cuda", generator=g).to(torch.bfloat16)
        w, ids = torch.rand(T, self.E, device="cuda", generator=g).topk(
            self.TOPK, dim=1
        )
        return (
            x,
            ids.to(torch.int32).contiguous(),
            (w / w.sum(1, keepdim=True)).contiguous(),
        )

    def _reference(self, x, ids, w):
        a, a_s = self.M.quant_mxfp8(x)
        xa = _fp8_dequant(a, a_s)
        out = torch.zeros(x.shape[0], self.N, device="cuda")
        for e in ids.unique().tolist():
            tok, slot = (ids == e).nonzero(as_tuple=True)
            gu = xa[tok] @ _fp4_dequant(self.w13_raw[e], self.w13s_raw[e]).T
            h = _mxfp8_round_trip(
                _situ_v2(gu[:, : self.I], gu[:, self.I :], self.BETA, self.LBETA)
            )
            y = (
                (h @ _fp4_dequant(self.w2_raw[e], self.w2s_raw[e]).T)
                .to(torch.bfloat16)
                .float()
            )
            out.index_add_(0, tok, w[tok, slot, None] * y)
        return out

    def _run(self, x, ids, w, out=None, **overrides):
        run = self.M._k3_a8w4_moe if overrides else self.M.k3_a8w4_moe
        return run(
            x,
            ids,
            w,
            self.w13,
            self.w13s,
            self.w2,
            self.w2s,
            beta=self.BETA,
            linear_beta=self.LBETA,
            out=out,
            **overrides,
        )

    @staticmethod
    def _rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    def test_matches_dense_reference(self):
        # partial last blocks (T=5), several blocks per expert (T=200), every block_m pick_block_m returns;
        # the output is a row-strided view, as the K3 latent half of its [latent | shared] buffer
        for T, block_m in ((5, 16), (64, 32), (200, 64)):
            self.assertEqual(self.M.pick_block_m(T, self.TOPK, self.E), block_m)
            x, ids, w = self._inputs(T, seed=T + block_m)
            buf = torch.zeros(T, self.N + 64, dtype=torch.bfloat16, device="cuda")
            out = self._run(x, ids, w, out=buf[:, : self.N])
            ref = self._reference(x, ids, w)
            rel = self._rel(out, ref)
            # measured 1.8e-3 on MI350X (bf16 down rows, MXFP8 intermediate both sides)
            self.assertLess(rel, 1e-2, f"T={T} block_m={block_m}: rel_l2={rel:.3e}")
            self.assertFalse(out.isnan().any().item())
            self.assertEqual(
                buf[:, self.N :].abs().max().item(), 0.0, "wrote past the view"
            )

    def test_skewed_routing_segments(self):
        # hot experts span many blocks and are split over several down work items, cold experts have none;
        # the split must not change a single output bit
        T, block_m = 300, 16
        x, ids, w = self._inputs(T, seed=3)
        g = torch.Generator(device="cuda").manual_seed(4)
        bias = torch.zeros(self.E, device="cuda")
        bias[:6] = 10.0
        bias[24:] = -10.0
        w, ids = (torch.rand(T, self.E, device="cuda", generator=g) + bias).topk(
            self.TOPK, dim=1
        )
        ids, w = (
            ids.to(torch.int32).contiguous(),
            (w / w.sum(1, keepdim=True)).contiguous(),
        )
        ref = self._reference(x, ids, w)
        whole = self._run(x, ids, w, block_m=block_m, seg_blocks=1 << 20)
        self.assertLess(self._rel(whole, ref), 1e-2)
        for seg in (3, 8):  # non-power-of-two tail and production segment size
            out = self._run(x, ids, w, block_m=block_m, seg_blocks=seg)
            self.assertTrue(torch.equal(out, whole), f"seg_blocks={seg}")

    def test_matches_aiter_a8w4(self):
        # K3.1's own expert count: AITER only has tuned A8W4 kernels for it (other shapes fall back to a CK
        # path this image cannot build). Random bytes in the preshuffled layout serve both sides alike.
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        from sglang.srt.utils.common import temp_set_env

        E, K, I, N, T = 896, self.K, self.I, self.N, 128
        g = torch.Generator(device="cuda").manual_seed(2)
        u8 = dict(dtype=torch.uint8, device="cuda", generator=g)
        w13 = torch.randint(0, 256, (E, 2 * I, K // 2), **u8)
        w13s = torch.randint(118, 124, (E * 2 * I, K // 32), **u8)
        w2 = torch.randint(0, 256, (E, N, I // 2), **u8)
        w2s = torch.randint(118, 124, (E * N, (I // 32 + 7) // 8 * 8), **u8)
        x = torch.randn(T, K, device="cuda", generator=g).to(torch.bfloat16)
        wt, ids = torch.rand(T, E, device="cuda", generator=g).topk(self.TOPK, dim=1)
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
                gate_mode=GateMode.SEPARATED.value,
                beta=self.BETA,
                linear_beta=self.LBETA,
            )
        out = self.M.k3_a8w4_moe(
            x, ids, wt, w13, w13s, w2, w2s, beta=self.BETA, linear_beta=self.LBETA
        )
        rel = self._rel(out, ref)
        # measured 3.2e-3 vs AITER's bf16-partials path on MI350X
        self.assertLess(rel, 2e-2, f"vs AITER A8W4: rel_l2={rel:.3e}")


if __name__ == "__main__":
    unittest.main()

"""DeepSeek-V4 dense 128x128 block-fp8 linears on aiter's e8m0 mxscale bpreshuffle
GEMM (SGLANG_DSV4_DENSE_MXSCALE) against the fp32-scale bpreshuffle route and an
fp32 dequant reference; the fused RMSNorm + ue8m0 quant producer against torch and
aiter's per_group_quant_hip."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# (N, K) of the DSv4-Pro dense fp8 projections: wqkv_a, wq_b TP8 / indexer wq_b,
# wo_b TP8, wo_b attn-TP1, wq_b attn-TP1, shared-expert down TP1, DSpark main_proj.
SHAPES = [
    (2048, 7168),
    (8192, 1536),
    (7168, 2048),
    (7168, 16384),
    (65536, 1536),
    (7168, 3072),
    (7168, 21504),
]
MS = [1, 7, 64, 129, 256, 2048]


def _ue8m0_ref_quant(x: torch.Tensor):
    """Per-token 1x128 fp8 quant, scale = ceil_pow2(amax / 448) as fp32."""
    m, k = x.shape
    g = x.float().view(m, k // 128, 128)
    s = g.abs().amax(-1, keepdim=True) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = s.view(torch.int32)
    e = (bits >> 23) & 0xFF
    e = torch.where(((bits & 0x7FFFFF) != 0) & (e < 0xFF), e + 1, e)
    scale = (e << 23).view(torch.float32)
    inv = torch.where(e > 0, 1.0 / scale, torch.zeros_like(scale))
    q = (g * inv).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    return q, e.view(m, k // 128).to(torch.uint8), scale.view(m, k // 128)


def _aiter_has_dense_mxscale() -> bool:
    if not (is_hip() and is_gfx95_supported()):
        return False
    from sglang.srt.layers.quantization.fp8_utils import aiter_has_dense_mxscale

    return aiter_has_dense_mxscale()


@unittest.skipUnless(_aiter_has_dense_mxscale(), "gfx950 aiter mxscale GEMM")
class TestDsv4DenseMxscale(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization import fp8_utils

        fp8_utils.set_force_ck_w8a8(True)
        fp8_utils.set_dense_mxscale(True)
        cls.fu = fp8_utils

    @classmethod
    def tearDownClass(cls):
        cls.fu.set_dense_mxscale(False)

    def _weight(self, n, k, seed):
        from aiter.ops.shuffle import shuffle_weight

        g = torch.Generator(device="cuda").manual_seed(seed)
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )
        exp = torch.randint(-12, -4, (n // 128, k // 128), device="cuda", generator=g)
        scale = torch.exp2(exp.float()).contiguous()
        w_deq = (
            w.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None]
        ).view(n, k)
        return shuffle_weight(w, (16, 16)), scale, w_deq

    def test_scale_to_e8m0_is_lossless_and_rejects_non_pow2(self):
        s = torch.exp2(torch.randint(-30, 10, (64, 12), device="cuda").float())
        e = self.fu.block_fp8_scale_to_e8m0_128(s)
        self.assertEqual(e.dtype, torch.float8_e8m0fnu)
        back = torch.exp2(e.view(torch.uint8).float() - 127)
        torch.testing.assert_close(back, s, rtol=0, atol=0)
        s2 = s.clone()
        s2[3, 4] *= 1.5
        self.assertIsNone(self.fu.block_fp8_scale_to_e8m0_128(s2))
        self.assertIsNone(self.fu.block_fp8_scale_to_e8m0_128(torch.zeros_like(s)))

    def test_rmsnorm_fp8_e8m0_producer(self):
        from aiter import dtypes
        from aiter.ops.quant import per_group_quant_hip

        from sglang.kernels.ops.layernorm.rmsnorm_fp8_e8m0 import (
            rmsnorm_fp8_e8m0_quant,
        )

        for k in (7168, 1536):
            for m in (1, 5, 96, 1024):
                with self.subTest(k=k, m=m):
                    torch.manual_seed(m + k)
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3
                    x[0, :256] = 0  # an all-zero 128-group: scale byte 0, q 0
                    w = torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5
                    (q, s), y = rmsnorm_fp8_e8m0_quant(x, w, 1e-6)
                    xf = x.float()
                    y_ref = (
                        xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
                    ) * w.float()
                    torch.testing.assert_close(
                        y.float(),
                        y_ref.to(torch.bfloat16).float(),
                        atol=2e-2,
                        rtol=1e-2,
                    )
                    self.assertEqual(tuple(s.shape), (m, k // 128))
                    self.assertEqual(s.dtype, torch.float8_e8m0fnu)
                    # column-major bytes: [K/128, M] storage
                    s_rm = s.view(torch.uint8).view(k // 128, m).t()
                    q_ref, e_ref, _ = _ue8m0_ref_quant(y)
                    self.assertTrue(torch.equal(s_rm, e_ref))
                    mism = (q.view(torch.uint8) != q_ref.view(torch.uint8)).float()
                    self.assertLess(mism.mean().item(), 1e-3)
                    # same bytes as aiter's own e8m0 quant of the bf16 norm output
                    q_a, s_a = per_group_quant_hip(
                        y,
                        quant_dtype=dtypes.fp8,
                        group_size=128,
                        transpose_scale=True,
                        scale_type=dtypes.fp8_e8m0,
                    )
                    # (aiter writes its own byte for an all-zero group; its q is 0 too)
                    live = e_ref != 0
                    s_a_rm = s_a.view(torch.uint8).view(k // 128, m).t()
                    self.assertTrue(torch.equal(s_a_rm[live], s_rm[live]))
                    mism = (q_a.view(torch.uint8) != q.view(torch.uint8)).float()
                    self.assertLess(mism.mean().item(), 1e-3)

    def test_linear_matches_fp32_route(self):
        fu = self.fu
        for n, k in SHAPES:
            wsh, s32, w_deq = self._weight(n, k, seed=n + k)
            mx = fu.block_fp8_scale_to_e8m0_128(s32)
            self.assertIsNotNone(mx)
            for m in MS:
                with self.subTest(n=n, k=k, m=m):
                    torch.manual_seed(m)
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    # fp32 route (current path)
                    y_fp32 = fu.aiter_w8a8_block_fp8_linear(x, wsh, [128, 128], s32)
                    # e8m0 route, bf16 input (the linear quantizes)
                    wsh._mxscale_w_scale = mx
                    try:
                        y_mx = fu.aiter_w8a8_block_fp8_linear(x, wsh, [128, 128], s32)
                        # e8m0 route, pre-quantized ue8m0 operand (fused producer)
                        q, sc = fu.quant_fp8_e8m0_128(x)
                        y_mx_pre = fu.aiter_w8a8_block_fp8_linear(
                            q, wsh, [128, 128], s32, input_scale=sc
                        )
                    finally:
                        del wsh._mxscale_w_scale
                    # exact reference of the e8m0 operands
                    q_ref, _, a_scale = _ue8m0_ref_quant(x)
                    a_deq = (
                        q_ref.float().view(m, k // 128, 128) * a_scale[..., None]
                    ).view(m, k)
                    ref_mx = a_deq @ w_deq.t()
                    ref_bf16 = x.float() @ w_deq.t()
                    nrm = ref_bf16.norm().clamp_min(1e-6)
                    err_mx = ((y_mx.float() - ref_mx).norm() / nrm).item()
                    self.assertLess(err_mx, 1e-2)
                    torch.testing.assert_close(y_mx_pre, y_mx, rtol=0, atol=0)
                    # both routes sit at the same distance from the bf16 product
                    e_fp32 = ((y_fp32.float() - ref_bf16).norm() / nrm).item()
                    e_mx = ((y_mx.float() - ref_bf16).norm() / nrm).item()
                    self.assertLess(e_mx, 1.25 * e_fp32 + 2e-3)
                    self.assertLess(
                        ((y_mx.float() - y_fp32.float()).norm() / nrm).item(), 6e-2
                    )
                    self.assertTrue(torch.isfinite(y_mx).all())

    def test_route_selection(self):
        fu = self.fu
        wsh, s32, _ = self._weight(2048, 7168, seed=1)
        x = torch.randn(8, 7168, device="cuda", dtype=torch.bfloat16)
        q, sc = fu.quant_fp8_e8m0_128(x)
        # an e8m0 operand into a linear without an e8m0 weight scale is an error
        with self.assertRaises(RuntimeError):
            fu.aiter_w8a8_block_fp8_linear(q, wsh, [128, 128], s32, input_scale=sc)
        # an fp32 operand into an e8m0 linear keeps the fp32 GEMM
        wsh._mxscale_w_scale = fu.block_fp8_scale_to_e8m0_128(s32)
        try:
            from sglang.srt.layers.quantization.fp8_utils import (
                aiter_per1x128_quant,
                view_aiter_fused_rms_transposed_fp8_scale,
            )

            q32, s32x = aiter_per1x128_quant(
                x, quant_dtype=torch.float8_e4m3fn, transpose_scale=True
            )
            s32x = view_aiter_fused_rms_transposed_fp8_scale(s32x)
            y = fu.aiter_w8a8_block_fp8_linear(
                q32, wsh, [128, 128], s32, input_scale=s32x
            )
            del wsh._mxscale_w_scale
            y_ref = fu.aiter_w8a8_block_fp8_linear(
                q32, wsh, [128, 128], s32, input_scale=s32x
            )
            torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
        finally:
            if hasattr(wsh, "_mxscale_w_scale"):
                del wsh._mxscale_w_scale

    def test_shape_whitelist_and_ready(self):
        fu = self.fu
        self.assertTrue(fu.dense_mxscale_wanted(2048, 7168))
        self.assertTrue(fu.dense_mxscale_wanted(65536, 1536))
        self.assertFalse(fu.dense_mxscale_wanted(512, 7168))

        class _L:
            def __init__(self, mx):
                self.weight = torch.empty(1)
                if mx:
                    self.weight._mxscale_w_scale = torch.empty(1)

        self.assertTrue(fu.dense_mxscale_ready(_L(True), None, _L(True)))
        self.assertFalse(fu.dense_mxscale_ready(_L(True), _L(False)))

    def test_process_weights_installs_e8m0_scale(self):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.weight_block_size = [128, 128]
        method.convert_mxfp8_to_block = False
        for (n, k), pow2, want in (
            ((8192, 1536), True, True),
            ((512, 7168), True, False),  # not whitelisted
            ((7168, 2048), False, False),  # non power-of-two scales
        ):
            with self.subTest(n=n, k=k, pow2=pow2):
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(
                    torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                s = torch.full((n // 128, k // 128), 0.25, device="cuda")
                if not pow2:
                    s[0, 0] = 0.3
                layer.weight_scale_inv = torch.nn.Parameter(s, requires_grad=False)
                method._maybe_prepare_dense_mxscale(layer)
                got = getattr(layer.weight, "_mxscale_w_scale", None)
                self.assertEqual(got is not None, want)
                if want:
                    self.assertTrue(
                        torch.equal(
                            got.view(torch.uint8),
                            torch.full_like(s, 125, dtype=torch.uint8),
                        )
                    )


if __name__ == "__main__":
    unittest.main()

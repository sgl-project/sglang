"""gfx950 test: one AITER ASM FP4 GEMM over a packed Qwen3.5 GDN in_proj
(qkvz in ASM layout + in_proj_ba as a padded tail) against the separate
qkvz ASM GEMM and an exact reference for the in_proj_ba rows."""

import unittest
from unittest import mock

import torch

from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported() and get_bool_env_var("SGLANG_USE_AITER")


def _make_layer(scheme, n, k, gen):
    layer = torch.nn.Module()
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    # E8M0 2^-7..2^0 keeps the outputs well inside bf16 range.
    scale = torch.randint(120, 128, (n, k // 32), dtype=torch.uint8, generator=gen)
    layer.weight = torch.nn.Parameter(weight.cuda(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.cuda(), requires_grad=False)
    raw = (weight.cuda(), scale.cuda())
    scheme.process_weights_after_loading(layer)
    return layer, raw


def _dequant(codes, scale, values):
    """Row-major fp4x2 codes and E8M0 scales to float32."""
    rows, k_packed = codes.shape
    lut = torch.tensor(values, dtype=torch.float32, device=codes.device)
    vals = torch.empty(rows, k_packed * 2, dtype=torch.float32, device=codes.device)
    vals[:, 0::2] = lut[(codes & 0xF).long()]
    vals[:, 1::2] = lut[(codes >> 4).long()]
    scale = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    return (vals.view(rows, -1, 32) * scale.unsqueeze(-1)).view(rows, -1)


def _unswizzle_scale(scale):
    rows, cols = scale.shape
    return (
        scale.view(torch.uint8)
        .view(rows // 32, cols // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(rows, cols)
    )


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx950 with SGLANG_USE_AITER=1")
class TestPackedMxfp4InProjGpu(CustomTestCase):
    def test_packed_gemm_matches_separate_projections(self):
        from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4

        # Qwen3.8-27B TP1 GDN in_proj: K=5120, in_proj_ba N=96.
        k, n_qkvz, n_ba = 5120, 12288, 96
        gen = torch.Generator().manual_seed(0)
        scheme = quark_w4a4_mxfp4.QuarkW4A4MXFP4({}, {})
        scheme.out_dtype = torch.bfloat16
        with mock.patch.object(quark_w4a4_mxfp4, "_use_aiter_asm_fp4_gemm", True):
            qkvz, _ = _make_layer(scheme, n_qkvz, k, gen)
            ba, (ba_w, ba_s) = _make_layer(scheme, n_ba, k, gen)
        self.assertTrue(qkvz.use_aiter_asm_fp4_gemm)
        self.assertFalse(ba.use_aiter_asm_fp4_gemm)

        weight, scale = quark_w4a4_mxfp4.pack_asm_fp4_weight_tail(
            qkvz.weight.data,
            qkvz.weight_scale.data,
            ba.weight.data,
            ba.weight_scale.data,
        )
        ba_ref_w = _dequant(ba_w, ba_s, quark_w4a4_mxfp4._MXFP4_VALUES)
        out_ref = torch.empty(0, dtype=torch.bfloat16, device="cuda")
        # 65..256 rows: the CUDA-graph-only range of the packed in_proj.
        for m in (1, 2, 7, 16, 33, 64, 65, 128, 200, 256):
            with self.subTest(m=m):
                x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                x_q, x_scales = quark_w4a4_mxfp4.per_1x32_f4_quant(x)
                packed = quark_w4a4_mxfp4.gemm_a4w4(
                    x_q, weight, x_scales, scale, out_ref
                )
                # The separate ASM GEMM quantizes x the same way; only the
                # kernel config (tuned per N) may change the summation order.
                qkvz_out = scheme.apply_weights(qkvz, x)
                torch.testing.assert_close(
                    packed[:, :n_qkvz].float(),
                    qkvz_out.float(),
                    rtol=1e-2,
                    atol=1e-3 * qkvz_out.float().abs().max().item(),
                )
                # in_proj_ba rows: exact dequantized operands, fp32 matmul.
                x_ref = _dequant(
                    x_q.view(torch.uint8),
                    _unswizzle_scale(x_scales)[:m, : k // 32],
                    quark_w4a4_mxfp4._MXFP4_VALUES,
                )
                ba_ref = x_ref @ ba_ref_w.t()
                torch.testing.assert_close(
                    packed[:, n_qkvz : n_qkvz + n_ba].float(),
                    ba_ref,
                    rtol=1e-2,
                    atol=1e-3 * ba_ref.abs().max().item(),
                )
                self.assertTrue(bool((packed[:, n_qkvz + n_ba :] == 0).all()))


if __name__ == "__main__":
    unittest.main()

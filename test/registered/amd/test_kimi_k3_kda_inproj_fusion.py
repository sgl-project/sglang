"""Layout check for the ROCm fused Kimi-K3 KDA input projection.

Below SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS the whole in-proj is one GEMM
over ``[q,k,v,g | f_a | b | pad]`` instead of a wide GEMM plus a tiny [f_a|b]
GEMV. Both layouts are views over the same buffer, so the two paths have to
agree; this pins the slice offsets, the tail view the split path still reads,
and the fact that the strided f_a slice is a legal input to the f_b GEMM and
to the fused decode kernel's shape gate.

The two paths run different GEMM kernels (N=6288 has no tuned aiter config,
N=6144 does), so they agree to bf16 rounding, not bitwise.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    _merge_weights_as_views,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN = 7168
HEADS_TP = 12  # num_heads / tp8
HEAD_DIM = 128
PROJ_TP = HEADS_TP * HEAD_DIM  # 1536
WIDE = 4 * PROJ_TP  # 6144, [q,k,v,g]
MERGED = 6288  # + f_a(128) + b(12) + pad(4)
# bf16 carries ~8 mantissa bits, so one ULP is ~4e-3 relative.
TOL = 8e-3


class _Fake(torch.nn.Module):
    """Minimal stand-in for a linear layer: _merge_weights_as_views only
    touches .weight.data."""

    def __init__(self, rows, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02,
            requires_grad=False,
        )


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3KDAInProjFusion(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = torch.device("cuda", 0)
        cls.qkvg = _Fake(WIDE, dev)
        cls.f_a = _Fake(HEAD_DIM, dev)
        cls.b = _Fake(HEADS_TP, dev)
        cls.pre = [m.weight.data.clone() for m in (cls.qkvg, cls.f_a, cls.b)]
        cls.f_b_w = (
            torch.randn(PROJ_TP, HEAD_DIM, dtype=torch.bfloat16, device=dev) * 0.02
        )
        cls.merged, cls.sizes = _merge_weights_as_views(
            [cls.qkvg, cls.f_a, cls.b], pad_rows_to=8
        )
        cls.split_sizes = [3 * PROJ_TP, PROJ_TP]
        cls.all_sizes = cls.split_sizes + [
            HEAD_DIM,
            HEADS_TP,
            MERGED - WIDE - HEAD_DIM - HEADS_TP,
        ]

    def test_merged_layout(self):
        self.assertEqual(self.sizes, [WIDE, HEAD_DIM, HEADS_TP])
        self.assertEqual(tuple(self.merged.shape), (MERGED, HIDDEN))
        self.assertEqual(sum(self.all_sizes), MERGED)

    def test_views_alias_and_preserve_values(self):
        """The merge must re-point, not reorder or copy-and-drop."""
        tail = self.merged[WIDE:]
        self.assertEqual(tail.data_ptr(), self.f_a.weight.data_ptr())
        self.assertTrue(tail.is_contiguous())
        self.assertTrue(self.qkvg.weight.is_contiguous())
        self.assertEqual(self.qkvg.weight.data_ptr(), self.merged.data_ptr())
        for got, want in zip((self.qkvg, self.f_a, self.b), self.pre):
            self.assertTrue(torch.equal(got.weight.data, want))

    def test_split_and_fused_paths_agree(self):
        tail = self.merged[WIDE:]
        for tokens in (1, 4, 8, 33, 128, 256):
            with self.subTest(tokens=tokens):
                x = torch.randn(
                    tokens, HIDDEN, dtype=torch.bfloat16, device=self.merged.device
                )

                wide = torch.nn.functional.linear(x, self.qkvg.weight)
                s_qkv, s_g = torch.split(wide, self.split_sizes, dim=-1)
                s_bfa = kimi_k3_tiny_gemm(x, tail)
                s_beta = s_bfa[..., HEAD_DIM : HEAD_DIM + HEADS_TP]
                s_fg = kimi_k3_tiny_gemm(s_bfa[..., :HEAD_DIM], self.f_b_w)

                allp = torch.nn.functional.linear(x, self.merged)
                f_qkv, f_g, f_fa, f_beta, _pad = torch.split(
                    allp, self.all_sizes, dim=-1
                )
                # The fused decode kernel's shape gate requires a unit last
                # stride but tolerates the wider row stride.
                self.assertEqual(f_fa.stride(-1), 1)
                self.assertEqual(f_qkv.stride(-1), 1)
                self.assertEqual(f_fa.stride(0), MERGED)
                f_fg = kimi_k3_tiny_gemm(f_fa, self.f_b_w)

                for name, got, want in (
                    ("qkv", f_qkv, s_qkv),
                    ("g", f_g, s_g),
                    ("beta", f_beta, s_beta),
                    ("forget_gate", f_fg, s_fg),
                ):
                    scale = want.float().abs().max().clamp_min(1e-6)
                    rel = ((got.float() - want.float()).abs().max() / scale).item()
                    self.assertLess(rel, TOL, f"{name} rel err {rel:.2e}")

    @unittest.skipUnless(torch.version.hip is not None, "ROCm-only fused path")
    def test_deferred_f_b_returns_raw_f_a(self):
        x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=self.merged.device)
        fused_states = torch.nn.functional.linear(x, self.merged)
        expected_f_a = torch.split(fused_states, self.all_sizes, dim=-1)[2]
        fake_attention = SimpleNamespace(
            use_full_rank_gate=True,
            _bfa_w=object(),
            _bfa_fa_size=HEAD_DIM,
            _bfa_b_size=HEADS_TP,
            _qkvgbfa_sizes=self.all_sizes,
            _qkvgbfa_bs_limit=256,
            _qkvgbfa_layer=None,
            _bfa_f_b_w=self.f_b_w,
            _bfa_alt_stream=None,
            fused_qkvg_proj=SimpleNamespace(
                quant_method=SimpleNamespace(
                    apply=lambda _layer, _hidden_states, _bias: fused_states
                )
            ),
        )

        _, _, deferred_f_a, _ = KimiK3DeltaAttention.forward_qkvbfg_fused(
            fake_attention, x, defer_f_b=True
        )
        self.assertEqual(deferred_f_a.shape[-1], HEAD_DIM)
        self.assertTrue(torch.equal(deferred_f_a, expected_f_a))

        expected_gate = kimi_k3_tiny_gemm(expected_f_a, self.f_b_w)
        _, _, eager_gate, _ = KimiK3DeltaAttention.forward_qkvbfg_fused(
            fake_attention, x, defer_f_b=False
        )
        torch.testing.assert_close(eager_gate, expected_gate)


if __name__ == "__main__":
    unittest.main()

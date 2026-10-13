"""Correctness checks for the ROCm Kimi-K3 fused KDA input projection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    _merge_weights_as_views,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

HIDDEN = 7168
HEADS_TP = 12
HEAD_DIM = 128
PROJ_TP = HEADS_TP * HEAD_DIM
WIDE = 4 * PROJ_TP
MERGED = 6288
TOL = 8e-3


class _FakeLinear(torch.nn.Module):
    def __init__(self, rows: int, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02,
            requires_grad=False,
        )


class _FakeQuantMethod:
    def __init__(self, output: torch.Tensor):
        self.output = output

    def apply(self, layer, hidden_states, bias):
        del hidden_states, bias
        assert layer.weight.shape[0] == MERGED
        return self.output


class _FakeWideProjection:
    def __init__(self, output: torch.Tensor):
        self.quant_method = _FakeQuantMethod(output)


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3KDAInProjFusion(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        device = torch.device("cuda", 0)
        cls.qkvg = _FakeLinear(WIDE, device)
        cls.f_a = _FakeLinear(HEAD_DIM, device)
        cls.b = _FakeLinear(HEADS_TP, device)
        cls.original = [
            module.weight.data.clone() for module in (cls.qkvg, cls.f_a, cls.b)
        ]
        cls.f_b_weight = (
            torch.randn(PROJ_TP, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.02
        )
        cls.merged, cls.sizes = _merge_weights_as_views(
            [cls.qkvg, cls.f_a, cls.b], pad_rows_to=8
        )
        cls.split_sizes = [3 * PROJ_TP, PROJ_TP]
        cls.all_sizes = [
            *cls.split_sizes,
            HEAD_DIM,
            HEADS_TP,
            MERGED - WIDE - HEAD_DIM - HEADS_TP,
        ]

    def test_merged_layout_and_views(self):
        self.assertEqual(self.sizes, [WIDE, HEAD_DIM, HEADS_TP])
        self.assertEqual(tuple(self.merged.shape), (MERGED, HIDDEN))
        self.assertEqual(self.qkvg.weight.data_ptr(), self.merged.data_ptr())
        self.assertEqual(self.f_a.weight.data_ptr(), self.merged[WIDE:].data_ptr())
        for module, original in zip(
            (self.qkvg, self.f_a, self.b), self.original, strict=True
        ):
            self.assertTrue(torch.equal(module.weight.data, original))

    def test_split_and_fused_paths_agree(self):
        tail = self.merged[WIDE:]
        for tokens in (1, 4, 8, 33, 128, 256):
            with self.subTest(tokens=tokens):
                x = torch.randn(
                    tokens, HIDDEN, dtype=torch.bfloat16, device=self.merged.device
                )
                wide = torch.nn.functional.linear(x, self.qkvg.weight)
                split_qkv, split_g = torch.split(wide, self.split_sizes, dim=-1)
                split_bfa = torch.nn.functional.linear(x, tail)
                split_fa = split_bfa[..., :HEAD_DIM]
                split_beta = split_bfa[..., HEAD_DIM : HEAD_DIM + HEADS_TP]
                split_fg = torch.nn.functional.linear(split_fa, self.f_b_weight)

                fused = torch.nn.functional.linear(x, self.merged)
                fused_qkv, fused_g, fused_fa, fused_beta, _padding = torch.split(
                    fused, self.all_sizes, dim=-1
                )
                fused_fg = torch.nn.functional.linear(fused_fa, self.f_b_weight)

                for name, actual, expected in (
                    ("qkv", fused_qkv, split_qkv),
                    ("g", fused_g, split_g),
                    ("f_a", fused_fa, split_fa),
                    ("beta", fused_beta, split_beta),
                    ("forget_gate", fused_fg, split_fg),
                ):
                    scale = expected.float().abs().max().clamp_min(1e-6)
                    rel = (
                        (actual.float() - expected.float()).abs().max() / scale
                    ).item()
                    self.assertLess(rel, TOL, f"{name} relative error {rel:.2e}")

    def test_deferred_fb_returns_fa_without_tiny_gemm(self):
        tokens = 8
        fused = torch.randn(
            tokens, MERGED, dtype=torch.bfloat16, device=self.merged.device
        )
        attention = KimiK3DeltaAttention.__new__(KimiK3DeltaAttention)
        torch.nn.Module.__init__(attention)
        attention.use_full_rank_gate = True
        attention._kda_group64_weight = None
        attention._kda_group64_scale = None
        attention._bfa_w = self.merged[WIDE:]
        attention._bfa_fa_size = HEAD_DIM
        attention._bfa_b_size = HEADS_TP
        attention._bfa_alt_stream = None
        attention._qkvgbfa_sizes = self.all_sizes
        attention._qkvgbfa_bs_limit = 256
        attention._qkvgbfa_layer = SimpleNamespace(weight=self.merged)
        attention.fused_qkvg_proj = _FakeWideProjection(fused)
        attention.f_b_proj = SimpleNamespace(weight=self.f_b_weight)

        def unexpected_gemm(*args, **kwargs):
            raise AssertionError("deferred f_b path must not launch tiny GEMM")

        with patch("sglang.kernels.ops.gemm.kimi_k3_tiny_gemm", unexpected_gemm):
            qkv, beta, f_a, g = attention.forward_qkvbfg_fused(
                torch.empty(
                    tokens,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device=self.merged.device,
                ),
                defer_f_b=True,
            )

        expected_qkv, expected_g, expected_fa, expected_beta, _padding = torch.split(
            fused, self.all_sizes, dim=-1
        )
        self.assertTrue(torch.equal(qkv, expected_qkv))
        self.assertTrue(torch.equal(g, expected_g))
        self.assertTrue(torch.equal(f_a, expected_fa))
        self.assertTrue(torch.equal(beta, expected_beta))


if __name__ == "__main__":
    unittest.main()

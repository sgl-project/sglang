"""CPU unit tests for zero-token DP rank handling in Qwen4-Exp norm/gate paths."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm as grouped_gemma_mod
from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpAttentionDecoderLayer,
    Qwen4ExpPLEGroupedNorm,
)
from sglang.test.test_utils import CustomTestCase


class TestQwen4ExpEmptyDPRank(CustomTestCase):
    """Validate 0-token batch behavior for idle DP ranks."""

    def test_qwen4_exp_ple_grouped_norm_empty_tokens(self):
        norm_single = Qwen4ExpPLEGroupedNorm(hidden_size=2560, group_size=2560)
        for empty_shape in ((0, 2560), (0, 1, 2560)):
            x = torch.empty(empty_shape, dtype=torch.bfloat16)
            out = norm_single(x)
            self.assertEqual(out.shape, empty_shape)
            self.assertEqual(out.dtype, torch.bfloat16)

        norm_multi = Qwen4ExpPLEGroupedNorm(hidden_size=2560, group_size=512)
        x_empty_multi = torch.empty((0, 2560), dtype=torch.float32)
        out_empty_multi = norm_multi(x_empty_multi)
        self.assertEqual(out_empty_multi.shape, (0, 2560))
        self.assertEqual(out_empty_multi.dtype, torch.float32)

        # Verify explicit group count matches -1 semantics on non-empty tokens.
        x_nonempty = torch.randn((3, 2560), dtype=torch.bfloat16)
        out_nonempty = norm_multi(x_nonempty)
        x_f = x_nonempty.float().reshape(3, 5, 512)
        var = (
            x_f.pow(2).mean(dim=-1, keepdim=True).expand_as(x_f).reshape_as(x_nonempty)
        )
        expected = (
            (x_nonempty.float() * torch.rsqrt(var + norm_multi.eps))
            * (norm_multi.weight.float() + 1.0)
        ).to(torch.bfloat16)
        torch.testing.assert_close(out_nonempty, expected)

    def test_qwen4_exp_attention_gate_empty_tokens(self):
        layer = Qwen4ExpAttentionDecoderLayer.__new__(Qwen4ExpAttentionDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.is_qsa = False
        layer.alt_stream = None

        def _fake_qkv_gate(positions, hidden_states, forward_batch):
            rows = hidden_states.shape[0]
            q = torch.empty((rows, 4, 8), dtype=hidden_states.dtype)
            k = torch.empty((rows, 4, 8), dtype=hidden_states.dtype)
            v = torch.empty((rows, 4, 8), dtype=hidden_states.dtype)
            gate = torch.zeros((rows, 4, 8), dtype=hidden_states.dtype)
            return q, k, v, gate

        layer._prepare_qkv_gate = _fake_qkv_gate
        layer.attn = lambda q, k, v, fb, **kw: torch.ones(
            (q.shape[0], 32), dtype=q.dtype
        )
        layer.o_proj = lambda x: (x, None)

        empty_hidden = torch.empty((0, 32), dtype=torch.bfloat16)
        empty_pos = torch.empty((0,), dtype=torch.long)
        out_empty = layer.self_attention(empty_pos, empty_hidden, forward_batch=None)
        self.assertEqual(out_empty.shape, (0, 32))
        self.assertEqual(out_empty.dtype, torch.bfloat16)

        nonempty_hidden = torch.ones((2, 32), dtype=torch.bfloat16)
        nonempty_pos = torch.arange(2, dtype=torch.long)
        out_nonempty = layer.self_attention(
            nonempty_pos, nonempty_hidden, forward_batch=None
        )
        self.assertEqual(out_nonempty.shape, (2, 32))
        torch.testing.assert_close(
            out_nonempty, torch.full((2, 32), 0.5, dtype=torch.bfloat16)
        )

    def test_grouped_gemma_rmsnorm_empty_skips_jit_and_validates(self):
        weight = torch.zeros(512, dtype=torch.bfloat16)
        with patch.object(
            grouped_gemma_mod,
            "_jit_grouped_gemma_rmsnorm_module",
            side_effect=AssertionError(
                "JIT module must not be invoked on 0-token input"
            ),
        ):
            x_empty = torch.empty((0, 512), dtype=torch.bfloat16)
            out = grouped_gemma_mod.grouped_gemma_rmsnorm(
                x_empty, weight, group_size=512
            )
            self.assertEqual(out.shape, (0, 512))
            self.assertEqual(out.dtype, torch.bfloat16)

            prealloc = torch.empty((0, 512), dtype=torch.bfloat16)
            out_prealloc = grouped_gemma_mod.grouped_gemma_rmsnorm(
                x_empty, weight, group_size=512, out=prealloc
            )
            self.assertEqual(out_prealloc.shape, (0, 512))
            self.assertEqual(out_prealloc.data_ptr(), prealloc.data_ptr())

            x_3d = torch.empty((0, 2, 512), dtype=torch.float16)
            out_3d = grouped_gemma_mod.grouped_gemma_rmsnorm(
                x_3d, torch.zeros(512, dtype=torch.float16), group_size=512
            )
            self.assertEqual(out_3d.shape, (0, 2, 512))
            self.assertEqual(out_3d.dtype, torch.float16)

            with self.assertRaises(RuntimeError):
                grouped_gemma_mod.grouped_gemma_rmsnorm(
                    torch.empty((0, 512), dtype=torch.float32),
                    torch.zeros(512, dtype=torch.float32),
                    group_size=512,
                )
            with self.assertRaises(RuntimeError):
                grouped_gemma_mod.grouped_gemma_rmsnorm(x_empty, weight, group_size=256)
            with self.assertRaises(ValueError):
                grouped_gemma_mod.grouped_gemma_rmsnorm(
                    x_empty,
                    weight,
                    group_size=512,
                    out=torch.empty((1, 512), dtype=torch.bfloat16),
                )

    def test_hyperconnection_empty_tokens(self):
        norm = GroupedGemmaRMSNorm(hidden_size=64, group_size=64)
        x = torch.empty((0, 4, 64), dtype=torch.bfloat16)
        out_norm = norm(x)
        self.assertEqual(out_norm.shape, (0, 4, 64))

        cfg = HyperConnectionConfig(hc_count=4, hidden_size=64, hc_per_branch_norm=True)
        gated = GatedResidual(cfg)
        h_in = torch.empty((0, 4 * 64), dtype=gated.params_dtype)
        mixed, residuals = gated.mix(h_in)
        self.assertEqual(mixed.shape, (0, 64))
        combined = gated.combine(
            torch.empty((0, 64), dtype=gated.params_dtype), residuals
        )
        self.assertEqual(combined.shape, (0, 4 * 64))


if __name__ == "__main__":
    unittest.main()

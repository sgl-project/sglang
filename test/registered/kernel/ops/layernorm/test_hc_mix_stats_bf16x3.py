import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.layernorm.hc_mix_stats_bf16x3 import (
    hc_mix_stats_sinkhorn_bf16x3,
    split_bf16_hc_weight,
)
from sglang.kernels.ops.layernorm.hc_mix_stats_deepgemm import (
    hc_mix_stats_sinkhorn_deepgemm,
    split_tf32_hc_weight,
)
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
EPS = 1e-6


def reference(x, weight, scale, base):
    x, weight, scale, base = [v.double() for v in (x, weight, scale, base)]
    mixes = (x @ weight.T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + EPS
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = torch.softmax((mixes[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4), -1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    return pre, post, comb


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "Compensated prefill projection targets Blackwell",
)
class TestHcMixStatsBf16x3(CustomTestCase):
    def _inputs(self, rows, seed):
        torch.manual_seed(seed)
        x = torch.randn(rows, 20480, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(24, 20480, device="cuda") * 0.02
        scale = torch.tensor([0.1, 0.2, 0.3], device="cuda")
        base = torch.randn(24, device="cuda") * 0.2
        return x, w, scale, base

    def test_prefill_coefficients(self):
        for rows in (4096, 4097, 16384, 65536):
            for seed in (0, 42):
                with self.subTest(rows=rows, seed=seed):
                    x, w, scale, base = self._inputs(rows, seed)
                    parts = split_bf16_hc_weight(w)
                    torch.testing.assert_close(
                        sum(p.float() for p in parts), w, rtol=2e-7, atol=0
                    )
                    actual = hc_mix_stats_sinkhorn_bf16x3(
                        x, parts, scale, base, 20, EPS, EPS
                    )
                    old = hc_mix_stats_sinkhorn_deepgemm(
                        x, split_tf32_hc_weight(w), scale, base, 20, EPS, EPS
                    )
                    # All rows against the existing compensated implementation.
                    for a, b in zip(actual, old):
                        self.assertTrue(torch.isfinite(a).all().item())
                        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                    # Independent FP64 reference, including a masked final tile.
                    indices = torch.cat(
                        (
                            torch.arange(32, device="cuda"),
                            torch.arange(rows - 32, rows, device="cuda"),
                        )
                    )
                    expected = reference(x[indices], w, scale, base)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(
                            a[indices].double(), b, rtol=2e-5, atol=2e-6
                        )

    def test_graph_replay(self):
        x, w, scale, base = self._inputs(4097, 13)
        parts = split_bf16_hc_weight(w)
        hc_mix_stats_sinkhorn_bf16x3(x, parts, scale, base, 20, EPS, EPS)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = hc_mix_stats_sinkhorn_bf16x3(x, parts, scale, base, 20, EPS, EPS)
        x.normal_()
        graph.replay()
        expected = reference(x[-32:], w, scale, base)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a[-32:].double(), b, rtol=2e-5, atol=2e-6)

    def test_model_dispatch_and_weight_refresh(self):
        x, w, scale, base = self._inputs(4096, 17)
        layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.config = SimpleNamespace(model_type="deepseek_v41")
        layer.hc_attn_fn = torch.nn.Parameter(w)
        layer.hc_ffn_fn = torch.nn.Parameter(w.clone())
        layer.hc_pre_from_prev_sublayer = True
        layer.hc_mult, layer.hc_sinkhorn_iters = 4, 20
        layer.rms_norm_eps = layer.hc_eps = EPS
        layer.input_layernorm = torch.nn.LayerNorm(5120, device="cuda")
        layer.post_attention_layernorm = torch.nn.LayerNorm(5120, device="cuda")
        with (
            envs.SGLANG_DSV41_COMPENSATED_MHC.override(True),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                True,
            ),
        ):
            layer.refresh_mhc_norm_weight_cache()
            previous = layer._hc_attn_bf16_parts
            with torch.no_grad():
                layer.hc_attn_fn.add_(0.1)
            layer.refresh_mhc_norm_weight_cache()
            self.assertFalse(torch.equal(previous[0], layer._hc_attn_bf16_parts[0]))
            torch.testing.assert_close(
                sum(p.float() for p in layer._hc_attn_bf16_parts),
                layer.hc_attn_fn,
                rtol=2e-7,
                atol=0,
            )
            target = "sglang.kernels.ops.layernorm.hc_mix_stats_bf16x3.hc_mix_stats_sinkhorn_bf16x3"
            for rows, invariant, expected in [
                (384, False, False),
                (4096, True, False),
                (4096, False, True),
            ]:
                with self.subTest(rows=rows, invariant=invariant), patch(
                    "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                    return_value=invariant,
                ), patch(target, wraps=hc_mix_stats_sinkhorn_bf16x3) as fast:
                    layer._hc_mix_and_combine(
                        x[:rows].view(rows, 4, 5120),
                        layer.hc_attn_fn,
                        scale,
                        base,
                        None,
                        lambda v: v,
                    )
                    self.assertEqual(fast.called, expected)
        with envs.SGLANG_DSV41_COMPENSATED_MHC.override(False):
            layer.refresh_mhc_norm_weight_cache()
        self.assertIsNone(layer._hc_attn_bf16_parts)
        self.assertIsNone(layer._hc_ffn_bf16_parts)


if __name__ == "__main__":
    unittest.main()

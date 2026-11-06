"""Numerics for the CPU fp8 weight-only compressed-tensors schemes
(CompressedTensorsW8A16Fp8, CompressedTensorsW8A16Fp8MoE): real scheme path
(process_weights_after_loading + apply_weights, which dispatch to
fp8_scaled_mm_cpu / fused_experts_cpu) vs a per-channel dequant reference.

Regression coverage: apply_weights previously (a) upcast to float32 via
plain `weight.to(x.dtype) * weight_scale` (dtype promotion bug), and (b)
crashed on a bf16 `bias` because fp8_scaled_mm_cpu requires float32 bias.
"""

import unittest
from types import SimpleNamespace

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A16Fp8,
    CompressedTensorsW8A16Fp8MoE,
)
from sglang.srt.utils import cpu_has_amx_support, is_cpu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import SiluAndMul, parametrize, precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-intel")


class TestCompressedTensorsW8A16Fp8MoEStrategy(CustomTestCase):
    def test_unsupported_weight_strategies_raise_not_implemented(self):
        for strategy in (QuantizationStrategy.BLOCK, QuantizationStrategy.TENSOR):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(
                    NotImplementedError, f"CHANNEL.*{strategy}"
                ):
                    CompressedTensorsW8A16Fp8MoE(
                        weight_quant=SimpleNamespace(strategy=strategy),
                        input_quant=None,
                    )


def _quantize_fp8_channelwise(w: torch.Tensor):
    """Per-output-channel symmetric fp8_e4m3; returns (w_fp8, scale fp32
    with a trailing dim of 1, dequant) all with a channel dim of `w.size(-2)`."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = w.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / fp8_max
    w_fp8 = (w.float() / scale).to(torch.float8_e4m3fn)
    w_dequant = w_fp8.float() * scale
    return w_fp8, scale, w_dequant


@unittest.skipUnless(is_cpu() and cpu_has_amx_support(), "requires an Intel AMX CPU")
class TestCompressedTensorsW8A16Fp8(CustomTestCase):
    @parametrize(
        input_layout=[
            ((1,), False),
            ((11,), False),
            ((5, 1), False),
            ((2, 3), False),
            ((2, 3, 4), False),
            ((2, 3), True),
        ],
        N=[32, 96],
        K=[64, 128],
        has_bias=[False, True],
    )
    def test_apply_weights(self, input_layout, N, K, has_bias):
        """CPU FP8 linear preserves leading dimensions, including strided batches."""
        torch.manual_seed(0)
        batch_shape, transpose_input = input_layout
        x = torch.randn(*batch_shape, K, dtype=torch.bfloat16) / 10
        if transpose_input:
            x = x.transpose(0, 1)
        w = torch.randn(N, K) / 10
        w_fp8, scale, w_dequant = _quantize_fp8_channelwise(w)
        bias = torch.randn(N, dtype=torch.bfloat16) / 10 if has_bias else None

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        if has_bias:
            layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)

        scheme = CompressedTensorsW8A16Fp8(
            strategy=QuantizationStrategy.CHANNEL, is_static_input_scheme=False
        )
        scheme.process_weights_after_loading(layer)
        out = scheme.apply_weights(layer, x, bias=layer.bias if has_bias else None)

        ref = x.float() @ w_dequant.T
        if has_bias:
            ref = ref + bias.float()
        ref = ref.to(x.dtype)

        atol = rtol = precision[x.dtype]
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@unittest.skipUnless(is_cpu() and cpu_has_amx_support(), "requires an Intel AMX CPU")
class TestCompressedTensorsW8A16Fp8MoE(CustomTestCase):
    @parametrize(M=[2, 5], topk=[1, 2])
    def test_apply_weights(self, M, topk):
        torch.manual_seed(0)
        E, K, N = 4, 64, 32

        x = torch.randn(M, K, dtype=torch.bfloat16) / 10
        w13 = torch.randn(E, 2 * N, K) / 10
        w2 = torch.randn(E, K, N) / 10
        w13_fp8, w13_scale, w13_dequant = _quantize_fp8_channelwise(w13)
        w2_fp8, w2_scale, w2_dequant = _quantize_fp8_channelwise(w2)

        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(w13_fp8, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_fp8, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        scheme = CompressedTensorsW8A16Fp8MoE(
            weight_quant=SimpleNamespace(strategy=QuantizationStrategy.CHANNEL),
            input_quant=None,
        )
        scheme.process_weights_after_loading(layer)
        scheme.create_moe_runner(layer, MoeRunnerConfig(activation="silu"))

        topk_ids = torch.randint(0, E, (M, topk))
        topk_weights = torch.rand(M, topk)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        dispatch_output = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(topk_weights, topk_ids, torch.zeros(M, E)),
        )

        combine_input = scheme.apply_weights(layer, dispatch_output)
        out = combine_input.hidden_states

        ref = torch.zeros(M, K, dtype=torch.float32)
        for m in range(M):
            for t in range(topk):
                e = topk_ids[m, t].item()
                gate_up = x[m].float() @ w13_dequant[e].T
                act = SiluAndMul(gate_up)
                ref[m] += topk_weights[m, t].item() * (act @ w2_dequant[e].T)
        ref = ref.to(x.dtype)

        atol = rtol = precision[x.dtype]
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()

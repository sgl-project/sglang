"""MXFP8 soft-clamped squared-ReLU experts on FlashInfer's CuTe-DSL MoE, with
the scales prepared by Mxfp8CuteDslMoEMethod, vs a dequantized reference."""

import types
import unittest

import torch
from torch.nn import Parameter

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-b", runner_config="4-gpu-b200")

E, TOP_K, HIDDEN, INTER, CLAMP, BLOCK = 16, 4, 768, 1024, 16.0, 32


def _supported() -> bool:
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
        return False
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        cutedsl_supports_clamped_relu2,
    )

    return cutedsl_supports_clamped_relu2()


def _mxfp8_weight(experts, rows, cols):
    """E4M3 values with row-major UE8M0 block-32 scales, and their dequantization."""
    values = (torch.randn(experts, rows, cols, device="cuda") * 2).to(
        torch.float8_e4m3fn
    )
    exponents = torch.randint(
        118, 124, (experts, rows, cols // BLOCK), device="cuda", dtype=torch.uint8
    )
    scale = torch.exp2(exponents.float() - 127.0).repeat_interleave(BLOCK, dim=-1)
    return values, exponents, values.float() * scale


def _qdq(x):
    """MXFP8 quantize-dequantize of activations, as the kernel sees them."""
    blocks = x.float().reshape(*x.shape[:-1], -1, BLOCK)
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(x.shape)


@unittest.skipUnless(_supported(), "needs SM100 and FlashInfer >= 0.7.4 (ClampedRelu2)")
class TestMxfp8CuteDslExperts(CustomTestCase):
    def test_matches_dequantized_reference(self):
        from flashinfer import ActivationType, CuteDslMoEWrapper, mxfp8_quantize

        from sglang.srt.layers.quantization.mxfp8_cutedsl_moe import (
            Mxfp8CuteDslMoEMethod,
        )

        torch.manual_seed(0)
        w1, s1, w1_ref = _mxfp8_weight(E, INTER, HIDDEN)
        w2, s2, w2_ref = _mxfp8_weight(E, HIDDEN, INTER)
        layer = types.SimpleNamespace(
            w13_weight=Parameter(w1, requires_grad=False),
            w13_weight_scale_inv=Parameter(s1, requires_grad=False),
            w2_weight=Parameter(w2, requires_grad=False),
            w2_weight_scale_inv=Parameter(s2, requires_grad=False),
        )
        method = Mxfp8CuteDslMoEMethod.__new__(Mxfp8CuteDslMoEMethod)
        method.process_weights_after_loading(layer)

        ones = torch.ones(E, dtype=torch.float32, device="cuda")
        wrapper = CuteDslMoEWrapper(
            num_experts=E,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            use_cuda_graph=False,
            max_num_tokens=256,
            num_local_experts=E,
            local_expert_offset=0,
            output_dtype=torch.bfloat16,
            device="cuda",
            activation_type=ActivationType.ClampedRelu2,
            quant_mode="mxfp8",
            swap_ab=torch.cuda.get_device_capability() in ((10, 0), (10, 3)),
            clamped_relu2_limit=torch.tensor([CLAMP], device="cuda"),
        )
        for tokens, x_scale in ((1, 1.0), (37, 1.0), (128, 4.0)):
            x = (torch.randn(tokens, HIDDEN, device="cuda") * x_scale).to(
                torch.bfloat16
            )
            ids = torch.stack(
                [torch.randperm(E, device="cuda")[:TOP_K] for _ in range(tokens)]
            ).to(torch.int32)
            weights = torch.softmax(torch.randn(tokens, TOP_K, device="cuda"), -1)
            xq, x_sf = mxfp8_quantize(x, is_sf_swizzled_layout=False)
            out = wrapper.run(
                x=xq,
                x_sf=x_sf.view(torch.uint8).reshape(tokens, HIDDEN // BLOCK),
                token_selected_experts=ids,
                token_final_scales=weights,
                w1_weight=layer.w13_weight,
                w1_weight_sf=layer.w13_weight_scale_inv,
                w1_alpha=ones,
                fc2_input_scale=None,
                w2_weight=layer.w2_weight,
                w2_weight_sf=layer.w2_weight_scale_inv,
                w2_alpha=ones,
            ).float()

            ref = torch.zeros(tokens, HIDDEN, device="cuda")
            xr = _qdq(x)
            for e in ids.unique().tolist():
                tok, slot = (ids == e).nonzero(as_tuple=True)
                z = xr[tok] @ w1_ref[e].T
                h = torch.relu(CLAMP * torch.tanh(z / CLAMP)) ** 2
                ref.index_add_(
                    0, tok, (_qdq(h) @ w2_ref[e].T) * weights[tok, slot, None]
                )
            rel = ((out - ref).norm() / ref.norm()).item()
            self.assertLess(rel, 3e-2, f"{tokens} tokens, x scale {x_scale}")


if __name__ == "__main__":
    unittest.main()

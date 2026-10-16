"""Small deterministic quantized experts for Marlin/DeepEP correctness tests."""

from types import SimpleNamespace

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
    prepare_moe_nvfp4_layer_for_marlin,
)
from sglang.test.test_marlin_utils import awq_marlin_quantize, marlin_quantize


def make_experts(
    format, experts=4, hidden=256, intermediate=128, *, gated=True, bias=False
):
    """Return the kernel payload and independent dequantized matrices/biases."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    sizes = [(intermediate * (2 if gated else 1), hidden), (hidden, intermediate)]
    refs, payload = [], []
    biases = [
        torch.randn(experts, n, device="cuda", dtype=dtype) / 20 if bias else None
        for n, _ in sizes
    ]
    if format in {"gptq4", "gptq8", "awq"}:
        quant_type = {
            "gptq4": scalar_types.uint4b8,
            "gptq8": scalar_types.uint8b128,
            "awq": scalar_types.uint4,
        }[format]
        for n, k in sizes:
            columns = []
            for _ in range(experts):
                w = torch.randn(k, n, device="cuda", dtype=dtype) / 20
                if format == "awq":
                    ref, q, scale, zero = awq_marlin_quantize(w, quant_type, 128)
                    values = (ref.T, q, scale, zero, None, None)
                else:
                    ref, q, scale, g_idx, sort, _ = marlin_quantize(
                        w, quant_type, 128, k > 128, torch.randperm(k)
                    )
                    values = (ref.T, q, scale, None, g_idx, sort)
                columns.append(values)
            stacked = [
                torch.stack(v).contiguous() if v[0] is not None else None
                for v in zip(*columns)
            ]
            refs.append(stacked[0])
            payload.append(stacked[1:])
        w13, scale13, zeros13, g_idx13, sort13 = payload[0]
        w2, scale2, zeros2, g_idx2, sort2 = payload[1]
        info = MarlinMoeQuantInfo(
            w13_qweight=w13,
            w2_qweight=w2,
            w13_scales=scale13,
            w2_scales=scale2,
            w13_g_idx_sort_indices=sort13,
            w2_g_idx_sort_indices=sort2,
            weight_bits=8 if format == "gptq8" else 4,
            w13_qzeros=zeros13,
            w2_qzeros=zeros2,
            w13_g_idx=g_idx13,
            w2_g_idx=g_idx2,
        )
        if bias:
            from sglang.srt.layers.quantization.marlin_utils import marlin_permute_bias

            info.w13_bias, info.w2_bias = [
                torch.stack([marlin_permute_bias(row) for row in v]) for v in biases
            ]
    else:
        layer = torch.nn.Module()
        layer.params_dtype = dtype
        layer.quant_config = SimpleNamespace(group_size=16)
        layer.moe_runner_config = SimpleNamespace(is_gated=gated)
        layer.intermediate_size_per_partition = intermediate
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
            dtype=dtype,
        )
        for i, (prefix, (n, k)) in enumerate(zip(("w13", "w2"), sizes)):
            codes = torch.randint(
                0, 16, (experts, n, k), device="cuda", dtype=torch.uint8
            )
            packed = codes[..., ::2] | (codes[..., 1::2] << 4)
            group = 32 if format == "mxfp4" else 16
            # Power-of-two scales are exactly representable in both formats.
            # Match the roughly 0.05 RMS weights used by the integer fixtures.
            exponents = torch.randint(-7, -4, (experts, n, k // group), device="cuda")
            scale = (2.0**exponents).to(dtype)
            ref = values[codes.long()] * scale.repeat_interleave(group, -1)
            refs.append(ref)
            if format == "mxfp4":
                scale = (exponents + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
            else:
                # NVFP4 checkpoints normalize group scales into the FP8
                # range and carry the small outer scale separately.
                scale = (scale * 4096).to(torch.float8_e4m3fn)
                setattr(
                    layer,
                    prefix + "_weight_scale_2",
                    torch.full((experts,), 1 / 4096, device="cuda", dtype=dtype),
                )
            setattr(
                layer,
                prefix + "_weight",
                torch.nn.Parameter(packed, requires_grad=False),
            )
            setattr(
                layer,
                prefix + "_weight_scale",
                torch.nn.Parameter(scale, requires_grad=False),
            )
            if bias:
                setattr(
                    layer,
                    prefix + ("_weight_bias" if format == "mxfp4" else "_bias"),
                    biases[i],
                )
        prepare = (
            prepare_moe_mxfp4_layer_for_marlin
            if format == "mxfp4"
            else prepare_moe_nvfp4_layer_for_marlin
        )
        prepare(layer)
        info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            w13_global_scale=getattr(layer, "w13_weight_scale_2", None),
            w2_global_scale=getattr(layer, "w2_weight_scale_2", None),
            w13_bias=getattr(
                layer, "w13_weight_bias" if format == "mxfp4" else "w13_bias", None
            ),
            w2_bias=getattr(
                layer, "w2_weight_bias" if format == "mxfp4" else "w2_bias", None
            ),
        )
    return info, (*refs, *biases)


def reference(hidden, ids, weights, matrices, config):
    w1, w2, b1, b2 = matrices
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        gate_up = hidden.float() @ w1[expert].float().T
        if b1 is not None:
            gate_up += b1[expert].float()
        # The two GEMMs store BF16 activations. Keep FP32 accumulation in
        # the reference, then round at the same mathematical boundaries.
        gate_up = gate_up.to(hidden.dtype).float()
        if config.is_gated:
            gate, up = gate_up.chunk(2, dim=-1)
            if config.activation == "situ":
                beta = config.gemm1_alpha or 4.0
                activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * up
            else:
                activated = torch.nn.functional.silu(gate) * up
        elif config.activation == "relu2":
            activated = torch.relu(gate_up).square()
        else:
            activated = torch.nn.functional.silu(gate_up)
        result = activated.to(hidden.dtype).float() @ w2[expert].float().T
        if b2 is not None:
            result += b2[expert].float()
        for slot in range(ids.shape[1]):
            route_weight = torch.where(ids[:, slot] == expert, weights[:, slot], 0)
            output += (result * route_weight[:, None]).to(hidden.dtype).float()
    # MXFP4's model folds scaling into top-k, matching its existing kernel.
    if config.routed_scaling_factor is not None:
        output *= config.routed_scaling_factor
    return output.to(hidden.dtype)


def reference_tolerances(format):
    # Match the existing fused-Marlin dequantized-reference tests, including
    # NVFP4's extra scale-rounding error (test_moe_wna16_marlin.py).
    return (
        {"rtol": 0.05, "atol": 0.25}
        if format == "nvfp4"
        else {"rtol": 0.04, "atol": 0.04}
    )


def assert_reference_close(output, expected, format):
    torch.testing.assert_close(output, expected, **reference_tolerances(format))
    error = (output.float() - expected.float()).norm()
    relative_error = (error / expected.float().norm().clamp_min(1e-12)).item()
    assert relative_error < 0.02, f"Relative L2 error {relative_error:.6f} exceeds 2%"
    return relative_error

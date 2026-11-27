"""Shared full-K DeepGEMM FP8 projection for trainer and serving.

BF16 inputs use the shared block128 activation quantizer; weights carry E4M3
payloads and packed UE8M0 scales. Callers must retain the full input dimension
(TP1 row projections). K-sharded reduction and backward are not supported.
"""

import torch

from sglang.srt.layers import deep_gemm_wrapper as dg
from sglang.srt.true_on_policy.fp8.fp8_expert_forward import quantize_rows


def linear(x, weight, weight_scale, bias=None, *, cache_scales=False):
    """``cache_scales``: ``weight_scale`` is a serving parameter (block_scales)."""
    if any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in (x, weight, weight_scale, bias)
    ):
        raise NotImplementedError("FP8 TOP linear backward is not implemented")
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.ndim < 1
        or weight.dtype != torch.float8_e4m3fn
        or weight_scale.dtype != torch.int32
        or weight.ndim != 2
        or weight_scale.ndim != 2
        or x.shape[-1] != weight.shape[-1]
        or weight.shape[-1] < 128
        or weight.shape[-1] % 128
        or weight.shape[0] % 64
        or not weight.is_contiguous()
        or weight_scale.shape[0] != weight.shape[0]
        or not x.is_cuda
        or x.device != weight.device
        or x.device != weight_scale.device
        or weight_scale.shape[1] < (weight.shape[1] // 128 + 3) // 4
    ):
        raise NotImplementedError(
            "TOP FP8 linear requires BF16 input, contiguous E4M3 weights, "
            "N64/K128 alignment and expanded packed UE8M0 scales"
        )
    original = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )
    if x.shape[0]:
        _gemm(quantize_rows(x), weight, weight_scale, out, cache_scales=cache_scales)
    if bias is not None:
        out = out + bias
    return out.reshape(*original[:-1], weight.shape[0])


def _gemm(lhs, weight, weight_scale, out, cache_scales=False):
    from sglang.srt.true_on_policy.fp8 import uses_hopper_scales

    rhs = (weight, weight_scale)
    if uses_hopper_scales():
        from .hopper_scales import block_scales

        # quantize_rows already wrote FP32 activation scales on Hopper.
        # Recover checkpoint block scales, including a final partial
        # 128-row block for projections whose output size is only N64.
        rhs = (
            weight,
            block_scales(weight_scale, weight.shape[1] // 128, cached=cache_scales),
        )
    dg.gemm_nt_f8f8bf16(lhs, rhs, out)


def clamped_swiglu_mlp(x, gate_up, down, swiglu_limit, *, cache_scales=False):
    """Clamped SwiGLU MLP on the routed experts' FP32 SwiGLU-to-FP8 quantizer.

    ``gate_up`` and ``down`` are (E4M3 weight, packed UE8M0 scale) pairs with full K.
    ``cache_scales``: both scales are serving parameters (block_scales).
    """
    hidden = linear(x, *gate_up, cache_scales=cache_scales).reshape(
        -1, gate_up[0].shape[0]
    )
    out = torch.empty(
        (hidden.shape[0], down[0].shape[0]), device=x.device, dtype=torch.bfloat16
    )
    if hidden.shape[0]:
        _gemm(
            quantize_rows(hidden, fused_swiglu=True, swiglu_limit=swiglu_limit),
            *down,
            out,
            cache_scales=cache_scales,
        )
    return out.reshape(*x.shape[:-1], down[0].shape[0])

"""Shared full-K 32x32 block-FP8 projection for trainer and serving (DeepSeek-V4.1).

BF16 rows are quantized per 32 channels to E4M3 with power-of-two (UE8M0) scales, the checkpoint's activation rule;
weights are the checkpoint's E4M3 payloads with FP32 power-of-two 32x32 block scales. The GEMM is SM90 DeepGEMM's
BLOCK32 recipe over the full K, with no BF16 weight cache, MXFP8 view or Triton fallback. Callers keep the full
input dimension (TP1 row projections). Backward is the trainer's.
"""

import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers import deep_gemm_wrapper as dg

GROUP = 32
RECIPE = (1, GROUP, GROUP)


def linear(x, weight, weight_scale, bias=None):
    if any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in (x, weight, weight_scale, bias)
    ):
        raise NotImplementedError("FP8 32x32 TOP linear backward is not implemented")
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.ndim < 1
        or weight.dtype != torch.float8_e4m3fn
        or weight.ndim != 2
        or not weight.is_contiguous()
        or x.shape[-1] != weight.shape[-1]
        or weight.shape[-1] % GROUP
        # DeepGEMM's SM90 BLOCK32 kernel (fp8_utils._deepgemm_block_fp8_supported).
        or weight.shape[0] % 8
        or weight_scale.dtype != torch.float32
        or tuple(weight_scale.shape)
        != (-(-weight.shape[0] // GROUP), weight.shape[1] // GROUP)
        or not x.is_cuda
        or x.device != weight.device
        or x.device != weight_scale.device
    ):
        raise NotImplementedError(
            "TOP FP8 32x32 linear requires BF16 input, a contiguous E4M3 weight with N%8 and K%32 alignment, and "
            "FP32 32x32 block scales"
        )
    original = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )
    if x.shape[0]:
        dg.gemm_nt_f8f8bf16(
            quantize_rows(x), (weight, weight_scale.contiguous()), out, recipe=RECIPE
        )
    if bias is not None:
        out = out + bias
    return out.reshape(*original[:-1], weight.shape[0])


def quantize_rows(x):
    """Per-row, per-32-channel E4M3 with power-of-two scales: each row's bytes depend on that row alone."""
    return sglang_per_token_group_quant_fp8(x, GROUP, scale_ue8m0=True)


# DeepSeek-V4.1's swiglu_limit; the clamp is part of the program, not a parameter of it.
SWIGLU_LIMIT = 10.0


def clamped_swiglu_mlp(x, gate_up, down, swiglu_limit):
    """DeepSeek-V4.1 shared expert: 32x32 FP8 gate_up GEMM, clamped SwiGLU in FP32 fused into the per-32 UE8M0 FP8
    quantizer of the down projection's input (no BF16 rounding between), 32x32 FP8 down GEMM over the full K.

    ``gate_up`` and ``down`` are (E4M3 weight, FP32 32x32 block scale) pairs with full K.
    """
    from sglang.kernels.ops.activation.silu_and_mul_clamp import (
        silu_and_mul_contig_post_quant,
    )

    if swiglu_limit != SWIGLU_LIMIT:
        raise NotImplementedError(
            f"TOP FP8 32x32 clamped SwiGLU requires swiglu_limit {SWIGLU_LIMIT}, got {swiglu_limit}"
        )
    hidden = linear(x, *gate_up).reshape(-1, gate_up[0].shape[0])
    rows, inter = hidden.shape[0], hidden.shape[1] // 2
    out = torch.empty((rows, down[0].shape[0]), device=x.device, dtype=torch.bfloat16)
    if rows:
        act = torch.empty((rows, inter), device=x.device, dtype=torch.float8_e4m3fn)
        act_scale = torch.empty(
            (rows, inter // GROUP), device=x.device, dtype=torch.float32
        )
        silu_and_mul_contig_post_quant(
            input=hidden.contiguous(),
            output=act,
            output_scale=act_scale,
            quant_group_size=GROUP,
            scale_ue8m0=True,
            transposed=False,
            swiglu_limit=swiglu_limit,
        )
        weight, weight_scale = down
        if (
            weight.dtype != torch.float8_e4m3fn
            or weight.shape[1] != inter
            or weight_scale.dtype != torch.float32
        ):
            raise NotImplementedError(
                "TOP FP8 32x32 clamped SwiGLU requires an E4M3 down projection with FP32 32x32 scales over the full K"
            )
        dg.gemm_nt_f8f8bf16(
            (act, act_scale),
            (weight, weight_scale.contiguous()),
            out,
            recipe=RECIPE,
        )
    return out.reshape(*x.shape[:-1], down[0].shape[0])

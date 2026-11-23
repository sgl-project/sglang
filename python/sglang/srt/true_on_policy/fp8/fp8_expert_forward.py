"""Shared FP8 expert forward for serving and the trainer.

The caller owns row routing and passes checkpoint weights/scales already
converted to UE8M0. Both masked serving and packed trainer rows use fused FP32
SwiGLU-to-FP8 quantization: no intermediate BF16 rounding of the activation.
The expert input is quantize_expert_input's bytes: serving's DeepEP dispatch
sends them (quantized at the source in normal mode, by DeepEP's fused cast in
low-latency mode), the trainer quantizes its BF16 rows here.
Backward and DeepEP transport are outside this operator.
"""

import functools

import torch

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.kernels.ops.quantization.per_token_group_quant import (
    _allocate_outputs,
    _jit_module,
)
from sglang.srt.layers import deep_gemm_wrapper as dg
from sglang.srt.true_on_policy.fp8 import uses_hopper_scales

# (output width, fused SwiGLU, masked, FP32 scales, SwiGLU clamp, DeepEP amax floor) -> JIT quant module.
_QUANT_MODULES = {}


def quantize_expert_input(x, *, masked_m=None):
    """The routed experts' input quantization, which DeepEP's low-latency FP8 cast fixes in decode.

    That cast (round_scale; packed UE8M0 on Blackwell) is per row, 1x128, with the ceil-UE8M0 scale of
    amax * (1/448) like quantize_rows, but floors amax at DeepEP's 1e-4 margin, so every
    expert-input quantization takes quantize_rows with that floor.
    """
    return quantize_rows(x, masked_m=masked_m, deepep_amax_floor=True)


def quantize_rows(
    x, *, fused_swiglu=False, masked_m=None, swiglu_limit=None, deepep_amax_floor=False
):
    """Block-128 E4M3 rows with UE8M0 (power-of-two) scales in the GEMM's MN-major layout.

    ``swiglu_limit`` (fused SwiGLU only): DeepSeek-V4's clamp of the BF16 gate (max) and up (+-) halves,
    applied inside the quantizer with each clamped value rounded back to BF16 before the activation.
    ``deepep_amax_floor`` (plain rows only): amax floored at 1e-4 instead of 1e-10.

    Blackwell gets packed exponent bytes. Hopper gets the same scales as FP32
    values written by the quantizer itself, with hopper_scales.unpack_ue8m0's
    bits (exponent 0 is 2^-127, 255 is NaN).
    """
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("FP8 expert input must be contiguous BF16")
    fp32_scales = uses_hopper_scales()
    if not dg.DEEPGEMM_SCALE_UE8M0 and not fp32_scales:
        raise NotImplementedError("FP8 requires Blackwell UE8M0 DeepGEMM or Hopper")
    # Same JIT kernel and output layout as per_token_group_quant(group_size=128, scale_ue8m0=True,
    # column_major_scales=True); calling it directly skips the per-call custom-op dispatch.
    if fp32_scales:
        output_q = torch.empty(
            (*x.shape[:-1], x.shape[-1] // (2 if fused_swiglu else 1)),
            device=x.device,
            dtype=torch.float8_e4m3fn,
        )
        # DeepGEMM's FP32 TMA layout: [..., groups, rows aligned to 4], viewed [..., rows, groups].
        output_s = create_per_token_group_quant_fp8_output_scale(
            x_shape=output_q.shape,
            device=x.device,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
    else:
        output_q, output_s = _allocate_outputs(
            x, 128, torch.float8_e4m3fn, True, True, fused_swiglu
        )
    clamp = swiglu_limit is not None
    if clamp and not fused_swiglu:
        raise ValueError("swiglu_limit applies to the fused SwiGLU quant only")
    if deepep_amax_floor and fused_swiglu:
        raise ValueError("the DeepEP amax floor applies to the expert input only")
    key = (
        output_q.shape[-1],
        fused_swiglu,
        masked_m is not None,
        fp32_scales,
        clamp,
        deepep_amax_floor,
    )
    module = _QUANT_MODULES.get(key)
    if module is None:
        if fp32_scales:
            row_major, aligned = False, True
        else:
            # The layout _allocate_outputs was asked for (column-major, UE8M0-packed), not one read off the strides:
            # an empty first call (an attention-DP rank with no rows) has strides that read as row-major, and the
            # module it would cache fails every later call of this width.
            row_major, aligned = False, (key[0] // 128) % 4 == 0
        module = _jit_module(
            torch.bfloat16,
            torch.float8_e4m3fn,
            128,
            True,
            row_major,
            aligned,
            fused_swiglu,
            masked_m is not None,
            is_arch_support_pdl(),
            ue8m0_fp32=fp32_scales,
            clamp_swiglu=clamp,
            deepep_amax_floor=deepep_amax_floor,
        )
        _QUANT_MODULES[key] = module
    limit = (_bf16_limit(float(swiglu_limit)),) if clamp else ()
    if masked_m is not None:
        module.per_token_group_quant(x, output_q, output_s, masked_m, -1, *limit)
    else:
        module.per_token_group_quant(x, output_q, output_s, *limit)
    return output_q, output_s


@functools.lru_cache(maxsize=None)
def _bf16_limit(limit: float) -> float:
    """The limit rounded to BF16, the dtype the clamped gate/up values are compared and stored in."""
    return float(torch.tensor(limit, dtype=torch.bfloat16))


def _gemm_input_scales(scale):
    """Received activation scales in the layout quantize_expert_input writes for the GEMM; at most a pure copy.

    Normal dispatch delivers [rows, groups] (row-major FP32 on Hopper); low-latency dispatch delivers
    [experts, rows, groups] already MN-major, FP32 on Hopper and packed UE8M0 int32 on Blackwell.
    """
    if uses_hopper_scales():
        if scale.dtype != torch.float32:
            raise ValueError("Hopper FP8 expert input scales are FP32")
        if scale.dim() == 3:
            if scale.stride(-2) != 1 or scale.stride(-1) % 4:
                raise ValueError(
                    "Hopper masked FP8 input scales are TMA-aligned MN-major"
                )
            return scale
        from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale

        return tma_align_input_scale(scale)
    if scale.dtype != torch.int32 or scale.stride(-2) != 1:
        raise ValueError("Blackwell FP8 expert input scales are packed MN-major int32")
    return scale


@torch.no_grad()
def expert_forward(
    x,
    w13,
    w2,
    *,
    x_scale=None,
    m_indices=None,
    masked_m=None,
    expected_m=None,
    swiglu_limit=None,
    cache_scales=False,
):
    """Return unweighted BF16 rows, with native FP8 weight/scale pairs.

    Contiguous segments must be 128-row aligned; masked rows are [expert, M, K].
    The caller ignores padding and performs slot-ordered weighting/reduction.
    ``x_scale``: ``x`` is quantize_expert_input's payload as DeepEP delivered it,
    with these scales (_gemm_input_scales lists the layouts).
    Forward only; there is no gradient. ``cache_scales``: the weight scales are
    serving parameters (hopper_scales.block_scales).
    """
    if x.requires_grad or any(t.requires_grad for t in (*w13, *w2)):
        raise NotImplementedError("FP8 surrogate backward is not implemented")
    if (m_indices is None) == (masked_m is None):
        raise ValueError("Specify exactly one of m_indices and masked_m")
    if masked_m is not None and expected_m is None:
        raise ValueError("Masked GEMM requires an expected row count")
    if x_scale is not None and x.dtype != torch.float8_e4m3fn:
        raise ValueError("Pre-quantized expert input is E4M3 rows")
    x = quantize_expert_input(x, masked_m=masked_m) if x_scale is None else (x, x_scale)
    payload, _ = x
    if payload.numel() == 0:
        return torch.empty(payload.shape, device=payload.device, dtype=torch.bfloat16)
    if x_scale is not None:
        x = (payload, _gemm_input_scales(x_scale))
    gate = torch.empty(
        (*payload.shape[:-1], w13[0].shape[-2]),
        device=payload.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty(payload.shape, device=payload.device, dtype=torch.bfloat16)

    def gemm(a, b, d):
        if uses_hopper_scales():
            from .hopper_scales import weight_scales

            # The activation scales are already FP32 on Hopper.
            b = weight_scales(b, cached=cache_scales)
        if masked_m is not None:
            dg.grouped_gemm_nt_f8f8bf16_masked(a, b, d, masked_m, expected_m)
        else:
            dg.grouped_gemm_nt_f8f8bf16_contig(a, b, d, m_indices)

    gemm(x, w13, gate)
    gemm(
        quantize_rows(
            gate, fused_swiglu=True, masked_m=masked_m, swiglu_limit=swiglu_limit
        ),
        w2,
        out,
    )
    return out

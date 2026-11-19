import torch

from sglang.kernels.ops.gemm.inkling2_cute_grouped_nvfp4 import routed_swap_matmul
from sglang.kernels.ops.moe.inkling2_moe import _sm_count, routed_quant_module


def _routed_gemm(device, rows, weight, scales, counts, offsets, schedule, quantize):
    """Grouped NVFP4 GEMM of rows that quantize(module, values, x_scales, amax,
    sm_count) writes, with the activation scales laid out as one 128-row scale
    block run per expert. Returns the FP32 accumulator and the per-row amax."""
    experts, output_size, packed_width = weight.shape
    width = packed_width * 2
    sm_count = _sm_count(device)
    scale_rows = (rows + 127 * experts + 127) // 128 * 128
    x_scales = torch.empty(scale_rows * (width // 16), dtype=torch.uint8, device=device)
    values = torch.empty((rows, packed_width), dtype=torch.uint8, device=device)
    amax = torch.empty(rows, dtype=torch.float32, device=device)
    quantize(routed_quant_module(width), values, x_scales, amax, sm_count)
    out = torch.empty((rows, output_size), dtype=torch.float32, device=device)
    routed_swap_matmul(values, x_scales, weight, scales, counts, offsets, schedule, out)
    return out, amax


def routed_expert_matmul(x, dst2src, top_k, weight, scales, counts, offsets, schedule):
    """Grouped NVFP4 GEMM of the routed token rows of x (row r reads token
    dst2src[r] // top_k). Returns the unscaled FP32 accumulator and per-row amax;
    rows outside offsets[0]..offsets[-1] are left unset."""

    def quantize(module, values, x_scales, amax, sm_count):
        module.quantize(x, dst2src, values, x_scales, amax, offsets, top_k, sm_count)

    return _routed_gemm(
        x.device, dst2src.numel(), weight, scales, counts, offsets, schedule, quantize
    )


def routed_swiglu_expert_matmul(
    accumulator,
    accumulator_amax,
    accumulator_weight_amax,
    accumulator_weight_denominator,
    gammas,
    gate_cap,
    up_cap,
    weight,
    scales,
    counts,
    offsets,
    out_dtype,
    schedule,
):
    """Scale the previous GEMM's accumulator, apply capped swiglu, quantize, and
    run the next grouped NVFP4 GEMM, for rows in offsets[0]..offsets[-1]."""
    if out_dtype != torch.bfloat16:
        raise ValueError("Routed NVFP4 swiglu rounds activations to bfloat16")

    def quantize(module, values, x_scales, amax, sm_count):
        module.swiglu_quantize(
            accumulator,
            accumulator_amax,
            accumulator_weight_amax,
            accumulator_weight_denominator,
            gammas,
            values,
            x_scales,
            amax,
            offsets,
            gate_cap,
            up_cap,
            sm_count,
        )

    return _routed_gemm(
        accumulator.device,
        accumulator.shape[0],
        weight,
        scales,
        counts,
        offsets,
        schedule,
        quantize,
    )

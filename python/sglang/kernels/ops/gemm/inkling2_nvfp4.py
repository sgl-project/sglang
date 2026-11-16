import torch
import triton

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.gemm.inkling2_cute_grouped_nvfp4 import routed_swap_matmul

SWAP_AB_MAX_ROWS = 20480


@cache_once
def _matmul_module():
    return load_jit(
        "inkling2_nvfp4",
        cuda_files=["inkling2/nvfp4.cuh"],
        cuda_wrappers=[
            ("run_wide_preswizzled", "inkling2_nvfp4::run_wide_preswizzled"),
            ("wide_x_scales_offset", "inkling2_nvfp4::wide_x_scales_offset"),
            ("wide_workspace_size", "inkling2_nvfp4::wide_workspace_size"),
            ("swizzle_weight_scales", "inkling2_nvfp4::swizzle_weight_scales"),
        ],
        extra_cuda_cflags=["-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_dependencies=["cutlass"],
    )


@cache_once
def routed_quant_module(width: int):
    return load_jit(
        "inkling2_routed_quant",
        str(width),
        cuda_files=["inkling2/routed_quant.cuh"],
        cuda_wrappers=[
            ("quantize", f"inkling2_routed_quant::quantize<{width}>"),
            ("swiglu_quantize", f"inkling2_routed_quant::swiglu_quantize<{width}>"),
            (
                "mxfp8_swiglu_quantize",
                f"inkling2_routed_quant::mxfp8_swiglu_quantize<{width}>",
            ),
            ("mxfp8_quantize", f"inkling2_routed_quant::mxfp8_quantize<{width}>"),
        ],
    )


def swizzled_scale_size(output_size: int, width: int) -> int:
    return triton.cdiv(output_size, 128) * 128 * triton.cdiv(width, 64) * 4


def swizzle_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert (experts, n, k // 16) block scales into the GEMM's scale layout."""
    if (
        scales.ndim != 3
        or scales.dtype != torch.float8_e4m3fn
        or not scales.is_contiguous()
        or not scales.is_cuda
    ):
        raise ValueError(
            f"expected contiguous CUDA float8_e4m3fn scales of shape (experts, n, k // 16), "
            f"got {tuple(scales.shape)} {scales.dtype}"
        )
    experts, output_size, scale_width = scales.shape
    out = torch.zeros(
        (experts, swizzled_scale_size(output_size, scale_width * 16)),
        dtype=scales.dtype,
        device=scales.device,
    )
    if out.numel():
        _matmul_module().swizzle_weight_scales(scales, out)
    return out


def _routed_buffers(module, device, wide, experts, rows, n, k, sm_count):
    """Workspace (wide CUTLASS GEMM only) and the activation scale buffer, laid
    out as one 128-row scale block run per expert."""
    if not wide:
        scale_rows = (rows + 127 * experts + 127) // 128 * 128
        return None, torch.empty(
            scale_rows * (k // 16), dtype=torch.uint8, device=device
        )
    workspace = torch.empty(
        module.wide_workspace_size(experts, rows, n, k, sm_count),
        dtype=torch.uint8,
        device=device,
    )
    return workspace, workspace[module.wide_x_scales_offset(experts, rows, n, k) :]


def routed_expert_matmul(x, dst2src, top_k, weight, scales, counts, offsets, schedule):
    """Grouped NVFP4 GEMM of the routed token rows of x (row r reads token
    dst2src[r] // top_k). Returns the unscaled FP32 accumulator and per-row amax;
    rows outside offsets[0]..offsets[-1] are left unset."""
    experts, output_size, packed_width = weight.shape
    width = packed_width * 2
    rows = dst2src.numel()
    module = _matmul_module()
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    wide = rows > SWAP_AB_MAX_ROWS
    workspace, x_scales = _routed_buffers(
        module, x.device, wide, experts, rows, output_size, width, sm_count
    )
    values = x.new_empty((rows, packed_width), dtype=torch.uint8)
    amax = torch.empty(rows, dtype=torch.float32, device=x.device)
    routed_quant_module(width).quantize(
        x, dst2src, values, x_scales, amax, offsets, top_k, sm_count
    )
    accumulator = torch.empty((rows, output_size), dtype=torch.float32, device=x.device)
    if wide:
        module.run_wide_preswizzled(
            values, weight, scales, counts, offsets, accumulator, workspace, sm_count
        )
    else:
        routed_swap_matmul(
            values, x_scales, weight, scales, counts, offsets, schedule, accumulator
        )
    return accumulator, amax


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
    experts, output_size, packed_width = weight.shape
    width = packed_width * 2
    rows = accumulator.shape[0]
    module = _matmul_module()
    sm_count = torch.cuda.get_device_properties(
        accumulator.device
    ).multi_processor_count
    wide = rows > SWAP_AB_MAX_ROWS
    workspace, x_scales = _routed_buffers(
        module, accumulator.device, wide, experts, rows, output_size, width, sm_count
    )
    values = accumulator.new_empty((rows, packed_width), dtype=torch.uint8)
    amax = torch.empty(rows, dtype=torch.float32, device=accumulator.device)
    routed_quant_module(width).swiglu_quantize(
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
    out = torch.empty(
        (rows, output_size), dtype=torch.float32, device=accumulator.device
    )
    if wide:
        module.run_wide_preswizzled(
            values, weight, scales, counts, offsets, out, workspace, sm_count
        )
    else:
        routed_swap_matmul(
            values, x_scales, weight, scales, counts, offsets, schedule, out
        )
    return out, amax

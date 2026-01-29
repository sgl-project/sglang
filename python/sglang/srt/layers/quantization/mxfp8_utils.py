

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter
from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize

def _quantize_and_swizzle_with_cutlass_es_kernel(weight: torch.Tensor):
    from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_quant

    weight = weight.contiguous()
    num_experts, m, k = weight.shape
    assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

    weight_flat = weight.view(-1, k).contiguous()
    problem_sizes = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=weight.device
    )
    problem_sizes[:, 0] = m
    problem_sizes[:, 1] = 0
    problem_sizes[:, 2] = k
    expert_offsets = torch.arange(
        0, num_experts * m, m, dtype=torch.int32, device=weight.device
    )
    aligned_m = ((m + 127) // 128) * 128
    blockscale_offsets = torch.arange(
        0,
        num_experts * aligned_m,
        aligned_m,
        dtype=torch.int32,
        device=weight.device,
    )
    qweight = torch.empty_like(weight_flat, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (num_experts * aligned_m, k // 32),
        dtype=torch.uint8,
        device=weight.device,
    )
    es_sm100_mxfp8_blockscaled_grouped_quant(
        weight_flat,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        qweight,
        scale,
    )
    qweight = qweight.view_as(weight)
    scale = scale.view(num_experts, aligned_m, k // 32)
    if aligned_m != m:
        scale = scale[:, :m, :]
    return qweight, scale

def _swizzle_mxfp8_sf(scale, num_warps):
    from triton_kernels.tensor import convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout

    scale_layout, scale_layout_opts = (
        layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=1, num_warps=num_warps
        )
    )
    scale = scale.transpose(-2, -1)
    scale = convert_layout(
        wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
    )
    return scale

def _swizzle_with_triton_kernel(
    weight_shape: tuple[int, int, int], scale: torch.Tensor
):
    num_experts, m, k = weight_shape
    aligned_m = ((m + 127) // 128) * 128
    scale = scale.view(num_experts, aligned_m, k // 32)
    num_warps = 8
    scale = _swizzle_mxfp8_sf(scale, num_warps)
    scale = scale.data.view(num_experts, aligned_m, k // 32)
    return scale

def _quantize_and_swizzle_with_triton_kernel(weight: torch.Tensor):

    weight = weight.contiguous()
    _, _, k = weight.shape
    assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

    weight_flat = weight.view(-1, k).contiguous()
    qweight, scale = mxfp8_group_quantize(weight_flat)
    qweight = qweight.view_as(weight)
    scale = _swizzle_with_triton_kernel(weight.shape, scale)
    return qweight, scale

# Keep parameter objects to preserve weight_loader attrs for hot reload.
# Prefer in-place copy; rebind only when shape/dtype changes (online quantize).
def _copy_or_rebind(param: Parameter, new_value: torch.Tensor) -> None:
    if (
        param.data.shape == new_value.shape
        and param.data.dtype == new_value.dtype
    ):
        param.data.copy_(new_value)
    else:
        param.data = new_value

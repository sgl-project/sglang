# SPDX-License-Identifier: Apache-2.0

import logging

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import gptq_marlin_gemm, gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)


def nvfp4_marlin_process_scales(marlin_scales):
    if not (marlin_scales >= 0).all():
        logger.warning_once(
            "NVFP4 Marlin assumes the scales to be >=0, but has encountered "
            "negative scales. Accuracy will likely be degraded. This is "
            "because it changes the scales from FP8-S1E4M3 to a special "
            "FP8-S0E5M3 format to speedup the dequantization."
        )

    # convert to half first, we would convert to fp8 later
    marlin_scales = marlin_scales.to(torch.half)

    # 8 is the number of scale number using by one thread
    marlin_scales = marlin_scales.view(marlin_scales.size(0) // 2, 2, -1, 8)
    marlin_scales = marlin_scales.permute(0, 2, 1, 3).reshape(
        marlin_scales.size(0) * 2, -1
    )

    # fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )

    # We assume that weight_scale (FP8-S1E4M3) is always greater
    # than or equal to 0. So we can convert
    # (weight_scale * (2 ** 7) to a special FP8-S0E5M3 format.
    # After multiplying by 2 ** 7, the top bit of FP8-S0E5M3 would always be 1
    # when weight_scale > 0. This allows us to have an exponent bias
    # closer to zero after dequantization.

    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()

    return marlin_scales


def mxfp4_marlin_process_scales(marlin_scales):
    # 8 is the number of scale number using by one thread
    marlin_scales = marlin_scales.view(marlin_scales.size(0) // 2, 2, -1, 8)
    marlin_scales = marlin_scales.permute(0, 2, 1, 3).reshape(
        marlin_scales.size(0) * 2, -1
    )

    # fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    return marlin_scales


def nvfp4_marlin_process_global_scale(global_scale):
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 1 = 14
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 1 = 126
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP4 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP4 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=size_n, k=size_k, device=input.device, dtype=input.dtype
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_bias=bias,
        b_scales=weight_scale,
        global_scale=weight_scale_2,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float4_e2m1f,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    return output.reshape(out_shape)


def prepare_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    is_nvfp4 = hasattr(layer, "weight_scale_2")
    group_size = 16 if is_nvfp4 else 32

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype

    assert layer.weight.shape == (part_size_n, part_size_k // 2)

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace(device)

    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()

    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    # Permute scales
    weight_scale = layer.weight_scale.T.contiguous()

    if not is_nvfp4:
        weight_scale = weight_scale.view(torch.float8_e8m0fnu)

    weight_scale = weight_scale.to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale, size_k=part_size_k, size_n=part_size_n, group_size=group_size
    )

    if is_nvfp4:
        weight_scale = nvfp4_marlin_process_scales(weight_scale)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        weight_scale_2 = layer.weight_scale_2.to(param_dtype)
        weight_scale_2 = nvfp4_marlin_process_global_scale(weight_scale_2)
        layer.weight_scale_2 = torch.nn.Parameter(weight_scale_2, requires_grad=False)
    else:
        weight_scale = mxfp4_marlin_process_scales(weight_scale)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)

    return

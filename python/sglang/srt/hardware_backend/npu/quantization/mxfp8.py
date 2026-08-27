from typing import List

import torch
from torch.nn import Module

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    ceil_to_ue8m0,
)

_NPU_ARCH35_MXFP8_BLOCK_SIZE = 32
_FP8_E4M3FN_MAX = torch.finfo(torch.float8_e4m3fn).max


def process_npu_arch35_mxfp8_linear_weights(
    layer: Module, weight_block_size: List[int]
) -> None:
    """Convert block-FP8 weights to the NPU arch35 MXFP8 layout.

    The checkpoint stores arbitrary FP32 scale factors per quantization block,
    while arch35 GEMM consumes one UE8M0 scale per 32 K elements. Requantize
    the dequantized weights so the FP8 values and UE8M0 scales remain consistent.
    """
    weight = block_quant_dequant(
        layer.weight.data,
        layer.weight_scale_inv.data,
        weight_block_size,
        torch.bfloat16,
    )
    n_dim, k_dim = weight.shape
    if k_dim % (2 * _NPU_ARCH35_MXFP8_BLOCK_SIZE) != 0:
        raise ValueError(
            "NPU arch35 MXFP8 linear requires K to be divisible by "
            f"{2 * _NPU_ARCH35_MXFP8_BLOCK_SIZE}, got {k_dim}."
        )

    weight_groups = weight.float().reshape(
        n_dim,
        k_dim // _NPU_ARCH35_MXFP8_BLOCK_SIZE,
        _NPU_ARCH35_MXFP8_BLOCK_SIZE,
    )
    scale = ceil_to_ue8m0(
        weight_groups.abs().amax(dim=-1, keepdim=True) / _FP8_E4M3FN_MAX
    )
    qweight = (weight_groups / scale).to(torch.float8_e4m3fn).reshape(n_dim, k_dim)
    scale_u8 = (scale.squeeze(-1).view(torch.int32) >> 23).to(torch.uint8)

    layer.weight.data = qweight.transpose(0, 1)
    layer.weight_scale_inv.data = scale_u8.reshape(
        n_dim, k_dim // (2 * _NPU_ARCH35_MXFP8_BLOCK_SIZE), 2
    ).transpose(0, 1)
    layer.weight_scale_inv.format_ue8m0 = True

    if getattr(layer, "_dsv4_npu_arch35_mxfp8_wo_a", False):
        batch_npu_arch35_wo_a_weights(layer)


def batch_npu_arch35_wo_a_weights(layer: Module) -> None:
    """Reshape DSV4's ``wo_a`` for arch35 batched MXFP8 matmul.

    ``npu_transpose_quant_batchmatmul`` expects weight
    ``[D, G*R] -> [G, D, R]`` and scale
    ``[D/64, G*R, 2] -> [G, D/64, R, 2]``.
    """
    num_groups = layer._dsv4_num_groups
    rank = layer._dsv4_o_lora_rank
    hidden_dim = layer.weight.shape[0]
    scale_k64 = layer.weight_scale_inv.shape[0]
    output_dim = num_groups * rank

    if layer.weight.shape != (hidden_dim, output_dim):
        raise ValueError(
            "Unexpected NPU arch35 wo_a weight layout after FP8 post-processing: "
            f"got {tuple(layer.weight.shape)}, expected ({hidden_dim}, {output_dim})."
        )
    if layer.weight_scale_inv.shape != (scale_k64, output_dim, 2):
        raise ValueError(
            "Unexpected NPU arch35 wo_a scale layout after FP8 post-processing: "
            f"got {tuple(layer.weight_scale_inv.shape)}, expected "
            f"({scale_k64}, {output_dim}, 2)."
        )
    if scale_k64 * 64 != hidden_dim:
        raise ValueError(
            "Unexpected NPU arch35 wo_a scale K dimension: "
            f"{scale_k64} packed pairs for hidden dim {hidden_dim}."
        )

    layer.weight.data = (
        layer.weight.data.T.reshape(num_groups, rank, hidden_dim)
        .transpose(1, 2)
        .contiguous()
    )
    layer.weight_scale_inv.data = (
        layer.weight_scale_inv.data.transpose(0, 1)
        .reshape(num_groups, rank, scale_k64, 2)
        .transpose(1, 2)
        .contiguous()
    )

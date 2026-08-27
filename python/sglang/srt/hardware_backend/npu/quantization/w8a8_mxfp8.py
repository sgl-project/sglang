from typing import List

import torch
from torch.nn import Module

_NPU_ARCH35_MXFP8_BLOCK_SIZE = 32


def process_npu_arch35_mxfp8_linear_weights(
    layer: Module, weight_block_size: List[int], scale_fmt: str
) -> None:
    """Convert UE8M0 block-FP8 weights to the NPU arch35 MXFP8 layout."""
    if scale_fmt != "ue8m0":
        raise ValueError(
            "NPU arch35 MXFP8 weight loading requires scale_fmt='ue8m0', "
            f"got {scale_fmt!r}."
        )
    _layout_npu_arch35_ue8m0_weights(layer, weight_block_size)


def _layout_npu_arch35_ue8m0_weights(
    layer: Module, weight_block_size: List[int]
) -> None:
    """Reinterpret UE8M0 block scales and transpose weights without requantizing."""
    block_n, block_k = weight_block_size
    group_size = _NPU_ARCH35_MXFP8_BLOCK_SIZE
    n_dim, k_dim = layer.weight.shape
    if block_k % group_size != 0:
        raise ValueError(
            f"UE8M0 block K size must be divisible by {group_size}, got {block_k}."
        )
    if k_dim % (2 * group_size) != 0:
        raise ValueError(
            "NPU arch35 MXFP8 linear requires K to be divisible by "
            f"{2 * group_size}, got {k_dim}."
        )

    expected_scale_shape = (
        (n_dim + block_n - 1) // block_n,
        (k_dim + block_k - 1) // block_k,
    )
    checkpoint_scale = layer.weight_scale_inv.data
    if tuple(checkpoint_scale.shape) != expected_scale_shape:
        raise ValueError(
            "Unexpected UE8M0 scale shape: "
            f"got {tuple(checkpoint_scale.shape)}, expected {expected_scale_shape}."
        )

    if checkpoint_scale.dtype == torch.float8_e8m0fnu:
        scale_u8 = checkpoint_scale.view(torch.uint8)
    elif checkpoint_scale.dtype == torch.uint8:
        scale_u8 = checkpoint_scale
    elif checkpoint_scale.dtype == torch.float32:
        # SGLang's block scale parameter is currently allocated as FP32. The
        # loader converts F8_E8M0 values to exact powers of two, so recover the
        # original exponent byte without materializing the weight in FP32.
        scale_u8 = ((checkpoint_scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    else:
        raise TypeError(
            "UE8M0 checkpoint scales must be float8_e8m0fnu, uint8, or float32, "
            f"got {checkpoint_scale.dtype}."
        )

    scale_u8 = scale_u8.repeat_interleave(block_n, dim=0)[:n_dim]
    scale_u8 = scale_u8.repeat_interleave(block_k // group_size, dim=1)
    scale_u8 = scale_u8[:, : k_dim // group_size]

    # Keep transpose views: the A5 kernel expects the original row-major
    # storage scanned in K-major logical order.
    layer.weight.data = layer.weight.data.transpose(0, 1)
    layer.weight_scale_inv.data = scale_u8.reshape(
        n_dim, k_dim // (2 * group_size), 2
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

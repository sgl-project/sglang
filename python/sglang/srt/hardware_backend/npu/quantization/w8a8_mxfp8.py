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


def _dequant_e4m3fn_to_float32(u8: torch.Tensor) -> torch.Tensor:
    """Decode float8_e4m3fn payload bytes to FP32 by explicit bit layout.

    Direct ``.to(torch.float32)`` on NPU fp8 tensors fails on some CANN builds
    (aclnnInplaceCopy error 561103), so decode manually. Only used at
    weight-load time, where the extra elementwise ops are irrelevant.
    """
    bits = u8.view(torch.uint8).to(torch.int32)
    sign = torch.where(bits >= 0x80, -1.0, 1.0)
    e = ((bits >> 3) & 0xF).to(torch.float32)
    m = (bits & 0x7).to(torch.float32)
    mag = torch.where(
        e == 0.0,
        m * (1.0 / 512.0),  # subnormal: m * 2^-9
        (1.0 + m / 8.0) * torch.pow(2.0, e - 7.0),
    )
    # e4m3fn reserves S.1111.111 for NaN (no infinities); max finite is 448.
    mag = torch.where((e == 15.0) & (m == 7.0), float("nan"), mag)
    return sign * mag


def requant_npu_arch35_block_fp8_to_mxfp8(
    layer: Module, weight_block_size: List[int]
) -> None:
    """Requantize a plain block-FP8 weight (fp32 block scales) to MXFP8 layout.

    Dequantizes the fp8 payload with the expanded block scales to BF16, then
    requantizes to MXFP8 (fp8 payload + 1x32 UE8M0 scale via
    npu_dynamic_mx_quant) so the native A5 quantized GEMM
    (``npu_w8a8_mxfp8_linear``) can run the layer. Chunked over rows to cap
    peak memory; runs once at load time.
    """
    block_n, block_k = weight_block_size
    group_size = _NPU_ARCH35_MXFP8_BLOCK_SIZE
    weight = layer.weight.data
    n_dim, k_dim = weight.shape
    if k_dim % (2 * group_size) != 0:
        raise ValueError(
            "NPU arch35 MXFP8 linear requires K to be divisible by "
            f"{2 * group_size}, got {k_dim}."
        )
    device = f"npu:{torch.npu.current_device()}"
    if not weight.is_npu:
        weight = weight.to(device)
    scale = layer.weight_scale_inv.data.to(device)

    bf16 = torch.empty(n_dim, k_dim, dtype=torch.bfloat16, device=weight.device)
    rows_per_chunk = block_n * max(1, 1024 // block_n)
    for r0 in range(0, n_dim, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, n_dim)
        s = scale[r0 // block_n : (r1 + block_n - 1) // block_n]
        s = s.repeat_interleave(block_n, dim=0)[: r1 - r0].repeat_interleave(
            block_k, dim=1
        )[:, :k_dim]
        bf16[r0:r1] = (_dequant_e4m3fn_to_float32(weight[r0:r1]) * s).to(
            torch.bfloat16
        )

    qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
        bf16, dst_type=torch.float8_e4m3fn
    )

    # Layout mirrors _layout_npu_arch35_ue8m0_weights: weight [in, out] and
    # scale [in//64, out, 2] as strided transpose views — DO NOT call
    # .contiguous() (the A5 kernel scans the row-major source K-major).
    layer.weight.data = qw.transpose(0, 1)
    if w_scale.dim() == 2:
        # Older torch_npu builds return [out, in//32]; reshape to 3D.
        w_scale = w_scale.reshape(w_scale.shape[0], w_scale.shape[1] // 2, 2)
    layer.weight_scale_inv.data = w_scale.transpose(0, 1)


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

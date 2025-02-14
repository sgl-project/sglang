from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.parameter import RowvLLMParameter, _ColumnvLLMParameter
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)
from sglang.srt.utils import is_hip

is_hip_ = is_hip()
_is_cuda = torch.cuda.is_available() and torch.version.cuda
if _is_cuda:
    from sgl_kernel import fp8_blockwise_scaled_mm


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale


def cutlass_block_fp8_supported() -> bool:
    if _is_cuda:
        major, minor = torch.cuda.get_device_capability()
        sm_version = major * 10 + minor
        cuda_version = tuple(map(int, torch.version.cuda.split(".")))
        if cuda_version >= (12, 0) and sm_version >= 90:
            return True
    return False


CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()


def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    # TODO: add more robust shape check here
    shape_supported_by_cutlass = (
        weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0
    )
    if CUTLASS_BLOCK_FP8_SUPPORTED and shape_supported_by_cutlass:
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=True
        )
        output = fp8_blockwise_scaled_mm(
            q_input, weight.T, x_scale, weight_scale.T, out_dtype=input.dtype
        )
    else:
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=False
        )
        output = w8a8_block_fp8_matmul(
            q_input, weight, x_scale, weight_scale, block_size, output_dtype=input.dtype
        )

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    fp8_max = finfo.max
    if is_hip_:
        fp8_max = 224.0
    scale = fp8_max / amax
    x_scl_sat = (x * scale).clamp(min=-fp8_max, max=fp8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_quant_to_tensor_quant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise quantization.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise quantization scale.
    Note only float8 is supported for now.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    x_dq_block_tiles = [
        [
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block_tiles[j][i][:, :] = x_dq_block_tiles[j][i] * x_s[j][i]

    x_q_tensor, scale = input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    return x_q_tensor, scale


class BlockQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    block-wise quantization. Uses both column and row parallelism.
    """

    pass

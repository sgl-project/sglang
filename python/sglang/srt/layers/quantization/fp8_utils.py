from typing import List, Optional, Tuple

import torch
from packaging.version import Version

from sglang.srt.utils import (
    get_device_capability,
    get_nvcc_cuda_version,
    is_cuda,
    is_hip,
)

if is_cuda():
    from sgl_kernel import fp8_scaled_mm

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    static_quant_fp8,
    w8a8_block_fp8_matmul,
)

is_hip_ = is_hip()


def cutlass_fp8_supported():
    if not is_cuda():
        return False
    major, minor = get_device_capability()
    cuda_version = get_nvcc_cuda_version()
    if major >= 9:
        return cuda_version >= Version("12.0")
    elif major == 8 and minor == 9:
        return cuda_version >= Version("12.4")
    return False


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

    q_input, x_scale = per_token_group_quant_fp8(input_2d, block_size[1])
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


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = True,
    use_per_token_if_dynamic: bool = False,
) -> torch.Tensor:
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    # cutlass w8a8 fp8 sgl-kernel only supports per-token scale
    if input_scale is not None:
        assert input_scale.numel() == 1
        # broadcast per-tensor scale to per-token scale when supporting cutlass
        qinput, x_scale = static_quant_fp8(
            input_2d, input_scale, repeat_scale=cutlass_fp8_supported
        )
    else:
        # default use per-token quantization if dynamic
        qinput, x_scale = per_token_group_quant_fp8(
            input_2d, group_size=input_2d.shape[1]
        )

    if cutlass_fp8_supported:
        assert (
            weight_scale.numel() == weight.shape[1]
        ), "cutlass w8a8 fp8 sgl-kernel only supports per-channel scale"
        output = fp8_scaled_mm(
            qinput, weight, x_scale, weight_scale, out_dtype=input.dtype, bias=bias
        )
        return output.view(*output_shape)

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    else:
        per_tensor_weights = weight_scale.numel() == 1
        per_tensor_activations = x_scale.numel() == 1

        if per_tensor_weights and per_tensor_activations:
            # Fused GEMM_DQ
            output = torch._scaled_mm(
                qinput,
                weight,
                out_dtype=input.dtype,
                scale_a=x_scale,
                scale_b=weight_scale,
                bias=bias,
            )
            # A fix for discrepancy in scaled_mm which returns tuple
            # for torch < 2.5 and a single value in torch >= 2.5
            if type(output) is tuple and len(output) == 2:
                output = output[0]

            return torch.narrow(output, 0, 0, input_2d.shape[0]).view(*output_shape)

        else:
            # Fallback for channelwise case, where we use unfused DQ
            # due to limitations with scaled_mm

            # Symmetric quantized GEMM by definition computes the following:
            #   C = (s_x * X) (s_w * W) + bias
            # This is equivalent to dequantizing the weights and activations
            # before applying a GEMM.
            #
            # In order to compute quantized operands, a quantized kernel
            # will rewrite the above like so:
            #   C = s_w * s_x * (X * W) + bias
            #
            # For the scaled_mm fallback case, we break this down, since it
            # does not support s_w being a vector.

            # Making sure the dummy tensor is on the same device as the weight
            global TORCH_DEVICE_IDENTITY
            if TORCH_DEVICE_IDENTITY.device != weight.device:
                TORCH_DEVICE_IDENTITY = TORCH_DEVICE_IDENTITY.to(weight.device)

            # GEMM
            # This computes C = (X * W).
            # Output in fp32 to allow subsequent ops to happen in-place
            output = torch._scaled_mm(
                qinput,
                weight,
                scale_a=TORCH_DEVICE_IDENTITY,
                scale_b=TORCH_DEVICE_IDENTITY,
                out_dtype=torch.float32,
            )
            # A fix for discrepancy in scaled_mm which returns tuple
            # for torch < 2.5 and a single value in torch >= 2.5
            if type(output) is tuple and len(output) == 2:
                output = output[0]
            # Unpad (undo num_token_padding)
            output = torch.narrow(output, 0, 0, input_2d.shape[0])
            x_scale = torch.narrow(x_scale, 0, 0, input_2d.shape[0])

            # DQ
            # C = sw * sx * (X * W) + bias
            output = output * x_scale * weight_scale.t()
            if bias is not None:
                output = output + bias
            return output.to(dtype=input.dtype).view(*output_shape)

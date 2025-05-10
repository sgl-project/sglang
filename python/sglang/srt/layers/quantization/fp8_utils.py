import os
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from sglang.srt.layers.quantization.deep_gemm import _ENABLE_JIT_DEEPGEMM
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_quant_fp8,
    static_quant_fp8,
    w8a8_block_fp8_matmul,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_cuda_version,
    get_device_capability,
    is_cuda,
    is_hip,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()

use_aiter_moe = get_bool_env_var("SGLANG_AITER_MOE")

if _is_hip and use_aiter_moe:
    from aiter import gemm_a8w8_blockscale

if _is_cuda:
    from sgl_kernel import fp8_blockwise_scaled_mm, fp8_scaled_mm

use_vllm_cutlass_w8a8_fp8_kernel = get_bool_env_var("USE_VLLM_CUTLASS_W8A8_FP8_KERNEL")

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def use_rowwise_torch_scaled_mm():
    _TORCH_VERSION = torch.__version__.split("+")[0]
    try:
        _TORCH_VERSION_TUPLE = tuple(map(int, _TORCH_VERSION.split(".")[:3]))
    except ValueError:
        _TORCH_VERSION_TUPLE = (0, 0, 0)
    if _is_hip:
        # The condition to determine if it is on a platform that supports
        # torch._scaled_mm rowwise feature.
        # The condition is determined once as the operations
        # are time consuming.
        return get_device_capability() >= (9, 4) and _TORCH_VERSION_TUPLE >= (2, 7, 0)
    return False


USE_ROWWISE_TORCH_SCALED_MM = use_rowwise_torch_scaled_mm()


def cutlass_fp8_supported():
    if not _is_cuda:
        return False
    major, minor = get_device_capability()
    cuda_version = get_cuda_version()
    if major >= 9:
        return cuda_version >= (12, 0)
    elif major == 8 and minor == 9:
        return cuda_version >= (12, 4)
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


def cutlass_block_fp8_supported() -> bool:
    if not get_bool_env_var("SUPPORT_CUTLASS_BLOCK_FP8"):
        return False
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
    elif _is_hip and use_aiter_moe:
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=False
        )
        output = torch.zeros(
            [q_input.shape[0], weight.shape[0]],
            dtype=input.dtype,
            device=q_input.device,
        )
        gemm_a8w8_blockscale(q_input, weight, x_scale, weight_scale, output)
    else:
        if _ENABLE_JIT_DEEPGEMM:
            q_input, x_scale = sglang_per_token_group_quant_fp8(
                input_2d,
                block_size[1],
                column_major_scales=True,
                scale_tma_aligned=True,
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
    x: torch.Tensor, dtype: torch.dtype = fp8_dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).float().clamp(min=1e-12)

    if _is_fp8_fnuz:
        dtype = fp8_dtype
        fp_max = fp8_max
    else:
        finfo = torch.finfo(dtype)
        fp_max = finfo.max

    scale = fp_max / amax
    x_scl_sat = (x.float() * scale).clamp(min=-fp_max, max=fp_max)
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

    x_q_tensor, scale = (
        scaled_fp8_quant(x_dq_block)
        if _is_cuda
        else input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    )
    return x_q_tensor, scale


def block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """This function converts block-wise quantization to unquantized.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The output is an unquantized tensor with dtype.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = torch.empty_like(x_q_block, dtype=dtype)

    for j in range(n_tiles):
        for i in range(k_tiles):
            x_q_block_tile = x_q_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            x_dq_block_tile = x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            x_dq_block_tile[:, :] = x_q_block_tile.to(torch.float32) * x_s[j][i]

    return x_dq_block


def channel_quant_to_tensor_quant(
    x_q_channel: torch.Tensor,
    x_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_dq_channel = x_q_channel.to(torch.float32) * x_s
    x_q_tensor, scale = (
        scaled_fp8_quant(x_dq_channel)
        if _is_cuda
        else input_to_float8(x_dq_channel, dtype=x_q_channel.dtype)
    )
    return x_q_tensor, scale


def _process_scaled_mm_output(output, input_2d_shape, output_shape):
    if type(output) is tuple and len(output) == 2:
        output = output[0]
    return torch.narrow(output, 0, 0, input_2d_shape[0]).view(*output_shape)


def _apply_fallback_scaled_mm(
    qinput,
    weight,
    x_scale,
    weight_scale,
    input_2d_shape,
    output_shape,
    bias,
    input_dtype,
):
    global TORCH_DEVICE_IDENTITY
    if TORCH_DEVICE_IDENTITY is None:
        TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32, device=weight.device)

    output = torch._scaled_mm(
        qinput,
        weight,
        scale_a=TORCH_DEVICE_IDENTITY,
        scale_b=TORCH_DEVICE_IDENTITY,
        out_dtype=torch.float32,
    )

    output = _process_scaled_mm_output(output, input_2d_shape, output_shape)
    x_scale = torch.narrow(x_scale, 0, 0, input_2d_shape[0])

    output = output * x_scale * weight_scale.t()
    if bias is not None:
        output = output + bias
    return output.to(dtype=input_dtype)


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = cutlass_fp8_supported(),
    use_per_token_if_dynamic: bool = False,
    pad_output: Optional[bool] = None,
    compressed_tensor_quant: bool = False,
) -> torch.Tensor:
    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:
        pad_output = not get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE")
    output_padding = 17 if pad_output else None

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    if compressed_tensor_quant:
        # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
        # for sgl-kernel fp8_scaled_mm, it support per channel W now
        if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
            qinput, x_scale = scaled_fp8_quant(
                input_2d,
                input_scale,
                use_per_token_if_dynamic=use_per_token_if_dynamic,
            )

            # Fused GEMM_DQ
            if VLLM_AVAILABLE and use_vllm_cutlass_w8a8_fp8_kernel:
                # Fall back to vllm cutlass w8a8 fp8 kernel
                output = ops.cutlass_scaled_mm(
                    qinput,
                    weight,
                    out_dtype=input.dtype,
                    scale_a=x_scale,
                    scale_b=weight_scale,
                    bias=bias,
                )
            else:
                assert (
                    weight_scale.numel() == weight.shape[1]
                ), "cutlass w8a8 fp8 sgl-kernel only supports per-channel scale"
                output = fp8_scaled_mm(
                    qinput,
                    weight,
                    x_scale,
                    weight_scale,
                    out_dtype=input.dtype,
                    bias=bias,
                )
            return output.view(*output_shape)

        # torch.scaled_mm supports per tensor weights + activations only
        # so fallback to naive if per channel or per token
        else:
            # Maybe apply padding to output, see comment in __init__
            qinput, x_scale = (
                scaled_fp8_quant(
                    input_2d,
                    input_scale,
                    num_token_padding=output_padding,
                    use_per_token_if_dynamic=use_per_token_if_dynamic,
                )
                if _is_cuda
                else ops.scaled_fp8_quant(
                    input_2d,
                    input_scale,
                    num_token_padding=output_padding,
                    use_per_token_if_dynamic=use_per_token_if_dynamic,
                )
            )

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
                return _process_scaled_mm_output(output, input_2d.shape, output_shape)

            elif (
                use_per_token_if_dynamic
                and not per_tensor_weights
                and not per_tensor_activations
                and USE_ROWWISE_TORCH_SCALED_MM
            ):
                # For now validated on ROCm platform
                # fp8 rowwise scaling in torch._scaled_mm is introduced in
                # https://github.com/pytorch/pytorch/pull/144432 using hipBLASLt
                # and ROCm 6.3, which only exists in torch 2.7 and above.
                # For CUDA platform please validate if the
                # torch._scaled_mm support rowwise scaled GEMM
                # Fused GEMM_DQ Rowwise GEMM
                output = torch._scaled_mm(
                    qinput,
                    weight,
                    out_dtype=input.dtype,
                    scale_a=x_scale,
                    scale_b=weight_scale.t(),
                    bias=bias,
                )
                return _process_scaled_mm_output(output, input_2d.shape, output_shape)

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
                return _apply_fallback_scaled_mm(
                    qinput,
                    weight,
                    x_scale,
                    weight_scale,
                    input_2d.shape,
                    output_shape,
                    bias,
                    input.dtype,
                )
    else:
        # cutlass w8a8 fp8 sgl-kernel only supports per-token scale
        if input_scale is not None:
            assert input_scale.numel() == 1
            # broadcast per-tensor scale to per-token scale when supporting cutlass
            qinput, x_scale = static_quant_fp8(
                input_2d, input_scale, repeat_scale=cutlass_fp8_supported
            )
        else:
            # default use per-token quantization if dynamic
            if _is_cuda:
                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)
            else:
                # TODO(kkhuang): temporarily enforce per-tensor activation scaling if weight is per-tensor scaling
                # final solution should be: 1. add support to per-tensor activation scaling.
                # 2. solve the torch.compile error from weight_scale.numel() == 1 and x_scale.numel() > 1 (below line#308)
                if _is_hip and weight_scale.numel() == 1:
                    qinput, x_scale = ops.scaled_fp8_quant(
                        input_2d,
                        input_scale,
                        use_per_token_if_dynamic=use_per_token_if_dynamic,
                    )
                else:
                    qinput, x_scale = per_token_group_quant_fp8(
                        input_2d, group_size=input_2d.shape[1]
                    )

        if cutlass_fp8_supported:
            try:
                if VLLM_AVAILABLE and use_vllm_cutlass_w8a8_fp8_kernel:
                    # Fall back to vllm cutlass w8a8 fp8 kernel
                    output = ops.cutlass_scaled_mm(
                        qinput,
                        weight,
                        out_dtype=input.dtype,
                        scale_a=x_scale,
                        scale_b=weight_scale,
                        bias=bias,
                    )
                else:
                    assert (
                        weight_scale.numel() == weight.shape[1]
                    ), "cutlass w8a8 fp8 sgl-kernel only supports per-channel scale"
                    output = fp8_scaled_mm(
                        qinput,
                        weight,
                        x_scale,
                        weight_scale,
                        out_dtype=input.dtype,
                        bias=bias,
                    )
                return output.view(*output_shape)
            except (ImportError, NameError, AttributeError):
                pass

        # torch.scaled_mm supports per tensor weights + activations only
        # so fallback to naive if per channel or per token
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
            return _process_scaled_mm_output(output, input_2d.shape, output_shape)

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
            return _apply_fallback_scaled_mm(
                qinput,
                weight,
                x_scale,
                weight_scale,
                input_2d.shape,
                output_shape,
                bias,
                input.dtype,
            )

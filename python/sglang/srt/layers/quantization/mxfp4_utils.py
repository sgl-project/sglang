# SPDX-License-Identifier: Apache-2.0
import torch

OCP_MX_BLOCK_SIZE = 32


def dequant_mxfp4(x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype) -> torch.Tensor:
    """
    Dequantizes the tensor `x`, that is stored in OCP MXPF4 format, to the dtype `float_dtype`.

    Args:
        x (torch.Tensor): The quantized input, expected to be of dtype `torch.uint8`, with two fp4 values [elem1, elem0] packed in one byte on the last dimension.
        scale (torch.Tensor): Corresponding OCP MX scales, expected to be of dtype torch.uint8, storing scales as e8m0.
        float_dtype (torch.dtype): `torch.float16` or `torch.bfloat16`.
    """
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                        "MX-FP4 models. Please install it with `pip install "
                        "amd-quark`.") from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def quant_dequant_mxfp4(x: torch.Tensor, scale_calculation_mode: str = "even") -> torch.Tensor:
    """
    Applies QDQ (dequantize(quantize())) to the tensor `x` simulating OCP MXFP4 quantization, computing OCP MX scales on the fly.

    Args:
        x (torch.Tensor): Input to apply QDQ on. The last dimension is used to compute scales (e.g. if `x` is [dim0, dim1], the scales computed on the fly will be of dim `[dim0, dim1 // 32]`).
    """
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                        "MX-FP4 models. Please install it with `pip install "
                        "amd-quark`.") from err

    return mx.qdq_mxfp4(x, scale_calculation_mode)

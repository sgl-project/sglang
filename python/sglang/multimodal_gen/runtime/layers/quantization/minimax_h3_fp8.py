# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3's measured online-FP8 GEMM policy.

H200 measurements favor Torch/NVJet for the four full-width projections at
2K packed sequence lengths (M=4144). Other shapes keep the generic dispatch.
"""

from __future__ import annotations

import torch

from sglang.kernels import KernelBackend, get_kernel
from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    use_triton_w8a8_fp8_kernel,
)

_H3_PROJECTIONS = frozenset({(5376, 21504), (7168, 5376), (5376, 28672), (14336, 5376)})


def _prefer_h3_nvjet(input, weight, weight_scale, input_scale, bias) -> bool:
    if (
        use_triton_w8a8_fp8_kernel
        or input.device.type != "cuda"
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(input.device) != (9, 0)
        or get_fused_op_backend() not in (None, KernelBackend.TORCH)
        or bias is not None
        or input.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or weight.dtype != torch.float8_e4m3fn
        or weight.device != input.device
        or weight.ndim != 2
        or weight.stride(0) != 1
        or tuple(weight.shape) not in _H3_PROJECTIONS
        or not 4096 <= input.numel() // input.shape[-1] < 8192
        or weight_scale.dtype != torch.float32
        or weight_scale.device != input.device
        or weight_scale.numel() != weight.shape[1]
        or not weight_scale.is_contiguous()
    ):
        return False
    # Native inputs must use dynamic per-token quantization. Supplied FP8 rows
    # already passed Fp8LinearMethod's input validation.
    return input_scale is None or (
        input.dtype == torch.float8_e4m3fn
        and input_scale.numel() == input.numel() // input.shape[-1]
    )


class MiniMaxH3Fp8LinearMethod(Fp8LinearMethod):
    """Keep loading, validation and producer contracts; specialize GEMM only."""

    @staticmethod
    def _apply_fp8_linear(**kwargs):
        x = kwargs["input"]
        weight = kwargs["weight"]
        weight_scale = kwargs["weight_scale"]
        input_scale = kwargs.get("input_scale")
        if not kwargs.get("cutlass_fp8_supported", False) or not _prefer_h3_nvjet(
            x, weight, weight_scale, input_scale, kwargs.get("bias")
        ):
            return apply_fp8_linear(**kwargs)
        x_2d = x.reshape(-1, x.shape[-1])
        if x.dtype == torch.float8_e4m3fn:
            qinput, scales = x_2d, input_scale
        else:
            qinput, scales = sglang_per_token_quant_fp8(x_2d)
        output = get_kernel("gemm.fp8_scaled_mm", KernelBackend.TORCH)(
            qinput,
            weight,
            scales,
            weight_scale,
            out_dtype=torch.bfloat16,
        )
        return output.view(*x.shape[:-1], weight.shape[1])


class MiniMaxH3Fp8Config(Fp8Config):
    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if type(method) is Fp8LinearMethod:
            return MiniMaxH3Fp8LinearMethod(self)
        return method


def specialize_h3_fp8_config(config):
    """Copy plain online FP8 config, leaving the caller's shared config intact."""
    if (
        type(config) is not Fp8Config
        or config.is_checkpoint_fp8_serialized
        or config.weight_block_size is not None
        or config.activation_scheme != "dynamic"
    ):
        return config
    specialized = MiniMaxH3Fp8Config.__new__(MiniMaxH3Fp8Config)
    specialized.__dict__.update(vars(config))
    return specialized

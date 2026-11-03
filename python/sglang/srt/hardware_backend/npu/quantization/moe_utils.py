"""Shared weight preparation, runner and quantization payloads for Ascend MoE."""

from typing import Tuple

import torch

from sglang.srt.hardware_backend.npu.quantization.dtype_utils import (
    _decode_fp32_mxfp_scale,
    _get_float4_e2m1fn_x2_dtype,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast


def create_ascend_moe_runner(
    layer,
    moe_runner_config,
    *,
    w13_kernel,
    w2_kernel,
):
    """Install both kernels before the runner selects its activation."""
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    layer.w13_kernel = w13_kernel
    layer.w2_kernel = w2_kernel
    moe_runner_config.layer = layer
    return MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)


def build_ascend_quant_info(layer):
    """Build the Ascend payload for checkpoints using weight_scale_inv."""
    from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

    return AscendQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale=layer.w13_weight_scale_inv,
        w2_weight_scale=layer.w2_weight_scale_inv,
    )


def reshape_w4a8_mxfp_weight_scale_for_npu(scale: torch.Tensor) -> torch.Tensor:
    """Pack MXFP4 scales from ``[E, N, K/32]`` to the A5 GMM layout."""
    if scale.dim() != 3:
        return scale
    num_experts, n, k32 = scale.shape
    if k32 % 2 != 0:
        raise ValueError(
            "MXFP4 scale K dimension must be divisible by 2 for the "
            f"[E, K/64, N, 2] layout, got {tuple(scale.shape)}."
        )
    return scale.view(num_experts, n, k32 // 2, 2).transpose(1, 2)


def prepare_w4a8_mxfp_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    npu_format=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert packed FP4 weights and scales to the shared NPU GMM layout.

    Scales are either raw E8M0 bytes (ModelSlim / DeepSeek-V4) or float32
    checkpoint values (HF), which need decoding before the shared packing.
    """
    fp4_dtype = _get_float4_e2m1fn_x2_dtype()
    if fp4_dtype is None:
        raise RuntimeError("NPU W4A8 MXFP MoE requires float4 support.")

    if weight.dtype == torch.int8:
        weight = weight.view(torch.uint8)
    cast_args = {
        "customize_dtype": torch.float8_e4m3fn,
        "input_dtype": fp4_dtype,
    }
    if npu_format is None:
        weight = npu_format_cast(weight, **cast_args)
    else:
        weight = npu_format_cast(weight, npu_format, **cast_args)
    if weight_scale.dtype == torch.float32:
        weight_scale = _decode_fp32_mxfp_scale(weight_scale)
    return weight.transpose(-1, -2), reshape_w4a8_mxfp_weight_scale_for_npu(
        weight_scale
    )


def process_npu_fp4_expert_weights(layer: torch.nn.Module, *, npu_format=None) -> None:
    """Convert MXFP4 experts to the NPU W4A8 kernel layout.

    Scales may be HF float32 checkpoint values or raw E8M0 bytes. The NPU
    kernel expects the weight in FRACTAL_NZ (transposed) and the scale as
    E8M0 bytes reshaped to [E, K//64, N, 2]. Parameter objects are preserved.
    """
    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight")
        scale_inv = getattr(layer, f"{prefix}_weight_scale_inv")
        weight.data, scale_inv.data = prepare_w4a8_mxfp_weight(
            weight.data, scale_inv.data, npu_format=npu_format
        )

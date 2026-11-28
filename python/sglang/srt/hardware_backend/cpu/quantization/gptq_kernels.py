# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import MoeRunnerConfig

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig

__all__ = ["GPTQIntelAMXLinearKernel", "GPTQIntelAMXMoEKernel"]


def _unpack_gptq_uint8(
    qweight: torch.Tensor,
    output_channels: int,
    input_channels: int,
    *,
    packed_dim: int = 0,
) -> torch.Tensor:
    """Unpack a GPTQ ``bits=8`` tensor four uint8 values per int32.

    ``qweight`` packs along the input dimension and ``qzeros`` packs along the
    output dimension, so the caller selects the packed dimension explicitly.
    """

    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=qweight.device)
    unpacked = (qweight.unsqueeze(-1) >> shifts) & 0xFF
    if packed_dim == 1:
        return unpacked.reshape(input_channels, output_channels).contiguous()
    return (
        unpacked.permute(0, 2, 1).reshape(input_channels, output_channels).contiguous()
    )


@torch.no_grad()
def _gptq_w8_to_channelwise_w8a8(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    is_symmetric: bool,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a GPTQ-packed group-scaled W8 tensor to per-channel INT8.

    AutoRound's ``auto_round:auto_gptq`` W8 export stores four uint8 values per
    int32 and applies one FP16 scale per quantization group (typically 128
    input channels). The CPU AMX W8A8 GEMM consumes one FP32 scale per output
    channel. We therefore dequantize the checkpoint representation and
    requantize each output channel to INT8 once at model-load time.

    The additional rounding is small for W8 checkpoints (the first W8 level
    already bounds per-group error), and this keeps the existing AMX W8A8 GEMM
    for 8-bit layers while W4 layers continue to use the AMX W4A8 kernel. Exact
    group-scale W8 execution would require a separate grouped-scale AMX kernel.
    """

    input_channels_packed, output_channels = qweight.shape
    input_channels = input_channels_packed * 4
    num_groups = scales.shape[0]
    if scales.shape[1] != output_channels:
        raise ValueError(
            "AutoRound W8 scale shape does not match qweight: "
            f"scales={tuple(scales.shape)}, qweight={tuple(qweight.shape)}."
        )
    if qzeros.shape != (num_groups, output_channels // 4):
        raise ValueError(
            "AutoRound W8 qzeros shape does not match qweight scales: "
            f"qzeros={tuple(qzeros.shape)}, expected="
            f"{(num_groups, output_channels // 4)}."
        )
    if input_channels % num_groups != 0:
        raise ValueError(
            "AutoRound W8 input channels must be divisible by the group count, "
            f"got input_channels={input_channels}, num_groups={num_groups}."
        )
    group_size = input_channels // num_groups
    if group_size % 4 != 0:
        raise ValueError(
            f"AutoRound W8 group_size must be a multiple of 4, got {group_size}."
        )
    if output_channels % 4 != 0:
        raise ValueError(
            "AutoRound W8 output channels must be a multiple of 4, "
            f"got {output_channels}."
        )

    weight = torch.empty(
        (output_channels, input_channels), dtype=torch.int8, device=qweight.device
    )
    weight_scale = torch.empty(
        (output_channels, 1), dtype=torch.float32, device=qweight.device
    )
    zero_point = 1 << 7

    for output_start in range(0, output_channels, chunk_size):
        output_end = min(output_start + chunk_size, output_channels)
        chunk_channels = output_end - output_start

        qweight_chunk = qweight[:, output_start:output_end]
        unpacked_weight = _unpack_gptq_uint8(
            qweight_chunk, chunk_channels, input_channels
        )

        if is_symmetric:
            unpacked_zeros = torch.full(
                (num_groups, chunk_channels),
                zero_point,
                dtype=torch.int32,
                device=qweight.device,
            )
        else:
            packed_zeros = qzeros[:, output_start // 4 : output_end // 4]
            unpacked_zeros = _unpack_gptq_uint8(
                packed_zeros,
                chunk_channels,
                num_groups,
                packed_dim=1,
            )
            unpacked_zeros = unpacked_zeros.to(torch.int32) + 1

        scale_chunk = scales[:, output_start:output_end].to(torch.float32)
        values = unpacked_weight.to(torch.int32).reshape(
            num_groups, group_size, chunk_channels
        )
        values = values - unpacked_zeros[:, None, :]
        dequantized = values.to(torch.float32) * scale_chunk[:, None, :]

        per_channel_scale = dequantized.abs().amax(dim=(0, 1)) / 127.0
        per_channel_scale = torch.where(
            per_channel_scale > 0,
            per_channel_scale,
            torch.ones_like(per_channel_scale),
        )
        quantized = torch.round(dequantized / per_channel_scale[None, None, :])
        quantized = quantized.clamp_(-127, 127).to(torch.int8)
        weight[output_start:output_end] = quantized.reshape(
            input_channels, chunk_channels
        ).t()
        weight_scale[output_start:output_end, 0] = per_channel_scale

    return weight, weight_scale


class GPTQIntelAMXLinearKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.quant_config.weight_bits == 8:
            weight, weight_scale = _gptq_w8_to_channelwise_w8a8(
                layer.qweight.data,
                layer.qzeros.data,
                layer.scales.data,
                is_symmetric=self.quant_config.is_sym,
            )
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
            for name in ("qweight", "qzeros", "scales", "g_idx"):
                delattr(layer, name)
            _amx_process_weight_after_loading(layer, ["weight"])
            if not layer.use_intel_amx_backend:
                raise ValueError(
                    "AutoRound W8 CPU AMX packing requires the input dimension "
                    "to be a multiple of 32 and the output dimension to be a "
                    f"multiple of 16 (or smaller than 16); got weight shape "
                    f"{tuple(layer.weight.shape)}."
                )
            return

        _amx_process_weight_after_loading(
            layer,
            ["qweight", "qzeros", "scales"],
            None,
            "gptq",
            is_symmetric=self.quant_config.is_sym,
        )
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.quant_config.weight_bits == 8:
            return torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                x.dtype,
                True,
            )
        return torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias,
        )


class GPTQIntelAMXMoEKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.moe_runner_config: Optional[MoeRunnerConfig] = None

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert self.quant_config.weight_bits == 4, (
            "AutoRound CPU AMX MoE supports 4-bit experts only."
        )
        _amx_process_weight_after_loading(
            layer,
            ["w13_qweight", "w13_qzeros", "w13_scales"],
            None,
            "gptq",
            is_symmetric=self.quant_config.is_sym,
        )
        _amx_process_weight_after_loading(
            layer,
            ["w2_qweight", "w2_qzeros", "w2_scales"],
            None,
            "gptq",
            is_symmetric=self.quant_config.is_sym,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        activation = self.moe_runner_config.activation
        assert activation in ("silu", "gelu"), (
            f"Only SiLU and GELU activations are supported, got {activation}."
        )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        output = torch.ops.sgl_kernel.fused_experts_cpu(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            False,  # inplace See [Note] inplace should be False in fused_experts.
            CPUQuantMethod.INT4_W4A8,
            layer.w13_scales,  # w1_scale
            layer.w2_scales,  # w2_scale
            layer.w13_qzeros,
            layer.w2_qzeros,
            None,  # block_size
            None,  # w1 bias
            None,  # w3 bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
            activation,
        )
        return StandardCombineInput(hidden_states=output)

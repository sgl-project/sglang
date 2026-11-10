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


class GPTQIntelAMXLinearKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer._amx_output_size = layer.qweight.shape[-1]
        output_padding = -layer._amx_output_size % 32
        layer._amx_output_padding = output_padding
        if output_padding:
            layer.qweight.data = torch.nn.functional.pad(
                layer.qweight.data, (0, output_padding)
            )
            layer.qzeros.data = torch.nn.functional.pad(
                layer.qzeros.data, (0, output_padding // 8)
            )
            layer.scales.data = torch.nn.functional.pad(
                layer.scales.data, (0, output_padding)
            )
        _amx_process_weight_after_loading(
            layer, ["qweight", "qzeros", "scales"], None, "gptq"
        )
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        # layer.bias stays unpadded since skip_bias_add hands it back to the caller.
        layer._amx_padded_bias = (
            torch.nn.functional.pad(layer.bias.data, (0, output_padding))
            if output_padding and layer.bias is not None
            else None
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_size = layer._amx_output_size
        output_padding = layer._amx_output_padding
        if bias is not None and output_padding:
            bias = (
                layer._amx_padded_bias
                if bias is layer.bias
                else torch.nn.functional.pad(bias, (0, output_padding))
            )
        output = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x.reshape(-1, x.shape[-1]),
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias,
        )
        if output_padding:
            output = output[:, :output_size].contiguous()
        return output.view(*x.shape[:-1], output_size)


class GPTQIntelAMXMoEKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.moe_runner_config: Optional[MoeRunnerConfig] = None

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _amx_process_weight_after_loading(
            layer, ["w13_qweight", "w13_qzeros", "w13_scales"], None, "gptq"
        )
        _amx_process_weight_after_loading(
            layer, ["w2_qweight", "w2_qzeros", "w2_scales"], None, "gptq"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
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
        )
        return StandardCombineInput(hidden_states=output)

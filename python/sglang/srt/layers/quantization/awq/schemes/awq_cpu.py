# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import MoeRunnerConfig

from .awq_linear import AWQLinearScheme
from .awq_moe import AWQMoEScheme

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.awq.awq import AWQConfig

__all__ = ["AWQIntelAMXLinearScheme", "AWQIntelAMXMoEScheme"]


class AWQIntelAMXLinearKernel:
    def __init__(self, quant_config: "AWQConfig"):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _amx_process_weight_after_loading(
            layer, ["qweight", "qzeros", "scales"], None, "awq"
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
        return torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias,
        )


class AWQIntelAMXLinearScheme(AWQLinearScheme):
    """Linear scheme for AWQ on Intel CPU with AMX."""

    def _init_kernel(self, quant_config: "AWQConfig"):
        return AWQIntelAMXLinearKernel(quant_config)


class AWQIntelAMXMoEKernel:
    def __init__(self, quant_config: "AWQConfig"):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _amx_process_weight_after_loading(
            layer, ["w13_qweight", "w13_qzeros", "w13_scales"], None, "awq"
        )
        _amx_process_weight_after_loading(
            layer, ["w2_qweight", "w2_qzeros", "w2_scales"], None, "awq"
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

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
            True,  # is_vnni
        )
        return StandardCombineInput(hidden_states=output)


class AWQIntelAMXMoEScheme(AWQMoEScheme):
    """MoE scheme for AWQ on Intel CPU with AMX."""

    def _init_kernel(self, quant_config: "AWQConfig"):
        return AWQIntelAMXMoEKernel(quant_config)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.kernel.create_moe_runner(layer, moe_runner_config)

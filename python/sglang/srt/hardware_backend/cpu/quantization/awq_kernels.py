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
    from sglang.srt.layers.quantization.awq.awq import AWQConfig

__all__ = ["AWQIntelAMXLinearKernel", "AWQIntelAMXMoEKernel"]


class AWQIntelAMXLinearKernel:
    def __init__(self, quant_config: AWQConfig):
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


class AWQIntelAMXMoEKernel:
    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config
        self.moe_runner_config: Optional[MoeRunnerConfig] = None

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
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        assert self.moe_runner_config is not None
        moe_runner_config = self.moe_runner_config

        assert moe_runner_config.activation in ["silu", "relu2"], (
            f"Only SiLU/ReLU2 activations are supported on CPU INT4, "
            f"got {moe_runner_config.activation!r}."
        )
        if moe_runner_config.activation == "relu2":
            assert not moe_runner_config.is_gated, (
                "relu2 is only valid for non-gated MoE experts on CPU INT4."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        if moe_runner_config.activation == "relu2":
            # CPU INT4 fused_experts currently only supports silu; emulate relu2
            # with per-expert INT4 GEMMs in Python without touching backend C++.
            out = x.new_zeros((x.shape[0], layer.w2_qweight.shape[-1]))
            token_idx, slot_idx = torch.nonzero(topk_ids >= 0, as_tuple=True)
            if token_idx.numel() == 0:
                return StandardCombineInput(hidden_states=out)

            selected_experts = topk_ids[token_idx, slot_idx].to(torch.long)
            selected_weights = topk_weights[token_idx, slot_idx]

            for expert_id in selected_experts.unique(sorted=False).tolist():
                expert_mask = selected_experts == expert_id
                if not torch.any(expert_mask):
                    continue

                rows = token_idx[expert_mask]
                row_weights = selected_weights[expert_mask].to(x.dtype).unsqueeze(-1)
                x_rows = x.index_select(0, rows)

                w13 = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
                    x_rows,
                    layer.w13_qweight[expert_id],
                    layer.w13_qzeros[expert_id],
                    layer.w13_scales[expert_id],
                    None,
                )
                w13 = torch.relu(w13)
                w13 = w13 * w13

                expert_out = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
                    w13,
                    layer.w2_qweight[expert_id],
                    layer.w2_qzeros[expert_id],
                    layer.w2_scales[expert_id],
                    None,
                )
                expert_out = expert_out * row_weights
                out.index_add_(0, rows, expert_out.to(out.dtype))

            return StandardCombineInput(hidden_states=out)

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
            moe_runner_config.activation,  # activation
        )
        return StandardCombineInput(hidden_states=output)

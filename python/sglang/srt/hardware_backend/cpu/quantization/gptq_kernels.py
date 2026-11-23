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


def _normalize_symmetric_qzeros(
    layer: torch.nn.Module, name: str, quant_config: GPTQConfig
) -> None:
    """Adapt explicit symmetric INT4 zeros to the AMX packer's legacy input."""
    # quantization.__init__ imports GPTQ schemes, which import this module.
    # Resolve dynamic overrides only after the kernel classes are initialized.
    from sglang.srt.layers.quantization.utils import get_dynamic_override

    prefix = getattr(layer, "prefix", getattr(layer, "layer_name", ""))
    sym = get_dynamic_override(
        quant_config, prefix, "sym", getattr(quant_config, "sym", None)
    )
    if sym is not True or quant_config.weight_bits != 4:
        return

    qzeros = getattr(layer, name)
    # The CPU GPTQ packer adds 1 to legacy serialized zeros. Some symmetric
    # checkpoints, including Qwen3.5, already store the mathematical zero 8.
    # Only normalize an explicitly symmetric tensor made entirely of packed
    # 8s (0x88888888 as int32). Legacy 7s and asymmetric/unknown formats keep
    # their existing interpretation. Replacing data never edits mmap storage.
    if (
        qzeros.dtype == torch.int32
        and qzeros.numel() > 0
        and bool(torch.all(qzeros == -2004318072))
    ):
        qzeros.data = torch.full_like(qzeros, 0x77777777)


class GPTQIntelAMXLinearKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _normalize_symmetric_qzeros(layer, "qzeros", self.quant_config)
        _amx_process_weight_after_loading(
            layer, ["qweight", "qzeros", "scales"], None, "gptq"
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


class GPTQIntelAMXMoEKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.moe_runner_config: Optional[MoeRunnerConfig] = None

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _normalize_symmetric_qzeros(layer, "w13_qzeros", self.quant_config)
        _amx_process_weight_after_loading(
            layer, ["w13_qweight", "w13_qzeros", "w13_scales"], None, "gptq"
        )
        _normalize_symmetric_qzeros(layer, "w2_qzeros", self.quant_config)
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

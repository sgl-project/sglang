"""Register the MoE LoRA engine as a fused MoE runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraBatch, MoeLoraRunner


@dataclass
class MoeLoraDispatchPayload(MoeQuantInfo):
    """Borrowed runner and batch state carried through the quant_info slot."""

    runner: MoeLoraRunner
    batch: MoeLoraBatch


@register_fused_func("none", "lora_cutedsl")
@register_fused_func("none", "lora_triton")
@register_fused_func("none", "lora_marlin")
def fused_experts_none_to_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeLoraDispatchPayload,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    del runner_config
    return quant_info.runner.run(dispatch_output, quant_info.batch)

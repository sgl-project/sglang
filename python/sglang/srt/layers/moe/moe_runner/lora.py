"""Runner registration for the ``lora`` MoE backend; the engine lives in
``sglang/srt/lora/moe``.

Deliberately no LoRA hooks: hooks interpose LoRA into someone else's expert
pipeline, while this backend is itself the LoRA implementation. Per-forward
state rides the ``quant_info`` slot like any other fused backend's payload;
it carries no quantization data.
"""

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
    """``runner`` is the layer's bound runner (resolved once at layer init);
    ``batch`` is this forward's weight view. Both are borrowed references."""

    runner: MoeLoraRunner
    batch: MoeLoraBatch


@register_fused_func("none", "lora")
def fused_experts_none_to_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeLoraDispatchPayload,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    del runner_config
    return quant_info.runner.run(dispatch_output, quant_info.batch)

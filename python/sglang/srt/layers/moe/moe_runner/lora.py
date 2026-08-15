"""Runner registration for the ``lora`` MoE backend.

This module is the entry point only — it registers the backend's fused func
and defines the payload that carries state across the boundary. The engine
itself lives in ``sglang/srt/lora/moe``.

The backend owns the whole expert computation from dispatch to combine, so
it plugs in like every other fused backend: one ``@register_fused_func``
entry with the standard
``(dispatch_output, quant_info, runner_config) -> CombineInput`` signature.

There are deliberately no LoRA *hooks* here. Hooks exist so an engine can
interpose LoRA into someone else's expert pipeline; this backend is itself
the LoRA implementation and injects its deltas inside its own pipeline
(gate/up delta consumed pre-activation, down delta folded into the weighted
combine), so it needs no interposition.

Per-forward LoRA state rides in the runner's ``quant_info`` slot, exactly as
other backends carry their per-layer payloads there. It holds no
quantization data: resident weights are described by the provider payloads in
``sglang/srt/lora/moe/quant_info.py``, bound once when the layer attaches, and
quantized activations arrive on ``dispatch_output``.
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
    from sglang.srt.lora.moe.config_backend import MoeLoraConfigBackend
    from sglang.srt.lora.moe.moe_lora_runner import MoeLoraBatch


@dataclass
class MoeLoraDispatchPayload(MoeQuantInfo):
    """Per-forward payload for the ``lora`` runner backend.

    ``config_backend`` is the layer's bound engine (providers, runners and
    launch configuration are constructed once at attach); ``batch`` is this
    forward's factor view. Both are borrowed references — constructing this
    struct copies no tensors.
    """

    config_backend: MoeLoraConfigBackend
    batch: MoeLoraBatch


@register_fused_func("none", "lora")
def fused_experts_none_to_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeLoraDispatchPayload,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run the MoE experts with LoRA fused into the pipeline."""
    backend = quant_info.config_backend
    batch = quant_info.batch
    choice = backend.select(batch, num_tokens=dispatch_output.hidden_states.shape[0])
    return backend.run_selected(
        choice,
        dispatch_output,
        batch,
    )

"""Resolve model-owned VPP layout before any layers or weights are allocated."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.distributed.vpp_layout import VPPLayout

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def resolve_vpp_layout(
    model_config: ModelConfig, parallel, *, is_draft_worker: bool = False
) -> VPPLayout | None:
    if parallel.pp_virtual_stages == 1:
        return None
    if is_draft_worker:
        raise ValueError("VPP draft model loading is not supported")
    if model_config.hf_config.architectures != ["DeepseekV4ForCausalLM"]:
        raise ValueError("VPP currently supports only DeepseekV4ForCausalLM")
    if model_config.hf_config.model_type != "deepseek_v4":
        raise ValueError("VPP currently supports DeepSeek V4; V4.1 is not wired")
    if not model_config.is_generation or model_config.is_multimodal:
        raise ValueError("VPP requires a text generation model")
    return VPPLayout(
        num_hidden_layers=model_config.num_hidden_layers,
        physical_size=parallel.pp_size,
        virtual_stages=parallel.pp_virtual_stages,
        partition=parallel.pp_vpp_layer_partition,
    )

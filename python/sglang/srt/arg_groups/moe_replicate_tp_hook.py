# SPDX-License-Identifier: Apache-2.0
"""Topology checks for --moe-replicate-tp: replicated MoE with prefill sequence
parallelism over the TP group."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolved_view,
    resolving_view,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import attn_dp_enabled_of

# Architectures whose decoder layers bind the replicated-MoE sequence-parallel
# batch variant; every MoE layer must be sparse and use Qwen2MoeSparseMoeBlock.
_SUPPORTED_MODELS: set[str] = {
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
}


def resolve_moe_replicate_tp(server_args: Any, model: Any) -> None:
    cfg, view = resolving_view(server_args), resolved_view(server_args)
    if not cfg.moe_replicate_tp:
        return
    architecture = model.hf_config.architectures[0]
    if architecture not in _SUPPORTED_MODELS:
        raise ValueError(
            f"--moe-replicate-tp supports {sorted(_SUPPORTED_MODELS)}; got "
            f"{architecture}."
        )
    if cfg.tp_size <= 1:
        raise ValueError("--moe-replicate-tp requires --tp-size > 1.")
    unsupported = {
        "moe-dp-size other than tp-size": cfg.moe_dp_size not in (1, cfg.tp_size),
        "ep-size > 1": view.ep_size > 1,
        "attn-cp-size > 1": view.attn_cp_size > 1,
        "enable-dp-attention": attn_dp_enabled_of(cfg),
        "pp-size > 1": cfg.pp_size > 1,
        "dwdp-size > 1": cfg.dwdp_size > 1,
        "moe-a2a-backend": view.moe_a2a_backend != "none",
        "enable-layernorm-sp": cfg.enable_layernorm_sp,
        "enable-attn-tp-input-scattered": cfg.enable_attn_tp_input_scattered,
        "enable-two-batch-overlap": cfg.enable_two_batch_overlap,
    }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with --moe-replicate-tp.")
    declare_resolution(
        server_args,
        "resolve_moe_replicate_tp",
        moe_dp_size=cfg.tp_size,
        # The sequence collectives have batch-dependent sizes.
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )

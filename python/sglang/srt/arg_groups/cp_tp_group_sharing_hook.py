# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for CP-TP group sharing: prefill CP
for hybrid linear-attention models where the CP group is the TP group."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolved_view,
    resolving_view,
)
from sglang.srt.configs.hybrid_arch import mambaish_config

# Architectures whose model code implements CP-TP group sharing: the CP group
# is the TP group, the residual stream / attention / indexer are CP-sharded,
# MoE and linear attention keep their TP partition (the linear attention
# gathers the sequence where its recurrence needs it). Model ports register
# here; every other model keeps the ordinary prefill CP path.
_SUPPORTED_MODELS: set[str] = {"Glm5NextForConditionalGeneration"}


def resolve_cp_tp_group_sharing(server_args: Any, model: Any) -> None:
    cfg, view = resolving_view(server_args), resolved_view(server_args)
    if (
        not cfg.enable_prefill_cp
        or view.attn_cp_size <= 1
        or model.hf_config.architectures[0] not in _SUPPORTED_MODELS
    ):
        return
    # Attention DP or attention TP would leave the CP group narrower than TP.
    if view.attn_cp_size != cfg.tp_size:
        raise ValueError(
            "CP-TP group sharing requires --attn-cp-size == --tp-size, got "
            f"attn_cp_size={view.attn_cp_size}, tp_size={cfg.tp_size}."
        )
    strategy = (
        "interleave"
        if model.hf_config.architectures[0] == "Glm5NextForConditionalGeneration"
        else "zigzag"
    )
    if cfg.cp_strategy != strategy:
        raise ValueError(f"CP-TP group sharing requires --cp-strategy {strategy}.")

    # This hook runs before speculative defaults and NEXTN alias resolution.
    # Native MTP keeps the target's CP/TP placement. DFLASH consumes the
    # target's gathered hidden states and runs its draft with ordinary TP.
    if cfg.speculative_algorithm not in (None, "DFLASH") and (
        model.hf_config.architectures[0] != "Glm5NextForConditionalGeneration"
        or cfg.speculative_algorithm.upper() not in ("EAGLE", "NEXTN")
        or cfg.speculative_draft_model_path not in (None, cfg.model_path)
        or cfg.speculative_eagle_topk not in (None, 1)
    ):
        raise ValueError(
            "CP/TP group sharing only supports DFLASH or native MTP with top-k 1."
        )

    unsupported = {
        "enable-dsa-cache-layer-split": cfg.enable_dsa_cache_layer_split,
        "dcp-size > 1": cfg.dcp_size > 1,
        "enable-mixed-chunk": cfg.enable_mixed_chunk,
        "moe-dp-size > 1": cfg.moe_dp_size > 1,
        "disaggregation-mode": cfg.disaggregation_mode != "null",
    }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with CP-TP group sharing.")

    linear_config = mambaish_config(model)
    if getattr(linear_config, "linear_num_key_heads", None) is not None:
        heads = {
            name: getattr(linear_config, name)
            for name in ("linear_num_key_heads", "linear_num_value_heads")
        }
    else:
        heads = {"num_heads": linear_config.linear_attn_config["num_heads"]}
    # Linear attention keeps its TP head partition over the whole TP group.
    for name, count in heads.items():
        if count % cfg.tp_size:
            raise ValueError(
                f"{name}={count} must be divisible by --tp-size={cfg.tp_size}."
            )

    declare_resolution(
        server_args,
        "resolve_cp_tp_group_sharing",
        enable_cp_tp_group_sharing=True,
    )

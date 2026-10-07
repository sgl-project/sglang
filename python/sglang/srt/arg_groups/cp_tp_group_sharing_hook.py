# SPDX-License-Identifier: Apache-2.0
"""Resolve hybrid models whose prefill CP and tensor parallel groups coincide."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

# Each model must implement a CP-sharded decoder boundary and full-sequence
# linear attention before opting in. Other models retain their existing path.
_MODEL_STRATEGIES = {"Glm5NextForConditionalGeneration": "interleave"}


def resolve_cp_tp_group_sharing(server_args: Any, model_config: Any) -> None:
    cfg = resolving_view(server_args)
    if not cfg.enable_prefill_cp or cfg.attn_cp_size <= 1:
        return
    architecture = model_config.hf_config.architectures[0]
    strategy = _MODEL_STRATEGIES.get(architecture)
    if strategy is None:
        return
    if cfg.attn_dp_size != 1:
        raise ValueError("CP/TP group sharing does not support attention DP.")
    if cfg.tp_size != cfg.attn_cp_size:
        raise ValueError("CP/TP group sharing requires --tp-size == --attn-cp-size.")
    if cfg.cp_strategy != strategy:
        raise ValueError(
            f"{architecture} CP/TP group sharing requires --cp-strategy {strategy}."
        )

    unsupported = {
        "speculative-algorithm": cfg.speculative_algorithm is not None,
        "dcp-size > 1": cfg.dcp_size > 1,
        "enable-mixed-chunk": cfg.enable_mixed_chunk,
        "moe-dp-size > 1": cfg.moe_dp_size > 1,
        "enable-dsa-cache-layer-split": cfg.enable_dsa_cache_layer_split,
    }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with CP/TP group sharing.")
    heads = model_config.hf_text_config.linear_attn_config["num_heads"]
    if heads % cfg.tp_size:
        raise ValueError(
            f"num_heads={heads} must be divisible by --tp-size={cfg.tp_size}."
        )

    declare_resolution(
        server_args,
        "resolve_cp_tp_group_sharing",
        cp_tp_group_sharing=True,
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )

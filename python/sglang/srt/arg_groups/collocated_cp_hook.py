# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for collocated prefill CP (hybrid
linear-attention models; the CP group is the TP group)."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.connector import ConnectorType
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.utils.common import parse_connector_type

# Architectures whose model code implements the collocated prefill CP contract:
# the CP group is the TP group, the residual stream / attention / indexer are
# CP-sharded, MoE and linear attention keep their TP partition (the linear
# attention gathers the sequence where its recurrence needs it). Model ports
# register here.
_SUPPORTED_MODELS: set[str] = set()


def resolve_collocated_cp(server_args: Any) -> None:
    cfg, view = resolving_view(server_args), resolved_view(server_args)
    if not cfg.enable_prefill_cp or view.attn_cp_size <= 1:
        return
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return

    model = model_config_of(server_args)
    linear_config = mambaish_config(model)
    if linear_config is None:
        return
    architecture = model.hf_config.architectures[0]
    if architecture not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Collocated prefill CP is not integrated with {architecture}. "
            f"Supported models: {sorted(_SUPPORTED_MODELS) or 'none'}."
        )
    if cfg.cp_strategy != "zigzag":
        raise ValueError("Collocated prefill CP requires --cp-strategy zigzag.")

    unsupported = {
        "speculative-algorithm": cfg.speculative_algorithm is not None,
        "dcp-size > 1": cfg.dcp_size > 1,
        "enable-mixed-chunk": cfg.enable_mixed_chunk,
        "enable-dp-attention": view.enable_dp_attention,
        "moe-dp-size > 1": cfg.moe_dp_size > 1,
    }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with collocated prefill CP.")

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

    if cfg.disaggregation_mode != "null":
        raise ValueError(
            "Collocated prefill CP with PD disaggregation is not supported."
        )

    declare_resolution(
        server_args,
        "resolve_collocated_cp",
        enable_collocated_cp=True,
        # The sequence collectives have batch-dependent sizes.
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )

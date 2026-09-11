# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for linear-attention prefill CP."""

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

_QWEN4_EXP_MODELS = {"Qwen4ExpForConditionalGeneration"}
# Architectures whose model code implements the collocated linear-attention CP
# contract (fold the CP ranks into the linear-attention head partition, gather
# the sequence where the recurrence needs it). Model ports register here.
_SUPPORTED_MODELS: set[str] = set(_QWEN4_EXP_MODELS)


def resolve_linear_attn_cp(server_args: Any) -> None:
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
            f"Linear-attention prefill CP is not integrated with {architecture}. "
            f"Supported models: {sorted(_SUPPORTED_MODELS) or 'none'}."
        )
    if cfg.cp_strategy != "zigzag":
        raise ValueError("Linear-attention prefill CP requires --cp-strategy zigzag.")

    unsupported = {
        "speculative-algorithm": cfg.speculative_algorithm is not None,
        "dcp-size > 1": cfg.dcp_size > 1,
        "enable-mixed-chunk": cfg.enable_mixed_chunk,
        "enable-dp-attention": view.enable_dp_attention,
        "moe-dp-size > 1": cfg.moe_dp_size > 1,
    }
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with linear-attention CP.")

    if architecture in _QWEN4_EXP_MODELS:
        # Qwen4-Exp keeps the residual stream sequence-sharded for the whole
        # layer stack (SP), which requires the CP group to be the TP group.
        if view.attn_cp_size != cfg.tp_size:
            raise ValueError(
                "Qwen4-Exp prefill CP requires --attn-cp-size == --tp-size "
                f"(collocated CP), got attn_cp_size={view.attn_cp_size}, "
                f"tp_size={cfg.tp_size}."
            )
        for flag, enabled in (
            ("ep-size > 1", view.ep_size > 1),
            ("pp-size > 1", cfg.pp_size > 1),
            # A2A resolution runs later and can promote ep_size to tp_size; a
            # non-none a2a backend would also disable the SP residual stream
            # (sp_cp_static_enabled) while the runner keeps the CP path.
            ("moe-a2a-backend other than none", view.moe_a2a_backend != "none"),
            ("chunked-prefill-size > 0", cfg.chunked_prefill_size > 0),
            ("radix cache", not cfg.disable_radix_cache),
        ):
            if enabled:
                raise ValueError(f"{flag} is not supported with Qwen4-Exp prefill CP.")
    if getattr(linear_config, "linear_num_key_heads", None) is not None:
        heads = {
            name: getattr(linear_config, name)
            for name in ("linear_num_key_heads", "linear_num_value_heads")
        }
    else:
        heads = {"num_heads": linear_config.linear_attn_config["num_heads"]}
    # With attention DP disabled, folding CP into the head partition uses all TP ranks.
    for name, count in heads.items():
        if count % cfg.tp_size:
            raise ValueError(
                f"{name}={count} must be divisible by --tp-size={cfg.tp_size}."
            )

    if cfg.disaggregation_mode != "null":
        raise ValueError("Linear-attention CP with PD disaggregation is not supported.")

    declare_resolution(
        server_args,
        "resolve_linear_attn_cp",
        enable_linear_attn_cp=True,
        # The sequence collectives have batch-dependent sizes.
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )

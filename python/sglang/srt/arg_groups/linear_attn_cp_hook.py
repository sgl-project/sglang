# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for linear-attention prefill CP."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import attention_backends_of
from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.connector import ConnectorType
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import parse_connector_type

_KIMI_K3_MODELS = {
    "KimiK3LinearForCausalLM",
    "KimiK3ForConditionalGeneration",
}
_SUPPORTED_MODELS = {"Qwen3NextForCausalLM", "KimiLinearForCausalLM"} | _KIMI_K3_MODELS


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
            "Supported models: Qwen3-Next, Kimi-Linear, and Kimi-K3."
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

    if architecture in _KIMI_K3_MODELS:
        for flag, enabled in (
            ("ep-size > 1", view.ep_size > 1),
            ("pp-size > 1", cfg.pp_size > 1),
            ("enable-attn-tp-input-scattered", cfg.enable_attn_tp_input_scattered),
            # A2A resolution runs later and can promote ep_size to tp_size.
            ("moe-a2a-backend other than none", view.moe_a2a_backend != "none"),
            ("enable-waterfill", cfg.enable_waterfill),
        ):
            if enabled:
                raise ValueError(f"--{flag} is not supported with Kimi-K3 prefill CP.")
        prefill_backend, _ = attention_backends_of(view)
        if prefill_backend not in ("fa3", "fa4"):
            raise ValueError(
                "Kimi-K3 prefill CP requires --prefill-attention-backend fa3 or fa4 "
                "(or --attention-backend fa3 or fa4) for absorbed MLA."
            )

        if prefill_backend == "fa4":
            if not get_platform().is_sm100_or_sm110:
                raise ValueError(
                    "Kimi-K3 prefill CP with fa4 requires SM100/SM110 for absorbed MLA."
                )
            # FA4's absorbed kernel accepts 512 latent dimensions and 64
            # unabsorbed Q/K dimensions, including K3's unrotated NoPE path.
            if (
                linear_config.kv_lora_rank != 512
                or linear_config.qk_rope_head_dim != 64
            ):
                raise ValueError(
                    "Kimi-K3 prefill CP with fa4 requires kv_lora_rank=512 "
                    "and qk_rope_head_dim=64."
                )

    if architecture == "Qwen3NextForCausalLM":
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

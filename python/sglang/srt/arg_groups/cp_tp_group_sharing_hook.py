# SPDX-License-Identifier: Apache-2.0
"""Model integration and topology checks for CP-TP group sharing: prefill CP
for hybrid linear-attention models where the CP group is the TP group."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform

# Architectures whose model code implements CP-TP group sharing: the CP group
# is the TP group, the residual stream / attention / indexer are CP-sharded,
# MoE and linear attention keep their TP partition (the linear attention
# gathers the sequence where its recurrence needs it). Model ports register
# here; every other model keeps the ordinary prefill CP path.
_KIMI_LINEAR_MODELS: set[str] = {
    "KimiLinearForCausalLM",
    "KimiK3LinearForCausalLM",
    "KimiK3ForConditionalGeneration",
}
_SUPPORTED_MODELS: set[str] = {
    "Glm5NextForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
    "KimiK31ForConditionalGeneration",
} | _KIMI_LINEAR_MODELS
# Models whose DSA layers run interleave CP; the others use zigzag.
_INTERLEAVE_MODELS: set[str] = {
    "Glm5NextForConditionalGeneration",
    "KimiK31ForConditionalGeneration",
}


def resolve_cp_tp_group_sharing(server_args: Any, model: Any) -> None:
    cfg, view = resolving_view(server_args), resolved_view(server_args)
    if not cfg.enable_prefill_cp or view.attn_cp_size <= 1:
        return
    arch = model.hf_config.architectures[0]
    if arch not in _SUPPORTED_MODELS:
        if mambaish_config(model) is not None:
            raise ValueError(
                f"Linear-attention prefill CP is not integrated with {arch}."
            )
        return
    is_kimi_linear = arch in _KIMI_LINEAR_MODELS
    # Attention DP or attention TP would leave the CP group narrower than TP.
    if view.attn_cp_size != cfg.tp_size:
        raise ValueError(
            "CP-TP group sharing requires --attn-cp-size == --tp-size, got "
            f"attn_cp_size={view.attn_cp_size}, tp_size={cfg.tp_size}."
        )
    strategy = "interleave" if arch in _INTERLEAVE_MODELS else "zigzag"
    if cfg.cp_strategy != strategy:
        raise ValueError(f"CP-TP group sharing requires --cp-strategy {strategy}.")

    # This hook runs before speculative defaults and NEXTN alias resolution.
    # Native MTP keeps the target's CP/TP placement. DFLASH consumes the
    # target's gathered hidden states and runs its draft with ordinary TP.
    # Both integrated and PD prefill consume the CP-gathered target states.
    if cfg.speculative_algorithm is not None:
        if arch == "KimiK31ForConditionalGeneration":
            if cfg.speculative_algorithm.upper() != "DFLASH":
                raise ValueError(
                    "Kimi-K3.1 CP-TP group sharing only supports "
                    "--speculative-algorithm DFLASH."
                )
        elif arch != "Glm5NextForConditionalGeneration":
            raise ValueError(
                "--speculative-algorithm is not supported with CP-TP group sharing."
            )
        elif cfg.speculative_algorithm != "DFLASH" and (
            cfg.speculative_algorithm.upper() not in ("EAGLE", "NEXTN")
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
    }
    if arch == "Qwen4ExpForConditionalGeneration" or is_kimi_linear:
        unsupported |= {
            # The MoE runs TP experts on rows gathered over the TP group.
            # Waterfill turns on DeepEP and EP after this hook.
            "ep-size > 1": view.ep_size > 1,
            "moe-a2a-backend": view.moe_a2a_backend != "none",
            "enable-waterfill": cfg.enable_waterfill,
            "pp-size > 1": cfg.pp_size > 1,
            # Prefill only: a decode node serves plain TP.
            "disaggregation-mode decode": cfg.disaggregation_mode == "decode",
        }
        if is_kimi_linear:
            unsupported |= {
                "attn-dp-size > 1": view.attn_dp_size > 1,
                "disaggregation-mode != null": cfg.disaggregation_mode != "null",
            }
    elif arch == "Glm5NextForConditionalGeneration":
        if cfg.disaggregation_mode != "null":
            unsupported |= {
                "disaggregation-mode decode": cfg.disaggregation_mode == "decode",
                "disaggregation-transfer-backend != mooncake": (
                    cfg.disaggregation_transfer_backend != "mooncake"
                ),
                "speculative-algorithm != DFLASH": cfg.speculative_algorithm
                != "DFLASH",
                "pp-size > 1": cfg.pp_size > 1,
                "dp-size > 1": cfg.dp_size > 1,
                "ep-size > 1": view.ep_size > 1,
            }
    elif arch == "KimiK31ForConditionalGeneration":
        # Prefill only: the decode node serves TP (+ DCP).
        unsupported["disaggregation-mode decode"] = cfg.disaggregation_mode == "decode"
    for flag, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"--{flag} is not supported with CP-TP group sharing.")
    # Decode sees this prefill as TP cp_size, while its sender stages by
    # attention TP 1, so the two would pick different staging layouts.
    if cfg.disaggregation_mode == "prefill" and envs.SGLANG_DISAGG_STAGING_BUFFER.get():
        raise ValueError(
            "SGLANG_DISAGG_STAGING_BUFFER is not supported with CP-TP group sharing."
        )

    linear_config = mambaish_config(model)
    if arch in ("KimiK3LinearForCausalLM", "KimiK3ForConditionalGeneration"):
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
            if (
                linear_config.kv_lora_rank != 512
                or linear_config.qk_rope_head_dim != 64
            ):
                raise ValueError(
                    "Kimi-K3 prefill CP with fa4 requires kv_lora_rank=512 "
                    "and qk_rope_head_dim=64."
                )
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

    fields = {"enable_cp_tp_group_sharing": True}
    if arch == "Qwen4ExpForConditionalGeneration" or is_kimi_linear:
        # These sequence collectives still have batch-dependent sizes.
        fields["cuda_graph_config"] = with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        )
    declare_resolution(server_args, "resolve_cp_tp_group_sharing", **fields)

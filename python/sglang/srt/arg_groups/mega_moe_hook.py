from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.connector import ConnectorType
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import parse_connector_type

logger = logging.getLogger(__name__)


def handle_mega_moe(server_args: ServerArgs) -> None:
    handle_moe_runner_backend_alias(server_args)
    check_mega_moe_compat(server_args)


def check_mega_moe_compat(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        # The fused kernel has no dispatch_a / combine_a split for TBO / SBO.
        raise ValueError(
            "--moe-a2a-backend megamoe has no two-batch / single-batch overlap "
            "decomposition; disable --enable-two-batch-overlap and "
            "--enable-single-batch-overlap."
        )
    platform = get_platform()
    if not (platform.is_cuda and (platform.is_sm90 or platform.is_sm100)):
        raise ValueError(
            "--moe-a2a-backend megamoe needs a CUDA SM90 GPU (block-FP8 experts) "
            "or an SM100-class GPU (MXFP4 / NVFP4 experts); it runs DeepGEMM "
            "kernels over CUDA symmetric memory."
        )


# MoE blocks with no fused-MoE fallback under megamoe.
MEGA_MOE_NO_FALLBACK_ARCHS = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "InternS2PreviewForConditionalGeneration",
        "MellumForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def mega_moe_needs_token_budget(
    model_arch: str, hf_quant_config: dict, quantization: str | None
) -> bool:
    # NVFP4 experts are repacked into the mega layout at load: no fallback.
    nvfp4_experts = quantization == "modelopt_fp4" or "FP4" in str(
        hf_quant_config.get("quant_algo", "")
    )
    return model_arch in MEGA_MOE_NO_FALLBACK_ARCHS or nvfp4_experts


def validate_mega_moe_token_budget_for_model(server_args: ServerArgs) -> None:
    # Runs after speculative resolution so the decode / verify bound is final.
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return
    if parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE:
        return
    model_config = model_config_of(server_args)
    model_arch = model_config.hf_config.architectures[0]
    if mega_moe_needs_token_budget(
        model_arch, model_config.hf_quant_config, cfg.quantization
    ):
        validate_mega_moe_token_budget(server_args, model_arch)


def mega_moe_decode_tokens_per_rank(cfg) -> int:
    cg_config = cfg.cuda_graph_config
    decode_max_bs = (cg_config.decode.max_bs if cg_config is not None else 0) or 0
    num_tokens_per_req = (
        (cfg.speculative_num_draft_tokens or 1) if cfg.speculative_algorithm else 1
    )
    return decode_max_bs * num_tokens_per_req


def validate_mega_moe_token_budget(server_args: ServerArgs, model_label: str) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return

    max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    if cfg.disaggregation_mode != "prefill":
        decode_tokens = mega_moe_decode_tokens_per_rank(cfg)
        if max_tokens_per_rank < decode_tokens:
            raise ValueError(
                f"{model_label} with MegaMoE requires "
                "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to cover the "
                "largest decode / verify forward on one rank. Current values: "
                f"decode cuda graph max bs x tokens per request = {decode_tokens}, "
                f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
                f"{max_tokens_per_rank}. Raise the env var to at least "
                f"{decode_tokens} or lower --cuda-graph-max-bs / "
                "--speculative-num-draft-tokens."
            )
    if cfg.disaggregation_mode == "decode":
        # decode node will skip the check because decode bs is not relevant with --chunk-prefill-size
        return

    if cfg.pp_size > 1 and cfg.enable_dynamic_chunking:
        return

    if cfg.chunked_prefill_size is None or cfg.chunked_prefill_size <= 0:
        raise ValueError(
            f"{model_label} with MegaMoE requires chunked prefill to be enabled. "
            "Set --chunked-prefill-size to a positive value; "
            "--chunked-prefill-size=-1 is unsafe because MegaMoE's per-rank "
            "token requirement would not have a strict prefill-forward bound."
        )

    if cfg.enable_prefill_cp:
        token_partition_size = cfg.attn_cp_size
        token_partition_name = "attn_cp_size"
        token_alignment = 1
        local_chunked_prefill_size = (
            cfg.chunked_prefill_size + token_partition_size - 1
        ) // token_partition_size
    elif cfg.enable_dp_attention:
        token_partition_size = cfg.dp_size
        token_partition_name = "dp_size"
        token_alignment = max(
            cfg.tp_size // cfg.dp_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = cfg.chunked_prefill_size // token_partition_size
    else:
        # Pure TP and PP with static chunking are handled here.
        token_partition_size = 1
        token_partition_name = "none"
        # global_num_tokens will ceil_align to attn_tp_size so the validation needs to do alignment as well
        token_alignment = max(
            cfg.tp_size // token_partition_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = cfg.chunked_prefill_size

    if local_chunked_prefill_size <= 0:
        raise ValueError(
            f"{model_label} with MegaMoE requires a positive effective per-rank "
            "chunked prefill size. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}."
        )

    required_tokens_per_rank = (
        (local_chunked_prefill_size + token_alignment - 1)
        // token_alignment
        * token_alignment
    )
    if max_tokens_per_rank < required_tokens_per_rank:
        raise ValueError(
            f"{model_label} with MegaMoE requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to "
            "cover each rank's effective prefill token budget. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}, "
            f"token_alignment={token_alignment}, "
            f"required_per_rank={required_tokens_per_rank}, "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
            f"{max_tokens_per_rank}. Set "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to at "
            f"least {required_tokens_per_rank}, or lower "
            "--chunked-prefill-size until the effective per-rank budget fits."
        )


def handle_moe_runner_backend_alias(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "megamoe":
        return

    if cfg.moe_a2a_backend not in ("none", "megamoe"):
        logger.warning(
            "--moe-runner-backend megamoe is an alias for "
            "--moe-a2a-backend megamoe; overriding "
            "--moe-a2a-backend %s.",
            cfg.moe_a2a_backend,
        )
    declare_resolution(
        server_args,
        "handle_moe_runner_backend_alias",
        moe_runner_backend="auto",
        moe_a2a_backend="megamoe",
    )

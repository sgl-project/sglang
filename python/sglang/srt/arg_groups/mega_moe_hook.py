from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def handle_mega_moe(server_args: ServerArgs) -> None:
    handle_moe_runner_backend_alias(server_args)
    check_mega_moe_compat(server_args)


def check_mega_moe_compat(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        # The mega kernel fuses dispatch, experts and combine, so there is no
        # dispatch_a / combine_a split for the TBO / SBO decomposition to call.
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


def validate_mega_moe_token_budget(server_args: ServerArgs, model_label: str) -> None:
    """Ensure the per-rank prefill budget fits MegaMoE's symmetric buffer.

    Called from the model-specific stage (chunked_prefill_size is resolved by
    then) for every architecture whose MoE block runs MegaMoE without a
    fallback path, so a too-small buffer fails at startup instead of on the
    first large prefill.
    """
    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe" or cfg.disaggregation_mode == "decode":
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
    max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
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

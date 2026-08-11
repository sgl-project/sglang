from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_DSV4_FLASH_COMPRESS_RATIOS = (0, 0) + (4, 128) * 20 + (4, 0)
_DSV4_FLASH_0731_COMPRESS_RATIOS = _DSV4_FLASH_COMPRESS_RATIOS + (0, 0)
_DSV4_SHARED_SUPPORTED_FLASH_PROFILES = {
    (43, _DSV4_FLASH_COMPRESS_RATIOS),
    (43, _DSV4_FLASH_0731_COMPRESS_RATIOS),
}


def is_dsv4_cache_shared_enabled(model_runner: ModelRunner) -> bool:
    """Whether DeepSeek V4 persistent cache pages are shared across CP ranks."""
    from sglang.srt.configs.model_config import is_deepseek_v4

    return (
        not model_runner.is_draft_worker
        and model_runner.server_args.enable_dsa_shared_kv_cache
        and is_deepseek_v4(model_runner.model_config.hf_config)
    )


def get_dsv4_shared_info(model_runner: ModelRunner) -> tuple[int | None, int]:
    if not is_dsv4_cache_shared_enabled(model_runner):
        return None, 1
    shared_size = get_parallel().attn_cp_size
    if shared_size <= 1:
        return None, 1
    return get_parallel().attn_cp_rank, shared_size


def validate_deepseek_v4_shared_target(
    server_args: ServerArgs, hf_config, model_arch: str
) -> None:
    """Reject model targets outside the DSV4 Shared release."""
    from sglang.srt.configs.model_config import is_deepseek_v4
    from sglang.srt.utils import is_cuda

    if server_args.enable_dsa_cache_layer_split:
        raise ValueError(
            "--enable-dsa-shared-kv-cache and "
            "--enable-dsa-cache-layer-split cannot be enabled together."
        )
    if not is_cuda():
        raise ValueError("--enable-dsa-shared-kv-cache requires NVIDIA CUDA.")
    if not is_deepseek_v4(hf_config):
        raise ValueError(
            "--enable-dsa-shared-kv-cache is currently supported only "
            "for DeepSeek V4."
        )
    if model_arch != "DeepseekV4ForCausalLM":
        raise ValueError(
            "DeepSeek V4 Shared KV release supports only the canonical "
            "DeepseekV4ForCausalLM target architecture, not NextN or DSpark."
        )


def validate_deepseek_v4_shared_release(server_args: ServerArgs, hf_config) -> None:
    """Reject configurations outside the validated Flash Prefill L1 release."""
    from sglang.srt.model_executor.cuda_graph_config import Backend

    ratios = tuple(getattr(hf_config, "compress_ratios", ()))
    layer_count = int(getattr(hf_config, "num_hidden_layers", len(ratios)))
    if (layer_count, ratios) not in _DSV4_SHARED_SUPPORTED_FLASH_PROFILES:
        profile = (
            layer_count,
            len(ratios),
            sum(ratio == 0 for ratio in ratios),
            sum(ratio == 4 for ratio in ratios),
            sum(ratio == 128 for ratio in ratios),
        )
        raise ValueError(
            "DeepSeek V4 Shared KV release currently supports only the exact "
            "43-layer Flash profiles with 44 or 46 config entries; "
            f"got layers/ratios/C1/C4/C128 = {profile}."
        )
    if not envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
        raise ValueError("DeepSeek V4 Shared KV release requires Compressor V2.")
    if envs.SGLANG_OPT_USE_OLD_COMPRESSOR.get():
        raise ValueError(
            "DeepSeek V4 Shared KV release does not support the old compressor."
        )
    if server_args.disaggregation_mode not in ("null", "prefill"):
        raise ValueError(
            "DeepSeek V4 --enable-dsa-shared-kv-cache is supported on the "
            "Prefill worker only; Decode PD workers use the ordinary DSV4 cache."
        )
    if (
        server_args.disaggregation_mode == "prefill"
        and server_args.disaggregation_transfer_backend != "mooncake"
    ):
        raise ValueError(
            "DeepSeek V4 Shared KV Prefill PD currently requires the Mooncake "
            "transfer backend."
        )
    if server_args.tp_size != 8 or server_args.dp_size != 1:
        raise ValueError(
            "DeepSeek V4 Shared KV release requires the validated TP8/CP8 topology."
        )
    if server_args.enable_memory_saver:
        raise ValueError(
            "DeepSeek V4 Shared KV release does not support the memory saver "
            "release/resume lifecycle yet."
        )
    if not server_args.enable_prefill_cp or server_args.cp_strategy != "interleave":
        raise ValueError(
            "--enable-dsa-shared-kv-cache requires --enable-prefill-cp and "
            "--cp-strategy interleave."
        )
    if server_args.enable_hisparse:
        raise ValueError("--enable-dsa-shared-kv-cache is incompatible with HiSparse.")
    if server_args.speculative_algorithm is not None:
        raise ValueError(
            "--enable-dsa-shared-kv-cache does not support speculative decoding yet."
        )
    if server_args.pp_size > 1:
        raise ValueError(
            "--enable-dsa-shared-kv-cache is not supported with pipeline "
            "parallelism (pp_size > 1)."
        )
    if server_args.enable_hierarchical_cache:
        raise ValueError(
            "DeepSeek V4 --enable-dsa-shared-kv-cache does not support L2 "
            "hierarchical cache until owner-local backup and restore are implemented."
        )
    if server_args.hicache_storage_backend is not None:
        raise ValueError(
            "--enable-dsa-shared-kv-cache does not support an L3 storage backend yet."
        )
    if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        raise ValueError(
            "--enable-dsa-shared-kv-cache does not support online C128 compression yet."
        )
    if server_args.cuda_graph_config.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "DeepSeek V4 Shared KV does not support Prefill CUDA graph; "
            "the transient Demand-cache epoch must advance on every layer call."
        )


def validate_deepseek_v4_mega_moe_token_budget(
    server_args: ServerArgs,
) -> None:
    """Ensure the DSV4 prefill budget fits MegaMoE's per-rank buffer."""
    mega_moe_enabled = (
        server_args.moe_a2a_backend == "megamoe"
        or envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()
    )
    if not mega_moe_enabled or server_args.disaggregation_mode == "decode":
        # decode node will skip the check because decode bs is not relevant with --chunk-prefill-size
        return

    if server_args.pp_size > 1 and server_args.enable_dynamic_chunking:
        return

    if (
        server_args.chunked_prefill_size is None
        or server_args.chunked_prefill_size <= 0
    ):
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires chunked prefill to be enabled. "
            "Set --chunked-prefill-size to a positive value; "
            "--chunked-prefill-size=-1 is unsafe because MegaMoE's per-rank "
            "token requirement would not have a strict prefill-forward bound."
        )

    if server_args.enable_prefill_cp:
        token_partition_size = server_args.attn_cp_size
        token_partition_name = "attn_cp_size"
        token_alignment = 1
        local_chunked_prefill_size = (
            server_args.chunked_prefill_size + token_partition_size - 1
        ) // token_partition_size
    elif server_args.enable_dp_attention:
        token_partition_size = server_args.dp_size
        token_partition_name = "dp_size"
        token_alignment = max(
            server_args.tp_size // server_args.dp_size // server_args.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = (
            server_args.chunked_prefill_size // token_partition_size
        )
    else:
        # Pure TP and PP with static chunking are handled here.
        token_partition_size = 1
        token_partition_name = "none"
        # global_num_tokens will ceil_align to attn_tp_size so the validation needs to do alignment as well
        token_alignment = max(
            server_args.tp_size // token_partition_size // server_args.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = server_args.chunked_prefill_size

    if local_chunked_prefill_size <= 0:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires a positive effective per-rank "
            "chunked prefill size. "
            f"Current values: chunked_prefill_size="
            f"{server_args.chunked_prefill_size}, "
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
            "DeepSeekV4 with MegaMoE requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to "
            "cover each rank's effective prefill token budget. "
            f"Current values: chunked_prefill_size="
            f"{server_args.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}, "
            f"token_alignment={token_alignment}, "
            f"required_per_rank={required_tokens_per_rank}, "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
            f"{max_tokens_per_rank}. Set "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to at "
            f"least {required_tokens_per_rank}, or lower "
            "--chunked-prefill-size until the effective per-rank budget fits. "
            "Otherwise MegaMoE falls back to the fused MoE path at runtime."
        )


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Residual imperative arm of the DeepSeek V4 defaults.

    The attention/page/window/MoE-runner declarations moved to the override
    registry (arg_groups/overrides.py: _deepseek_v4_overrides) and the
    kv-cache dtype default to the resolution pipeline
    (_deepseek_v4_kv_cache_dtype, invoked below at its legacy slot). This
    keeps, at the legacy slot: the ROCm env fill (env-write policy), the
    max_running_requests fill (the speculative hook is a later writer of
    that field) and the validations.
    """
    from sglang.srt.utils import is_hip

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there;
    if is_hip():
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.
    from sglang.srt.arg_groups.overrides import (
        _deepseek_v4_kv_cache_dtype,
        run_post_process_pass,
    )

    run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 256
        logger.warning(
            f"Setting max_running_requests to {server_args.max_running_requests} for {model_arch}."
        )

    if server_args.speculative_algorithm is not None:
        assert server_args.speculative_algorithm in (
            "EAGLE",
            "DSPARK",
        ), f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
        if server_args.speculative_algorithm == "EAGLE":
            assert (
                server_args.speculative_eagle_topk == 1
            ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    if not server_args.enable_prefill_cp:
        return

    if server_args.cp_strategy != "interleave":
        raise ValueError(
            "DeepSeekV4 only supports interleave CP strategy, "
            f"got {server_args.cp_strategy}"
        )

    server_args.enable_dsa_prefill_context_parallel = True
    server_args.enable_prefill_context_parallel = False
    server_args.dsa_prefill_cp_mode = "round-robin-split"
    server_args.enable_dp_attention = True
    server_args.moe_dense_tp_size = 1
    server_args.attn_cp_size = server_args.tp_size // server_args.dp_size
    assert (
        server_args.dp_size == 1
    ), "For round-robin split mode, dp attention is not supported."
    assert (
        server_args.tp_size <= 8
    ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
    if server_args.moe_a2a_backend not in ("none", "deepep", "megamoe"):
        raise ValueError(
            "DeepSeekV4 CP supports moe_a2a_backend in "
            "('none', 'deepep', 'megamoe'), "
            f"got {server_args.moe_a2a_backend!r}."
        )
    if (
        getattr(server_args, "enable_dsa_shared_kv_cache", False)
        and envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set()
    ):
        logger.warning(
            "Preserving the explicit SGLANG_OPT_FLASHMLA_SPARSE_PREFILL setting "
            "for DeepSeekV4 context parallelism."
        )
    else:
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default because "
            "DeepSeekV4 context parallelism is enabled."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"dp_size={server_args.dp_size}, moe_dense_tp_size={server_args.moe_dense_tp_size}, "
        f"attn_cp_size={server_args.attn_cp_size}, ep_size={server_args.ep_size}, tp_size={server_args.tp_size}"
    )

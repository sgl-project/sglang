from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


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

    ratios = list(getattr(hf_config, "compress_ratios", ()))
    layer_count = int(getattr(hf_config, "num_hidden_layers", len(ratios)))
    expected_ratios = [0, 0] + [4, 128] * 20 + [4, 0]
    if layer_count != 43 or ratios != expected_ratios:
        profile = (
            layer_count,
            len(ratios),
            sum(ratio == 0 for ratio in ratios),
            sum(ratio == 4 for ratio in ratios),
            sum(ratio == 128 for ratio in ratios),
        )
        raise ValueError(
            "DeepSeek V4 Shared KV release currently supports only the full "
            "43-layer Flash profile (44 config entries; C1/C4/C128 = 3/21/20); "
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
            "Prefill worker only; Decode must use the ordinary DSV4 cache."
        )
    if (
        server_args.disaggregation_mode == "prefill"
        and server_args.disaggregation_transfer_backend != "mooncake"
    ):
        raise ValueError(
            "DeepSeek V4 Shared KV Prefill PD currently requires the Mooncake "
            "transfer backend."
        )
    # DSV4 resolves attn_cp_size later in the adjustment pass. Validate the
    # canonical inputs that deterministically produce CP8 instead.
    if server_args.tp_size != 8 or server_args.dp_size != 1:
        raise ValueError(
            "DeepSeek V4 Shared KV release requires the validated TP8/CP8 topology."
        )
    if server_args.max_running_requests != 1:
        raise ValueError(
            "DeepSeek V4 Shared KV release requires --max-running-requests 1; "
            "batched staging is not yet validated without host synchronization."
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

    if getattr(server_args, "enable_dsa_shared_kv_cache", False):
        server_args.enable_prefill_context_parallel = False
    server_args.enable_dsa_prefill_context_parallel = True
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

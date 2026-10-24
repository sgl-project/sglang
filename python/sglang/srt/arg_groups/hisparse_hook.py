from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv", "flashinfer_sparse_mla"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "triton", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if get_platform().is_hip:
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if get_platform().is_hip:
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv", "flashinfer_sparse_mla"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use one of {sorted(allowed_backends)}, or omit the option "
            "to let SGLang pick a backend for this platform."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse_feature_combos(server_args: ServerArgs) -> None:

    cfg = resolving_view(server_args)
    # A mixed batch cannot merge the HiSparse decode batch.
    if cfg.enable_mixed_chunk:
        raise ValueError("--enable-hisparse does not support --enable-mixed-chunk yet.")


def validate_hisparse_v2_dsv41(server_args: ServerArgs) -> None:
    """DeepSeek-V4.1: the ratio-1/2 KV demotes to HiCache, SWA stays on GPU."""
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    from sglang.srt.mem_cache.sparsity.hisparse_v2_config import (
        resolve_hisparse_v2_config,
    )

    cfg = resolving_view(server_args)
    hf_config = model_config_of(server_args).hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    ratios = set(getattr(text_config, "compress_ratios", ()))
    if not ratios & {1, 2} or ratios - {0, 1, 2}:
        raise ValueError(
            "--enable-hisparse-v2 supports the DeepSeek-V4.1 ratio-1/2 layout only"
        )
    required = (
        (cfg.enable_hierarchical_cache, "--enable-hierarchical-cache"),
        (not cfg.disable_radix_cache, "radix cache enabled"),
        (cfg.hicache_write_policy == "write_back", "--hicache-write-policy write_back"),
        (cfg.hicache_mem_layout == "page_first", "--hicache-mem-layout page_first"),
        (cfg.hicache_host_memory_mode == "cache", "--hicache-host-memory-mode cache"),
        (get_platform().is_hip, "ROCm (the DeepSeek-V4 HIP attention backend)"),
    )
    for valid, requirement in required:
        if not valid:
            raise ValueError(f"--enable-hisparse-v2 requires {requirement}")
    enable_spec = (
        cfg.speculative_algorithm is not None
        and cfg.speculative_algorithm.upper() != "NONE"
    )
    unsupported = (
        (enable_spec, "speculative decoding"),
        (cfg.enable_mixed_chunk, "mixed prefill/decode batches"),
        (cfg.enable_session_radix_cache or cfg.enable_streaming_session, "sessions"),
        (cfg.disaggregation_mode != "null", "PD disaggregation"),
        (cfg.pp_size != 1 or cfg.attn_cp_size != 1 or cfg.dcp_size != 1, "PP or CP"),
        (cfg.enable_two_batch_overlap, "two-batch overlap"),
        (cfg.prefill_only_disable_kv_cache, "prefill without KV allocation"),
        (cfg.enable_unified_memory, "unified memory pools"),
        (cfg.enable_return_routed_experts, "routed-expert recording"),
        (cfg.enable_encoder_swa_bounded_replay, "encoder SWA bounded replay"),
        (is_unified_kv_triton(), "the unified-KV attention path"),
    )
    for enabled, feature in unsupported:
        if enabled:
            raise ValueError(f"--enable-hisparse-v2 does not support {feature} yet")
    config = resolve_hisparse_v2_config(
        config=cfg.hisparse_config,
        model_top_k=text_config.index_topk,
        hicache_ratio=cfg.hicache_ratio,
        page_size=cfg.page_size,
    )
    if config.block_size % 64:
        raise ValueError(
            "HiSparse V2 swap_in_block_size must be a multiple of the 64-lane wave"
        )


def validate_hisparse_v2(server_args: ServerArgs) -> None:
    from sglang.srt.configs.model_config import (
        get_dsa_index_kpool,
        is_deepseek_dsa,
        is_deepseek_v4,
        is_hybrid_swa_model,
    )

    cfg = resolving_view(server_args)
    hf_config = model_config_of(server_args).hf_config
    if cfg.enable_hisparse:
        raise ValueError(
            "--enable-hisparse and --enable-hisparse-v2 are mutually exclusive"
        )
    if is_deepseek_v4(hf_config):
        validate_hisparse_v2_dsv41(server_args)
        return
    if (
        not is_deepseek_dsa(hf_config)
        or get_dsa_index_kpool(hf_config) != 1
        or is_hybrid_swa_model(hf_config.architectures, hf_config)
    ):
        raise ValueError(
            "--enable-hisparse-v2 currently supports unpooled DSA models without hybrid SWA only"
        )
    required = (
        (cfg.enable_hierarchical_cache, "--enable-hierarchical-cache"),
        (not cfg.disable_radix_cache, "radix cache enabled"),
        (cfg.hicache_write_policy == "write_back", "--hicache-write-policy write_back"),
        (cfg.hicache_mem_layout == "page_first", "--hicache-mem-layout page_first"),
        (cfg.hicache_host_memory_mode == "cache", "--hicache-host-memory-mode cache"),
        (cfg.page_size == 64, "--page-size 64"),
    )
    for valid, requirement in required:
        if not valid:
            raise ValueError(f"--enable-hisparse-v2 requires {requirement}")
    unsupported = (
        (get_platform().is_hip or cfg.device != "cuda", "non-CUDA devices"),
        (cfg.enable_mixed_chunk, "mixed prefill/decode batches"),
        (cfg.enable_session_radix_cache or cfg.enable_streaming_session, "sessions"),
        (cfg.disaggregation_mode != "null", "PD disaggregation"),
        (cfg.pp_size != 1 or cfg.attn_cp_size != 1 or cfg.dcp_size != 1, "PP or CP"),
        (cfg.enable_two_batch_overlap, "two-batch overlap"),
        (cfg.prefill_only_disable_kv_cache, "prefill without KV allocation"),
        (cfg.enable_unified_memory, "unified memory pools"),
        (cfg.enable_return_routed_experts, "routed-expert recording"),
    )
    for enabled, feature in unsupported:
        if enabled:
            raise ValueError(f"--enable-hisparse-v2 does not support {feature} yet")
    validate_hisparse_kv_cache_dtype(server_args)
    from sglang.srt.configs.model_config import get_dsa_index_topk
    from sglang.srt.mem_cache.sparsity.hisparse_v2_config import (
        resolve_hisparse_v2_config,
        resolve_hisparse_v2_query_steps,
    )

    resolve_hisparse_v2_query_steps(
        algorithm=cfg.speculative_algorithm,
        num_draft_tokens=cfg.speculative_num_draft_tokens,
        eagle_topk=cfg.speculative_eagle_topk,
        adaptive=cfg.speculative_adaptive,
        multi_layer=cfg.enable_multi_layer_eagle,
    )
    resolve_hisparse_v2_config(
        config=cfg.hisparse_config,
        model_top_k=get_dsa_index_topk(hf_config),
        hicache_ratio=cfg.hicache_ratio,
        page_size=cfg.page_size,
    )
    for attr, label in (
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ):
        validate_hisparse_dsa_backend(server_args, attr, label)


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""

    cfg = resolving_view(server_args)
    if cfg.enable_hisparse_v2:
        validate_hisparse_v2(server_args)
    if not cfg.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
        is_minimax_sparse,
    )

    hf_config = model_config_of(server_args).hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_m3_hisparse = is_minimax_sparse(hf_config)
    is_hip = get_platform().is_hip
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse or is_m3_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5), DeepSeek V4, and MiniMax M3 now. "
    )

    assert cfg.disable_radix_cache, (
        "Hierarchical sparse attention currently requires --disable-radix-cache."
    )
    validate_hisparse_feature_combos(server_args)

    enable_spec = (
        cfg.speculative_algorithm is not None
        and cfg.speculative_algorithm.upper() != "NONE"
    )
    if enable_spec and (cfg.speculative_eagle_topk or 1) != 1:
        raise ValueError(
            "The minimal HiSparse spec integration supports linear speculation "
            "only; use --speculative-eagle-topk 1."
        )
    if enable_spec and cfg.speculative_adaptive:
        raise ValueError(
            "HiSparse speculative decoding uses statically sized swap state; "
            "--speculative-adaptive is not supported."
        )

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    # MiniMax M3 uses its own Triton sparse kernels.
    if is_m3_hisparse:
        return

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)

# SPDX-License-Identifier: Apache-2.0
"""Resolution and validation for logical-page KV cache sharding."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
    use_mla_backend,
)
from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import attn_dp_enabled_of
from sglang.srt.utils.common import get_device_memory_capacity

logger = logging.getLogger(__name__)


def validate_kv_shard_attention_backend(server_args: Any, cfg: Any = None) -> None:
    cfg = cfg or resolving_view(server_args)
    prefill_backend, _ = attention_backends_of(resolved_view(server_args))
    if prefill_backend == "fa3":
        return
    if prefill_backend != "trtllm_mla":
        raise ValueError(
            "--enable-kv-cache-sharding currently requires the fa3 prefill "
            "attention backend, or trtllm_mla for plain-TP MLA; "
            f"got {prefill_backend!r}."
        )
    if not use_mla_backend(server_args):
        raise ValueError("KV sharding with trtllm_mla requires an MLA model.")
    if cfg.attn_cp_size > 1:
        raise ValueError(
            "KV sharding with trtllm_mla only supports plain-TP MLA "
            "(attn_cp_size must be 1)."
        )
    if cfg.disable_chunked_prefix_cache:
        raise ValueError("KV sharding with trtllm_mla requires chunked prefix caching.")


def handle_kv_cache_sharding(server_args: Any, gpu_mem: Optional[float] = None) -> None:
    """Validate KV sharding and declare its derived eager-prefill settings."""
    cfg = resolving_view(server_args)
    if not cfg.enable_kv_cache_sharding:
        return

    if gpu_mem is None:
        # `handle_gpu_memory_settings` resolves capacity internally now, so the
        # pipeline no longer threads it through. Only the chunk-size rounding
        # below reads it, and only when mem_fraction_static was left unset.
        gpu_mem = get_device_memory_capacity(cfg.device)

    if cfg.disaggregation_mode != "prefill":
        raise ValueError(
            "--enable-kv-cache-sharding is only supported on PD prefill "
            "workers: a rank's pool holds only its stripe of every cached "
            "page, which is incompatible with local decode."
        )
    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "--enable-kv-cache-sharding does not support speculative decoding."
        )
    if cfg.dllm_algorithm is not None:
        raise ValueError(
            "--enable-kv-cache-sharding does not support diffusion language "
            "models: DLLM admission bypasses the shard-scratch budget and "
            "resolves out of PD-prefill mode."
        )
    if cfg.radix_cache_backend is not None:
        raise ValueError(
            "--enable-kv-cache-sharding requires the built-in "
            "UnifiedRadixCache with a rotation-aware UnifiedTreeCore and does "
            "not support "
            f"--radix-cache-backend={cfg.radix_cache_backend!r}."
        )
    tree_core_backend = envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get()
    # A disabled radix cache never matches or inserts, so its tree core never
    # sees a rotation base; chunked continuations keep the base on the request.
    if not cfg.disable_radix_cache and tree_core_backend != "python":
        raise ValueError(
            "--enable-kv-cache-sharding requires the Python UnifiedTreeCore: "
            f"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={tree_core_backend!r} is "
            "not supported for sharded KV."
        )
    if cfg.enable_unified_cache_external_linker:
        raise ValueError(
            "--enable-kv-cache-sharding does not support "
            "--enable-unified-cache-external-linker: load-back allocates "
            "through the flat allocator API."
        )
    if cfg.enable_streaming_session:
        raise ValueError(
            "--enable-kv-cache-sharding does not support streaming sessions: "
            "their virtual nodes do not carry KV-shard rotation metadata."
        )
    if cfg.enable_session_radix_cache:
        raise ValueError(
            "--enable-kv-cache-sharding does not support session radix cache yet."
        )
    if cfg.enable_flexkv:
        raise ValueError(
            "--enable-kv-cache-sharding does not support --enable-flexkv: "
            "FlexKV load-back allocates through the stock flat allocator API."
        )
    if cfg.enable_unified_memory:
        raise ValueError(
            "--enable-kv-cache-sharding does not support --enable-unified-memory: "
            "the unified pool uses a different virtual-id allocator and layout."
        )
    if cfg.dcp_size > 1:
        raise ValueError(
            "--enable-kv-cache-sharding and decode context parallelism "
            "(--dcp-size) are mutually exclusive users of the widened "
            "allocator index space."
        )
    if cfg.enable_hierarchical_cache:
        raise ValueError(
            "--enable-kv-cache-sharding does not support "
            "--enable-hierarchical-cache yet: HiCache backup/load assumes "
            "each rank holds full pages."
        )
    if cfg.enable_lmcache:
        raise ValueError(
            "--enable-kv-cache-sharding does not support --enable-lmcache."
        )
    if cfg.enable_hisparse:
        raise ValueError(
            "--enable-kv-cache-sharding does not support --enable-hisparse: "
            "HiSparse selects a non-sharded allocator and assumes each rank "
            "owns complete KV pages."
        )
    # The dynamic sizer only runs with pipeline parallelism. Its profiling
    # and predicted chunks can exceed the fixed shard-scratch capacity.
    if cfg.enable_dynamic_chunking and cfg.pp_size > 1:
        raise ValueError(
            "--enable-kv-cache-sharding does not support dynamic chunk sizing "
            "with pipeline parallelism: profiling and predicted chunks can "
            "exceed the fixed sharded scratch capacity. Disable "
            "--enable-dynamic-chunking."
        )
    if cfg.enable_two_batch_overlap:
        raise ValueError(
            "--enable-kv-cache-sharding does not support "
            "--enable-two-batch-overlap: the sharded pool's per-batch gather "
            "plan is a process-wide singleton, but TBO rebuilds attention "
            "metadata multiple times per extend forward and clobbers the plan."
        )

    model_config = model_config_of(server_args)
    if model_config.is_encoder_decoder:
        raise ValueError(
            "--enable-kv-cache-sharding does not support encoder-decoder "
            "models: cross-attention uses a separate encoder page table that "
            "is not part of the sharded gather plan."
        )
    if model_config.attention_chunk_size is not None:
        raise ValueError(
            "--enable-kv-cache-sharding does not support chunked/local-attention "
            "models yet: local-attention metadata is built from the page table "
            "before the scratch translation."
        )

    validate_kv_shard_attention_backend(server_args, cfg)
    if cfg.disaggregation_transfer_backend not in ("mooncake", "nixl"):
        raise ValueError(
            "--enable-kv-cache-sharding currently only supports the mooncake "
            "and nixl transfer backends: both route sends through "
            "CommonKVSender._prepare_send_indices and pair destinations by "
            "fancy index. Got --disaggregation-transfer-backend "
            f"{cfg.disaggregation_transfer_backend!r}."
        )
    if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
        raise ValueError(
            "--enable-kv-cache-sharding is incompatible with "
            "SGLANG_DISAGG_STAGING_BUFFER (the staging path assumes "
            "contiguous send slices)."
        )

    if use_mla_backend(server_args):
        if is_deepseek_dsa(model_config.hf_config):
            raise ValueError(
                "--enable-kv-cache-sharding does not support DSA models yet "
                "(indexer buffers are not striped)."
            )
        if cfg.attn_cp_size <= 1:
            if attn_dp_enabled_of(cfg):
                raise ValueError(
                    "--enable-kv-cache-sharding for MLA models without "
                    "prefill CP shards across attention-TP and requires plain "
                    "TP attention; set --attn-dp-size=1."
                )
            if cfg.tp_size <= 1:
                raise ValueError(
                    "--enable-kv-cache-sharding for MLA models without "
                    "prefill CP shards across attention-TP ranks and needs "
                    "tp_size > 1."
                )
    elif not cfg.enable_prefill_cp or cfg.attn_cp_size <= 1:
        raise ValueError(
            "--enable-kv-cache-sharding for GQA models shards across "
            "attention-CP ranks and requires --enable-prefill-cp with "
            "attn_cp_size > 1."
        )

    page_size = cfg.page_size if cfg.page_size is not None else 1
    if page_size <= 1:
        raise ValueError(
            "--enable-kv-cache-sharding requires a real page size "
            "(--page-size 64 recommended): the physical page is the P/D "
            "transfer descriptor and gather-copy unit; page_size=1 "
            "degenerates both into per-token operations."
        )
    chunk_size = cfg.chunked_prefill_size
    if chunk_size is None or chunk_size <= 0:
        raise ValueError(
            "--enable-kv-cache-sharding requires chunked prefill "
            "(--chunked-prefill-size > 0)."
        )

    declarations = {}
    # Chunk boundaries must stay on physical pages. Uneven ownership across
    # ranks is supported; only the prefix gather pads to a full-group span.
    if chunk_size % page_size != 0:
        rounded = (chunk_size + page_size - 1) // page_size * page_size
        declarations["chunked_prefill_size"] = rounded

        # The raw input, deliberately not the resolving view: the question is
        # whether the user pinned mem_fraction_static, and the resolved value
        # would answer yes for every run.
        raw_mem_fraction = (getattr(server_args, "_raw_input", None) or {}).get(
            "mem_fraction_static"
        )
        activation_delta_mb = 1.5 * (max(rounded, 2048) - max(chunk_size, 2048))
        if raw_mem_fraction is None and activation_delta_mb > 0:
            if not gpu_mem or cfg.mem_fraction_static is None:
                raise ValueError(
                    "Cannot safely auto-round --chunked-prefill-size for KV "
                    "sharding without a known device-memory capacity; pass a "
                    f"multiple of page_size ({page_size})."
                )
            corrected_mem_fraction = (
                float(cfg.mem_fraction_static) - activation_delta_mb / gpu_mem
            )
            if corrected_mem_fraction <= 0:
                raise ValueError(
                    "Auto-rounding --chunked-prefill-size for KV sharding "
                    "would consume all automatically reserved GPU memory; "
                    f"pass a smaller multiple of {page_size}."
                )
            declarations["mem_fraction_static"] = corrected_mem_fraction

        logger.warning(
            "Rounding --chunked-prefill-size up from %d to %d "
            "(a multiple of page_size = %d) for KV sharding.",
            chunk_size,
            rounded,
            page_size,
        )

    if cfg.cuda_graph_config.prefill.backend != Backend.DISABLED:
        logger.warning(
            "Prefill CUDA graph (%s) is incompatible with "
            "--enable-kv-cache-sharding; disabling it.",
            cfg.cuda_graph_config.prefill.backend,
        )
        declarations["cuda_graph_config"] = with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        )

    if declarations:
        declare_resolution(
            server_args,
            "handle_kv_cache_sharding",
            **declarations,
        )

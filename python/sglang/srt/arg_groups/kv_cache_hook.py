# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for KV-cache dtype and pool compatibility."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolved_view,
    resolving_view,
    use_mla_backend,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.utils.common import (
    is_blackwell_supported,
    is_cuda,
    is_sm100_supported,
    is_sm120_supported,
)

logger = logging.getLogger(__name__)


def handle_mxfp8_kv_cache_compatibility(server_args: Any) -> None:
    """MXFP8 KV cache uses operands available only on SM100+ (Blackwell)."""
    cfg = resolving_view(server_args)
    if cfg.kv_cache_dtype != "mxfp8":
        return
    if not is_blackwell_supported():
        raise ValueError(
            "--kv-cache-dtype mxfp8 requires an SM100+ (Blackwell) GPU for the "
            "block-scaled operands used by the FA4 MXFP8 attention path."
        )


def handle_kv4_compatibility(server_args: Any) -> None:
    """Check FP4 KV cache compatibility with the attention backend"""
    from sglang.srt.arg_groups.overrides import attention_backends_of

    cfg = resolving_view(server_args)

    if cfg.kv_cache_dtype not in ("nvfp4", "fp4_mx_block16"):
        return

    uses_mla = use_mla_backend(server_args)
    prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
    attention_backend = resolved_view(server_args).attention_backend

    if is_cuda():
        if cfg.kv_cache_dtype == "nvfp4" and not (
            is_sm100_supported() or is_sm120_supported()
        ):
            raise RuntimeError(
                "--kv-cache-dtype=nvfp4 requires Blackwell SM100 or SM120. "
                "Use --kv-cache-dtype=fp4_mx_block16 for the block-size-16 FP4 recipe."
            )
        if (
            prefill_backend != decode_backend and prefill_backend != "fa4"
        ):  # Take care of prefill=fa4 later
            logger.warning(
                f"Attention: Using KV4 with PREFILL = {prefill_backend} "
                f"and DECODE = {decode_backend}. "
                f"Compatibility issues are unlikely, but may occur in rare edge cases."
            )
        else:
            if prefill_backend == "fa4":
                if uses_mla:  # FA4 + MLA
                    KV4_FA4_MLA_BACKEND_CHOICES = [
                        "cutlass_mla",
                        "flashinfer",
                        "trtllm_mla",
                    ]
                    assert decode_backend in KV4_FA4_MLA_BACKEND_CHOICES, (
                        f"KV4 FA4 MLA expects decode_attention_backend to be one of "
                        f"{KV4_FA4_MLA_BACKEND_CHOICES}, but got {decode_backend}"
                    )
                else:  # FA4 + MHA
                    KV4_FA4_MHA_BACKEND_CHOICES = [
                        "triton",
                        "torch_native",
                        "flex_attention",
                    ]
                    assert decode_backend in KV4_FA4_MHA_BACKEND_CHOICES, (
                        f"KV4 FA4 MHA expects decode_attention_backend to be one of "
                        f"{KV4_FA4_MHA_BACKEND_CHOICES}, but got {decode_backend}"
                    )
            else:
                if uses_mla:  # !FA4 + MLA
                    KV4_ATTENTION_MLA_BACKEND_CHOICES = [
                        "cutlass_mla",
                        "flashinfer",
                        "trtllm_mla",
                    ]
                    assert attention_backend in KV4_ATTENTION_MLA_BACKEND_CHOICES, (
                        f"KV4 MLA expects attention_backend to be one of "
                        f"{KV4_ATTENTION_MLA_BACKEND_CHOICES}, but got {attention_backend}"
                    )
                else:  # !FA4 + MHA
                    KV4_ATTENTION_MHA_BACKEND_CHOICES = [
                        "triton",
                        "torch_native",
                        "flex_attention",
                        "trtllm_mha",
                    ]
                    assert attention_backend in KV4_ATTENTION_MHA_BACKEND_CHOICES, (
                        f"KV4 MHA expects attention_backend to be one of "
                        f"{KV4_ATTENTION_MHA_BACKEND_CHOICES}, but got {attention_backend}"
                    )
    else:
        raise RuntimeError("KV4 is not tested on non-CUDA platforms.")


def handle_prefill_only_disable_kv_cache(server_args: Any) -> None:
    """Validate --prefill-only-disable-kv-cache backend constraint.

    Must run after _handle_attention_backend_compatibility() (which fills
    the default attention_backend if unset) and _handle_multi_item_scoring()
    (which may further mutate it). The assertion below guards against
    accidental call-site reordering: if the resolved attention_backend is
    still None, backends haven't settled yet and the resolved (prefill,
    decode) pair would be a stale (None, None).
    """
    from sglang.srt.arg_groups.overrides import attention_backends_of

    cfg = resolving_view(server_args)

    if not cfg.prefill_only_disable_kv_cache:
        return

    assert resolved_view(server_args).attention_backend is not None, (
        "_handle_prefill_only_disable_kv_cache must run after "
        "_handle_attention_backend_compatibility() so the prefill backend is resolved."
    )

    prefill_backend, _ = attention_backends_of(resolved_view(server_args))
    if prefill_backend not in ("fa3", "fa4"):
        raise ValueError(
            "--prefill-only-disable-kv-cache currently requires the FA prefill backend "
            f"(fa3/fa4), but got prefill backend {prefill_backend!r}. Other prefill-only "
            "workloads and backends may be supported in a future change."
        )


def handle_cache_compatibility(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    if (
        cfg.disaggregation_decode_retraction_backup == "host_pool"
        and cfg.disaggregation_mode != "decode"
    ):
        raise ValueError(
            "--disaggregation-decode-retraction-backup=host_pool is only "
            "supported on a PD decode server."
        )
    if cfg.disaggregation_decode_retraction_backup == "host_pool" and cfg.dcp_size > 1:
        raise ValueError(
            "--disaggregation-decode-retraction-backup=host_pool does not "
            "support --dcp-size > 1."
        )
    if (
        cfg.disaggregation_decode_retraction_backup == "host_pool"
        and cfg.enable_priority_scheduling
        and not cfg.disable_priority_preemption
    ):
        raise ValueError(
            "--disaggregation-decode-retraction-backup=host_pool requires "
            "--disable-priority-preemption when priority scheduling is enabled."
        )

    if cfg.enable_hierarchical_cache and cfg.disable_radix_cache:
        raise ValueError(
            "The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive "
            "and cannot be used at the same time. Please use only one of them."
        )

    if cfg.disaggregation_decode_enable_offload_kvcache:
        if cfg.disaggregation_mode != "decode":
            raise ValueError(
                "The argument disaggregation-decode-enable-offload-kvcache is only supported for decode side."
            )
        if cfg.hicache_storage_backend is None:
            raise ValueError(
                "The argument disaggregation-decode-enable-offload-kvcache is only supported when hicache-storage-backend is provided."
            )
        if cfg.disaggregation_decode_retraction_backup == "host_pool":
            raise ValueError(
                "The arguments disaggregation-decode-enable-offload-kvcache and "
                "disaggregation-decode-retraction-backup=host_pool are mutually exclusive: "
                "both build a decode host pool."
            )

    # Validate the effective ratio: model branches may declare a reset
    # (e.g. Step3p forces 1.0 under hierarchical cache) that supersedes
    # the user input before it ever takes effect.
    if not (0 < resolved_view(server_args).swa_full_tokens_ratio <= 1.0):
        raise ValueError("--swa-full-tokens-ratio should be in range (0, 1.0].")


def handle_unified_memory_pool(server_args: Any) -> None:
    from sglang.srt.arg_groups.overrides import attention_backends_of

    cfg = resolving_view(server_args)
    if not cfg.enable_unified_memory:
        return
    if cfg.disaggregation_mode != "null":
        # Constraints of the whole-envelope transfer; see
        # UnifiedMLATokenToKVPool.get_contiguous_buf_infos.
        assert cfg.disaggregation_transfer_backend == "mooncake", (
            "--enable-unified-memory with PD disaggregation supports only "
            "the mooncake transfer backend; got "
            f"{cfg.disaggregation_transfer_backend!r}."
        )
        assert cfg.pp_size == 1, (
            "--enable-unified-memory with PD disaggregation does not support "
            "pipeline parallelism (whole-envelope transfer has no per-layer "
            "entries to subset)."
        )
        assert not envs.SGLANG_DISABLE_LAZY_COMPACTION.get(), (
            "--enable-unified-memory with PD disaggregation requires lazy "
            "compaction; unset SGLANG_DISABLE_LAZY_COMPACTION."
        )
        assert not cfg.enable_hisparse, (
            "--enable-unified-memory with PD disaggregation is not compatible "
            "with --enable-hisparse: the decode-side HiSparse prealloc path "
            "ships host/C4 rows straight from the allocator, bypassing the "
            "virtual->physical translation the unified pool needs."
        )
    assert cfg.speculative_algorithm in (None, "DSPARK"), (
        "--enable-unified-memory only supports --speculative-algorithm "
        "DSPARK (chain draft); other speculative algorithms are not yet "
        "audited for the unified pool's virtual/dense loc translation. Got "
        f"--speculative-algorithm={cfg.speculative_algorithm!r}."
    )
    if cfg.speculative_algorithm == "DSPARK":
        assert cfg.speculative_eagle_topk in (None, 1), (
            "--enable-unified-memory + DSPARK supports a linear draft "
            "chain only (--speculative-eagle-topk in {None, 1}); tree "
            "verify is not audited for the unified pool. Got "
            f"--speculative-eagle-topk={cfg.speculative_eagle_topk!r}."
        )
        # Both roles: verify routes to either backend depending on
        # --speculative-attention-mode.
        spec_allowed = {"triton", "trtllm_mla", "cutedsl_mla", "tokenspeed_mla"}
        spec_backends = set(attention_backends_of(resolved_view(server_args)))
        spec_backends.discard(None)
        assert spec_backends <= spec_allowed, (
            "--enable-unified-memory + DSPARK requires spec-verify-audited "
            f"attention backends {sorted(spec_allowed)} for both prefill "
            f"and decode; got {sorted(spec_backends)}. flashinfer / fa3 do "
            "not translate speculative verify indices to the unified "
            "pool's dense space yet."
        )
    assert not (cfg.enable_hierarchical_cache or cfg.enable_lmcache), (
        "--enable-unified-memory is not yet compatible with hierarchical / "
        "host-tiered KV cache (--enable-hierarchical-cache / --enable-lmcache): "
        "the unified-memory-pool init wires up no host pools, and its device mamba / "
        "full-attention slots are VIRTUAL — the host-offload path does not "
        "translate them to physical."
    )
    assert cfg.dcp_size == 1, (
        "--enable-unified-memory is not yet compatible with decode context "
        "parallelism (--dcp-size > 1): the pool has no DCP-aware masked write "
        "path (UnifiedMHATokenToKVPool.set_kv_buffer asserts dcp_kv_mask is None), "
        "so a DCP run would boot and then fail on the first KV write."
    )
    # Only monolithic decode cuda-graph capture is wired; piecewise prefill
    # capture is not. Guard when the user opts into it.
    _cg_cfg = cfg.cuda_graph_config
    if _cg_cfg is not None and _cg_cfg.prefill.backend == Backend.TC_PIECEWISE:
        raise ValueError(
            "--enable-unified-memory supports monolithic (decode) "
            "cuda-graph capture only; disable piecewise prefill capture "
            "(e.g. --cuda-graph-backend-prefill=disabled)."
        )


def handle_page_major_kv_layout(server_args: Any):
    # The unified pool stores state in the page-major envelope-strided layout, so
    # enabling it implies --enable-page-major-kv-layout — routing it through the
    # single page-major path + stride-aware Triton asserts (set before the guard).
    from sglang.srt.arg_groups.overrides import attention_backends_of

    cfg = resolving_view(server_args)
    if cfg.enable_unified_memory:
        declare_resolution(
            server_args,
            "_handle_page_major_kv_layout",
            enable_page_major_kv_layout=True,
        )
    if not cfg.enable_page_major_kv_layout:
        return
    # Only the Triton attention kernels read the strided 4-D envelope K/V
    # views; FA3 / FlashInfer do not. EXCEPTION: the unified-memory MLA pool
    # exposes each layer as a DENSE contiguous per-layer view
    # (build_dense_mla_views), which the paged MLA kernels consume directly,
    # with their kv_indices / block tables remapped to dense ids. Names below
    # are the RESOLVED ids from attention_backends_of: "flashinfer" is
    # FlashInferMLAAttnBackend for an MLA model, "trtllm_mla" the trtllm
    # decode kernel; "cutedsl_mla" and "tokenspeed_mla" subclass
    # TRTLLMMLABackend and inherit its dense read/write path; "fa3" remaps its
    # page_table (in-kernel for captured decode, one funnel for eager).
    # flashmla / cutlass_mla share the create_flashmla block-table path and
    # can be added the same way once exercised.
    if cfg.enable_unified_memory and use_mla_backend(server_args):
        allowed_full = {
            "triton",
            "fa3",
            "trtllm_mla",
            "flashinfer",
            "cutedsl_mla",
            "tokenspeed_mla",
        }
    else:
        allowed_full = {"triton"}
    backends = set(attention_backends_of(resolved_view(server_args)))
    backends.discard(None)
    assert backends <= allowed_full, (
        "--enable-page-major-kv-layout requires the Triton attention backend "
        "for the full-attention layers (unified-memory MLA also allows the "
        f"paged MLA backends); got {sorted(backends)}, allowed "
        f"{sorted(allowed_full)}. Pass a compatible --attention-backend."
    )
    # The Mamba/KDA state is stored in envelope-strided views; only
    # stride-audited kernels may read it (Stage 4 audit, per slot):
    # - decode: triton; flashinfer (recurrent_kda compiles the state slot
    #   stride as a free int64); helion (specializes KDA state strides 0-3
    #   and rejects a non-unit innermost stride); cutedsl (KDA fused sigmoid-
    #   gating update is stride-safe) on KDA-hybrid models only.
    # - prefill: triton; flashkda (the wrapper gathers/scatters a contiguous
    #   per-slot copy); helion; cutedsl (kernel_h compiles h0/ht with dynamic
    #   int64 strides), with the same KDA-only caveat.
    # - mamba (mamba2/short-conv state): triton only.
    # use_mla_backend() distinguishes the KDA-hybrid family (K3/KimiLinear
    # are MLA-hybrid) from GDN models (GQA-hybrid) for the KDA-only caveat.
    decode_allowed = {"triton", "flashinfer"}
    prefill_allowed = {"triton", "flashkda"}
    if use_mla_backend(server_args):
        decode_allowed.update({"cutedsl", "helion"})
        prefill_allowed.update({"cutedsl", "helion"})
    resolved_linear_decode = cfg.linear_attn_decode_backend or cfg.linear_attn_backend
    resolved_linear_prefill = cfg.linear_attn_prefill_backend or cfg.linear_attn_backend
    assert resolved_linear_decode in decode_allowed | {None}, (
        "--enable-page-major-kv-layout: linear-attention DECODE backend must "
        f"be one of {sorted(decode_allowed)} for the strided conv/SSM state; "
        f"got {resolved_linear_decode!r}."
    )
    assert resolved_linear_prefill in prefill_allowed | {None}, (
        "--enable-page-major-kv-layout: linear-attention PREFILL backend must "
        f"be one of {sorted(prefill_allowed)} for the strided conv/SSM state; "
        f"got {resolved_linear_prefill!r}."
    )
    assert cfg.mamba_backend in (None, "triton"), (
        "--enable-page-major-kv-layout requires the Triton Mamba kernels for "
        f"the strided conv/SSM state; got {cfg.mamba_backend!r}. Pass "
        "--mamba-backend triton."
    )


def validate_prefill_only_disable_kv_cache_args(server_args: Any):
    """Validate --prefill-only-disable-kv-cache flag/precondition constraints.

    Backend resolution is checked separately by
    _handle_prefill_only_disable_kv_cache after backends settle.
    """
    cfg = resolving_view(server_args)
    if not cfg.prefill_only_disable_kv_cache:
        return

    # This flag is intentionally scoped to embedding mode for now. Other
    # prefill-only paths (for example scoring and MIS) can benefit from
    # the same idea later, but some of them still stage K/V through the
    # paged cache today.
    if not cfg.is_embedding:
        raise ValueError(
            "--prefill-only-disable-kv-cache currently requires --is-embedding. "
            "Other prefill-only workloads may be supported in a future change once "
            "their attention paths stop reading or writing the paged KV cache."
        )
    if cfg.kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
        raise ValueError(
            "--prefill-only-disable-kv-cache does not currently support "
            "--kv-cache-dtype=nvfp4 or --kv-cache-dtype=fp4_mx_block16 because "
            "the FP4 pool uses a separate allocation path."
        )
    if cfg.kv_cache_dtype == "mxfp8":
        raise ValueError(
            "--prefill-only-disable-kv-cache does not currently support "
            "--kv-cache-dtype=mxfp8 because the MXFP8 pool stores separate "
            "scale-factor buffers."
        )

    # Structural preconditions for the FA backend's fa_skip_kv_cache path,
    # which is the only embedding path that doesn't read or write the pool:
    # - chunked_prefill_size == -1 keeps a request in a single forward,
    #   so K/V never has to be reused across prefill chunks.
    # - disable_radix_cache stops the prefix cache from indexing pool
    #   slots that no longer hold real data.
    if cfg.chunked_prefill_size != -1:
        raise ValueError(
            "--prefill-only-disable-kv-cache requires --chunked-prefill-size=-1 so the FA "
            "backend takes the fa_skip_kv_cache path; otherwise the pool would be touched "
            "between prefill chunks."
        )
    if not cfg.disable_radix_cache:
        raise ValueError(
            "--prefill-only-disable-kv-cache requires --disable-radix-cache because the "
            "radix cache indexes KV pool slots that no longer hold real data."
        )

    # Context-parallel prefill stages K/V through cp_allgather_and_save_kv_cache,
    # which writes to the pool via set_kv_buffer. NoOpMHATokenToKVPool intentionally
    # raises on writes, so the engine would boot fine but fail on the first request.
    if resolved_view(server_args).attn_cp_size > 1:
        raise ValueError(
            "--prefill-only-disable-kv-cache is incompatible with --attn-cp-size > 1: "
            "the context-parallel attention path writes K/V to the pool via set_kv_buffer, "
            "which the no-op pool intentionally rejects."
        )
    if cfg.enable_prefill_cp:
        raise ValueError(
            "--prefill-only-disable-kv-cache is incompatible with "
            "--enable-prefill-cp: the prefill-CP path stages K/V through "
            "the paged cache, which the no-op pool does not support."
        )

    # HiSparse selects a different pool class (HiSparseDSATokenToKVPool /
    # HiSparseTokenToKVPoolAllocator) that is not the no-op pool.
    if cfg.enable_hisparse:
        raise ValueError(
            "--prefill-only-disable-kv-cache is incompatible with --enable-hisparse: "
            "HiSparse uses a dedicated pool family that is not the no-op MHA pool."
        )

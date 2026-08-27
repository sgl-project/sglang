# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for KV-cache dtype and pool compatibility."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    resolved_view,
    resolving_view,
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
    cfg = resolving_view(server_args)

    if cfg.kv_cache_dtype not in ("nvfp4", "fp4_mx_block16"):
        return

    use_mla_backend = server_args.use_mla_backend()
    prefill_backend, decode_backend = server_args._resolved_attention_backends()
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
                if use_mla_backend:  # FA4 + MLA
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
                if use_mla_backend:  # !FA4 + MLA
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
    cfg = resolving_view(server_args)

    if not cfg.prefill_only_disable_kv_cache:
        return

    assert resolved_view(server_args).attention_backend is not None, (
        "_handle_prefill_only_disable_kv_cache must run after "
        "_handle_attention_backend_compatibility() so the prefill backend is resolved."
    )

    prefill_backend, _ = server_args._resolved_attention_backends()
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
        spec_backends = set(server_args._resolved_attention_backends())
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

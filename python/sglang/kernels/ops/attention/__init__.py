"""Attention compute kernels (Triton): decode / extend / prefill / metadata.

The Triton kernels migrated here live in this package
(``sglang.kernels.ops.attention.<module>``); import them from there. Their
``KernelSpec`` metadata is registered below for inventory (backend = Triton).
KV-cache write kernels went to the ``kvcache`` group instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

# (module, public_fn) migrated from layers/attention/triton_ops + model_executor.
_TRITON_KERNELS = [
    ("decode_attention", "decode_attention_fwd"),
    ("extend_attention", "extend_attention_fwd"),
    ("extend_attention", "build_unified_kv_indices"),
    ("prefill_attention", "context_attention_fwd"),
    ("merge_state", "merge_state_triton"),
    ("suffix_attention_merge", "merge_suffix_attention_in_place"),
    ("metadata", "get_num_kv_splits_triton"),
    ("metadata", "prepare_swa_spec_page_table_triton"),
    ("metadata", "normal_decode_set_metadata"),
    ("dsa.metadata", "fused_dsa_decode_metadata"),
    ("dsa.metadata", "fused_dsa_target_verify_metadata"),
    ("dsa.metadata", "fused_dsa_draft_extend_metadata"),
    ("dsa.kpool_metadata.decode", "fused_dsa_decode_metadata_kpool"),
    ("dsa.kpool_metadata.verify", "fused_dsa_target_verify_metadata_kpool"),
    ("dsa.kpool_metadata.draft_extend", "fused_dsa_draft_extend_metadata_kpool"),
    ("rocm_mla_decode_rope", "decode_attention_fwd_grouped_rope"),
    ("verify_splitkv", "verify_splitkv_fwd"),
    ("unified_attention_3d_mtp", "unified_attention_3d_mtp_func"),
    ("pad", "pad_sequence_with_mask"),
    ("pad", "pad_draft_extend_query"),
    ("pad", "unpad_draft_extend_output"),
    ("pad", "seqlens_expand_triton"),
    ("position", "compute_position_triton"),
    ("table_qk_norm_rope", "table_qk_norm_rope_"),
    ("dsv4.attn_metadata_kernels", "expand_prefill_causally"),
    ("dsv4.attn_metadata_kernels", "build_page_table_positions"),
    ("dsv4.attn_metadata_kernels", "build_causal_swa_page_indices"),
    ("kv_indices", "create_flashinfer_kv_indices_triton"),
    ("kv_indices", "create_flashmla_kv_indices_triton"),
    ("kv_indices", "create_chunked_prefix_cache_kv_indices"),
    ("kv_indices", "get_num_kv_index_blocks_flashmla"),
    ("kv_indices", "get_num_page_per_block_flashmla"),
    ("trtllm_mha_page_table", "build_trtllm_mha_page_table"),
    ("trtllm_mha_graph_metadata", "update_trtllm_mha_graph_metadata"),
    ("aiter_unified_page_table", "scatter_ragged_to_page_table_kernel"),
    ("aiter_unified_page_table", "scatter_req_to_token_to_page_table_kernel"),
    ("fused_qk_rope_reshape_and_cache", "fused_qk_rope_reshape_and_cache"),
    ("mla_qk_prep", "concat_and_cast_mha_k_triton"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn

register_kernel(
    KernelSpec(
        op="attention.kda_qwen38_qsa_sm121",
        backend=KernelBackend.KDA,
        target=("sglang.kernels.kda_kernels.qwen38_qsa_sm121:qwen38_qsa_sm121"),
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(12, 1), max_sm=(12, 1))}
        ),
    )
)


def can_use_kda_qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
) -> bool:
    """Check the exact E2E-qualified Qwen3.8/SM121 QSA contract."""
    from sglang.kernels.kda_kernels.qwen38_qsa_sm121 import (
        can_use_qwen38_qsa_sm121,
    )

    return can_use_qwen38_qsa_sm121(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k)


def qwen38_qsa_sm121_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 0,
    softmax_scale: float = 1.0,
    causal: bool = True,
    **_: object,
) -> torch.Tensor:
    """Run the only SM121 packed-QSA kernel for its qualified contract."""
    del causal
    if max_seqlen_q != 1:
        raise ValueError(f"QSA requires max_seqlen_q=1, got {max_seqlen_q}")
    if not can_use_kda_qwen38_qsa_sm121(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k
    ):
        raise ValueError(
            "unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, "
            "TP1 24Q/2KV or TP2 12Q/1KV, bs<=128, and selected KV<=2055"
        )
    return get_kernel("attention.kda_qwen38_qsa_sm121", KernelBackend.KDA)(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_k,
        softmax_scale,
    )


__all__ = ["can_use_kda_qwen38_qsa_sm121", "qwen38_qsa_sm121_varlen"]


# Vendored linear-attention (flash-linear-attention port) kernels relocated
# in Phase 2.5 (RFC #29630); representative entry points for inventory.
for _mod, _fn in [
    ("fla.chunk", "chunk_gated_delta_rule"),
    ("fla.fused_recurrent", "fused_recurrent_gated_delta_rule"),
    ("fla.kda", "fused_recurrent_kda_fwd"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn

# Linear-attention / MiniMax-sparse / diffusion kernels migrated in Phase 2.5
# (RFC #29630); registered for inventory.
for _grp, _mod, _fn in [
    ("attention", "linear.seg_la", "seg_la_fwd"),
    ("attention", "linear.lightning_attn", "lightning_attention"),
    ("attention", "linear.lightning_attn", "linear_decode_forward_triton"),
    (
        "attention",
        "minimax_sparse.decode.flash_with_topk_idx",
        "flash_decode_with_topk_idx",
    ),
    (
        "attention",
        "minimax_sparse.prefill.flash_with_topk_idx",
        "flash_prefill_with_topk_index",
    ),
]:
    register_kernel(
        KernelSpec(
            op=f"{_grp}.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.{_grp}.{_mod}:{_fn}",
        )
    )
del _grp, _mod, _fn

# DeepSeek DSA / DSV4 kernels migrated in Phase 2.5 (RFC #29630);
# registered for inventory. Import them from their modules.
for _mod, _fn in [
    ("dsa.triton_sparse_mla", "triton_sparse_mla_fwd"),
    ("dsa.transform_index", "transform_index_page_table_prefill"),
    ("dsa.transform_index", "transform_index_page_table_decode"),
    ("dsa.transform_index", "prepare_trtllm_nope_sparse_metadata"),
    ("dsa.transform_index", "transform_index_page_table_dcp"),
    ("dsa.cp_split", "dsa_cp_interleave_q_seqs"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn.lstrip('_')}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn

# Generic attention kernels migrated in Phase 2.5 (RFC #29630).
for _mod, _fn in [
    ("utils", "mla_quantize_and_rope_for_fp8"),
    ("flash_mla_sm120", "flash_mla_with_kvcache_sm120"),
    ("dcp_kernels", "create_dcp_kv_indices"),
    ("dcp_kernels", "correct_attn_out"),
    ("dcp_kernels", "dcp_lse_combine_triton"),
    ("dcp_kernels", "dcp_pack_a2a_send"),
    ("pa_page_table", "_build_pa_page_table"),
    ("dsv4.triton_sparse_decode", "triton_sparse_attn_decode"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn.lstrip('_')}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn

# RoPE / QK-norm fusion kernels migrated from srt/layers top-level strays
# (RFC #29630, Phase 2.5); registered for inventory.
for _mod, _fn in [
    ("dsv4.rope", "precompute_freqs_cis"),
    ("fused_qk_norm_rope_store", "fused_qk_norm_rope_swa_store"),
    ("fused_qk_rmsnorm_rope_gate", "fused_qk_gemma_rmsnorm_rope_gate"),
    ("rotary_triton", "triton_mrope_fused"),
    ("rotary_triton", "triton_ernie45_rope_fused_inplace"),
    ("mrope", "apply_interleaved_rope_triton"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn


# Kernels introduced with Kimi-K3, inventoried by logical operator group.
for _mod, _fn, _backend, _device in [
    ("attn_res", "attn_res_fused_tma", KernelBackend.JIT, CapabilityRequirement.CUDA),
    ("attn_res_hip", "attn_res_hip", KernelBackend.TRITON, CapabilityRequirement.HIP),
    (
        "kda_decode_mtp",
        "fused_kda_decode_mtp_dspark",
        KernelBackend.CUTE_DSL,
        CapabilityRequirement.CUDA,
    ),
    (
        "kda_flydsl.kimi_k3_kda_decode",
        "flydsl_kimi_k3_kda_decode_with_f_b",
        KernelBackend.FLYDSL,
        CapabilityRequirement.HIP,
    ),
    # TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin
    (
        "dsv4.flydsl_v4nm.mla_v4_decode",
        "flydsl_mla_decode_fwd_v4_nm",
        KernelBackend.FLYDSL,
        CapabilityRequirement.HIP,
    ),
    (
        "dsv4.flydsl_v4nm.mla_v4_decode",
        "flydsl_mla_v4_decode_fused",
        KernelBackend.FLYDSL,
        CapabilityRequirement.HIP,
    ),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=_backend,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
            capabilities=frozenset({_device}),
        )
    )
del _mod, _fn, _backend, _device


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="attention.get_block_table",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.minicpm_sala.get_block_table:get_block_table",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="attention.fast_kpool_topk_transform_fused",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.dsa.kpool_topk_transform:fast_kpool_topk_transform_fused",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="attention.topk_transform_kpool_v2",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.dsv4.topk:topk_transform_kpool_v2",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
register_kernel(
    KernelSpec(
        op="attention.topk_transform_kpool_ragged_v2",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.dsv4.topk:topk_transform_kpool_ragged_v2",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fast_topk",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.fast_topk:fast_topk",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="attention.deep_select_topk",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.deep_select:topk",
        # Exactly SM90, SM100 and SM103, the architectures DeepSelect is tuned for.
        capabilities=frozenset(
            CapabilityRequirement.cuda(min_sm=sm, max_sm=sm)
            for sm in ((9, 0), (10, 0), (10, 3))
        ),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_k_indexer_norm_rope",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.dsa.indexer_k:fused_k_indexer_norm_rope",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_k_indexer_norm_rope_store",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.dsa.indexer_k:fused_k_indexer_norm_rope_store",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_rope_wo_a_bf16",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.dsv4.wo_a:fused_rope_wo_a_bf16",
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 9))}
        ),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fp4_index_logits_decode",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.attention.dsv4.fp4_indexer:fp4_index_logits_decode",
        capabilities=CUDA_ONLY,
    )
)

register_kernel(
    KernelSpec(
        op="attention.pack_prefix_current",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.attention.dllm_kv_pack:pack_prefix_current",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="attention.flash_attn_gqa_512",
        backend=KernelBackend.CUTE_DSL,
        target="sglang.kernels.ops.attention.flash_attention_v4:flash_attn_gqa_512",
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(11, 9))}
        ),
    )
)
register_kernel(
    KernelSpec(
        op="attention.gemma_qkv_norm_rope",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.attention.gemma_qkv_norm_rope:gemma_qkv_norm_rope",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="attention.update_msa_v1_decode_plan",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.attention.msa_v1.decode_plan:update_msa_v1_decode_plan",
    )
)

for _fn in ("fp4_index_logits_paged", "finish_paged_indexer_topk"):
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.dsv4.fp4_indexer:{_fn}",
            capabilities=CUDA_ONLY,
        )
    )
del _fn

# Muse Spark QK norm (+ RoPE, + KV store), built per layer: inventoried through
# its class methods.
for _bk in ("torch", "triton", "helion"):
    register_kernel(
        KernelSpec(
            op="attention.muse_spark_qk_norm_rope_store",
            backend=KernelBackend(_bk),
            target="sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store:FusedQKNorm."
            + ("forward_native" if _bk == "torch" else f"forward_{_bk}"),
            capabilities=frozenset() if _bk == "torch" else CUDA_ONLY,
        )
    )
del _bk
for _mod, _fn in [
    ("mla_qk_prep", "concat_and_cast_q_fp8_pad"),
    ("mla_qk_prep", "absorbed_bmm_concat_cast_q_fp8"),
    ("metadata", "q8kv8_topk_length_from_indices"),
    ("minimax_sparse.decode.topk_sparse", "flash_decode_with_gqa_share_sparse"),
    ("minimax_sparse.prefill.topk_sparse", "flash_prefill_with_gqa_share_sparse"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
        )
    )
del _mod, _fn
register_kernel(
    KernelSpec(
        op="attention.absorbed_bmm_concat_cast_q_fp8",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.qprep_bf16_fp8_sm90:q8kv8_qprep_fwd",
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(9, 9))}
        ),
    )
)
register_kernel(
    KernelSpec(
        op="attention.topk_index_reduce",
        backend=KernelBackend.TORCH,
        target="sglang.kernels.ops.attention.minimax_sparse.common.index:topk_index_reduce",
    )
)
# The context-parallel indexer serves only the gfx950 ROCm top-k contract.
for _fn in ("score_local_blocks", "select_local_candidates", "merge_candidates"):
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.attention.minimax_sparse.decode.indexer_cp:{_fn}",
            capabilities=frozenset({CapabilityRequirement.HIP}),
        )
    )
del _fn
register_kernel(
    KernelSpec(
        op="attention.gluon_sparse_prefill",
        backend=KernelBackend.AITER,
        target="sglang.kernels.ops.attention.minimax_sparse.prefill.gluon_prefill:gluon_sparse_prefill",
        capabilities=frozenset({CapabilityRequirement.HIP}),
    )
)
register_kernel(
    KernelSpec(
        op="attention.mtp_verify_attn_fwd_asm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.vattn_asm_gfx950:mtp_verify_attn_fwd_asm",
        capabilities=frozenset({CapabilityRequirement.HIP}),
    )
)
for _mod, _fn in [
    ("helion.kda_decode", "helion_fused_recurrent_kda_packed_decode"),
    ("helion.kda_replayssm", "helion_fused_recurrent_kda_replayssm_decode"),
    ("helion.kda_prefill", "chunk_kda"),
    ("helion.kda_verify", "helion_kda_verify"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.HELION,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
            capabilities=CUDA_ONLY,
        )
    )
del _mod, _fn
# tcgen05 + TMEM: datacenter Blackwell only (SM100 / SM103), not SM120.
for _mod, _fn in [
    ("linear.gdn_blackwell", "chunk_gated_delta_rule_cutedsl"),
    ("linear.kda_blackwell", "chunk_kda_cutedsl"),
    ("linear.kda_nvidia_prefill", "chunk_kda_fwd"),
]:
    register_kernel(
        KernelSpec(
            op=f"attention.{_fn}",
            backend=KernelBackend.CUTE_DSL,
            target=f"sglang.kernels.ops.attention.{_mod}:{_fn}",
            capabilities=frozenset(
                {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 9))}
            ),
        )
    )
del _mod, _fn
register_kernel(
    KernelSpec(
        op="attention.chunk_kda_fwd",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.linear.kda_ptx_prefill:chunk_kda_fwd",
        # Exactly the kda_ptx_prefill.SM_ARCHS build targets (sm_100a, sm_103a).
        capabilities=frozenset(
            CapabilityRequirement.cuda(min_sm=sm, max_sm=sm)
            for sm in ((10, 0), (10, 3))
        ),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fp8_attention",
        backend=KernelBackend.CUTE_DSL,
        target="sglang.kernels.ops.attention.fp8_fa_sm120:fp8_attention",
        # MmaMXF8Op assembles only for SM120 / SM121.
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 1))}
        ),
    )
)

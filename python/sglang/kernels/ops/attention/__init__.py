"""Attention compute kernels (Triton): decode / extend / prefill / metadata.

The Triton kernels migrated here live in this package
(``sglang.kernels.ops.attention.<module>``); import them from there. Their
``KernelSpec`` metadata is registered below for inventory (backend = Triton).
KV-cache index/write kernels went to the ``kvcache`` group instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
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
    ("dsa_metadata", "fused_dsa_decode_metadata"),
    ("dsa_metadata", "fused_dsa_target_verify_metadata"),
    ("dsa_metadata", "fused_dsa_draft_extend_metadata"),
    ("rocm_mla_decode_rope", "decode_attention_fwd_grouped_rope"),
    ("verify_splitkv", "verify_splitkv_fwd"),
    ("unified_attention_3d_mtp", "unified_attention_3d_mtp_func"),
    ("pad", "pad_sequence_with_mask"),
    ("pad", "pad_draft_extend_query"),
    ("pad", "unpad_draft_extend_output"),
    ("pad", "seqlens_expand_triton"),
    ("position", "compute_position_triton"),
    ("dsv4_attn_metadata_kernels", "expand_prefill_causally"),
    ("dsv4_attn_metadata_kernels", "build_page_table_positions"),
    ("dsv4_attn_metadata_kernels", "build_causal_swa_page_indices"),
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
        backend=KernelBackend.TRITON,
        target=("sglang.kernels.kda_kernels.qwen38_qsa_sm121:qwen38_qsa_sm121"),
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(12, 1), max_sm=(12, 1))}
        ),
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description=(
                "Qwen3.8 packed QSA decode: D=256, 12:1 GQA, 1 <= q_rows <= 128"
            ),
        ),
        description=(
            "SM121 Qwen3.8 QSA decode optimized by Codex/Kimi K3 through KDA-1.5."
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
    return get_kernel("attention.kda_qwen38_qsa_sm121", KernelBackend.TRITON)(
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
    ("dsa.cp_split", "dsa_cp_round_robin_split_q_seqs_kernel"),
    ("dsv4.fp4_indexer", "quantize_fp4_indexer_tensor"),
    ("dsv4.fp4_indexer", "store_fp4_index_k_cache"),
    ("dsv4.rms_normalize_hip", "rms_normalize_triton"),
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
    ("utils", "launch_reshape_and_cache_flash"),
    ("utils", "launch_reshape_and_cache_shuffle_5d"),
    ("flash_mla_sm120", "flash_mla_with_kvcache_sm120"),
    ("dcp_kernels", "create_dcp_kv_indices"),
    ("dcp_kernels", "correct_attn_out"),
    ("dcp_kernels", "dcp_lse_combine_triton"),
    ("dcp_kernels", "dcp_pack_a2a_send"),
    ("pa_page_table", "_build_pa_page_table"),
    ("nsa_triton_decode", "triton_sparse_attn_decode"),
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
    ("deepseek_v4_rope", "precompute_freqs_cis"),
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

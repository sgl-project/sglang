"""Cake (FlashInfer) backends for the ``kvcache`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels`, which import FlashInfer only
when a kernel is actually called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

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

register_kernel(
    KernelSpec(
        op="kvcache.fused_qk_rmsnorm_rope_append_paged_kv_cache",
        backend=KernelBackend.FLASHINFER,
        target=(
            "sglang.kernels.cake_kernels.kvcache:"
            "fused_qk_rmsnorm_rope_append_paged_kv_cache"
        ),
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(10, 3))}
        ),
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "BF16 packed QKV [T,(Hq+2Hkv)*128] -> RoPE'd Q [T,Hq,128]; "
                "K/V appended into NHD paged caches; (Hq,Hkv) in {(8,1),(64,8)}"
            ),
        ),
        description=(
            "Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16) "
            "distributed by FlashInfer."
        ),
    )
)

register_kernel(
    KernelSpec(
        op="kvcache.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache",
        backend=KernelBackend.FLASHINFER,
        target=(
            "sglang.kernels.cake_kernels.kvcache:"
            "fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache"
        ),
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(10, 3))}
        ),
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16", "float8_e4m3fn", "float32", "int32"),
            in_place=True,
            description=(
                "BF16 packed QKV [T,(Hq+2Hkv)*128] -> FP8 E4M3 Q [T,Hq,128] + f32 "
                "q_scale (dynamic per-token/head or static q_scale_inv) + int32 "
                "split_k_flag [B,Hkv]; K/V quantized by scalar k_scale/v_scale "
                "into NHD float8_e4m3fn paged caches; (Hq,Hkv) in {(8,1),(64,8)}"
            ),
        ),
        description=(
            "Cake fused QK RMSNorm + NeoX RoPE + FP8 E4M3 quantization + paged "
            "KV append distributed by FlashInfer (post-baseline, FI PR #5956)."
        ),
    )
)


def cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    qk_norm_policy: int = 0,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    clear_unused_last_page_rows: bool = True,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return get_kernel(
        "kvcache.fused_qk_rmsnorm_rope_append_paged_kv_cache",
        KernelBackend.FLASHINFER,
    )(
        qkv,
        cos_sin,
        seq_lens,
        q_indptr,
        page_indices,
        key_cache,
        value_cache,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qk_norm_policy=qk_norm_policy,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        eps=eps,
        out_q=out_q,
        out_k=out_k,
        out_v=out_v,
        clear_unused_last_page_rows=clear_unused_last_page_rows,
    )


def cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    is_prefill: bool,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    quant_policy: int,
    max_seqlen: int = 0,
    upper_max: float = 448.0,
    q_scale_inv: Optional[torch.Tensor] = None,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    qk_norm_policy: int = 0,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    split_k_flag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Cake FP8 entry point; callers gate on the adapter's ``supports_*``."""
    return get_kernel(
        "kvcache.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache",
        KernelBackend.FLASHINFER,
    )(
        qkv,
        cos_sin_cache,
        seq_lens,
        q_indptr,
        page_indices,
        paged_kv_cache,
        is_prefill,
        k_scale,
        v_scale,
        quant_policy,
        max_seqlen=max_seqlen,
        upper_max=upper_max,
        q_scale_inv=q_scale_inv,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        out_q=out_q,
        out_k=out_k,
        out_v=out_v,
        q_scale=q_scale,
        split_k_flag=split_k_flag,
    )


__all__ = [
    "cake_fused_qk_rmsnorm_rope_append_paged_kv_cache",
    "cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache",
]

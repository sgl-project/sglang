"""Cake (FlashInfer) backends for the ``kvcache`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels`, which import FlashInfer only
when a kernel is actually called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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


__all__ = ["cake_fused_qk_rmsnorm_rope_append_paged_kv_cache"]

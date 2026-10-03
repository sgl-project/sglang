"""Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16) via FlashInfer.

FlashInfer entry: ``flashinfer.cake_fused_qk_rope_append``
(``cake_fused_qk_rmsnorm_rope_append_paged_kv_cache``), JIT module
``flashinfer.jit.cake_fused_qk_rope_append``. Contract at FlashInfer
``46340689a5ab``: BF16 packed ``qkv [T, (Hq + 2*Hkv) * 128]``, head_dim 128,
``(Hq, Hkv) in {(8, 1), (64, 8)}``, FP32 ``cos_sin [max_pos, 128]`` with NeoX
pairing, int32 ``seq_lens[B]`` (post-append), ``q_indptr[B+1]``, dense int32
``page_indices[B, max_pages]``, BF16 NHD caches ``[pages, page_size, Hkv, 128]``.
Built for sm_90a / sm_100a / sm_103a. No workspace, no host sync, so the launch
is CUDA-graph capturable.

Not supported here (keep the existing SGLang path): HND caches, FP8 caches
(a separate FlashInfer entry postdates the baseline), head_dim != 128, other
head configurations, non-dense page tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.cake_kernels._support import (
    SM90,
    SM100,
    SM103,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.cake_fused_qk_rope_append"
FI_JIT_MODULE = "flashinfer.jit.cake_fused_qk_rope_append"
ARCHS = (SM90, SM100, SM103)
HEAD_DIM = 128
HEAD_CONFIGS = ((8, 1), (64, 8))


def supports_fused_qk_rmsnorm_rope_append(
    qkv: torch.Tensor,
    key_cache: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    return (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(qkv, ARCHS)
        and qkv.dtype == torch.bfloat16
        and qkv.ndim == 2
        and qkv.is_contiguous()
        and (num_q_heads, num_kv_heads) in HEAD_CONFIGS
        and qkv.shape[1] == (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
        and key_cache.dtype == torch.bfloat16
        and key_cache.ndim == 4
        and key_cache.shape[2] == num_kv_heads
        and key_cache.shape[3] == HEAD_DIM
    )


def fused_qk_rmsnorm_rope_append_paged_kv_cache(
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
    """Forward to FlashInfer; returns the BF16 ``[T, Hq, 128]`` RoPE'd Q.

    ``qk_norm_policy``: 0 none, 1 RoPE then RMSNorm, 2 RMSNorm then RoPE.
    K/V are appended in place into the NHD caches.
    """
    from flashinfer.cake_fused_qk_rope_append import (
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache,
    )

    return cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
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

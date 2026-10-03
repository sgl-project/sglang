"""Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16 and FP8) via FlashInfer.

FlashInfer entry: ``flashinfer.cake_fused_qk_rope_append``
(``cake_fused_qk_rmsnorm_rope_append_paged_kv_cache``), JIT module
``flashinfer.jit.cake_fused_qk_rope_append``. Contract at FlashInfer
``46340689a5ab``: BF16 packed ``qkv [T, (Hq + 2*Hkv) * 128]``, head_dim 128,
``(Hq, Hkv) in {(8, 1), (64, 8)}``, FP32 ``cos_sin [max_pos, 128]`` with NeoX
pairing, int32 ``seq_lens[B]`` (post-append), ``q_indptr[B+1]``, dense int32
``page_indices[B, max_pages]``, BF16 NHD caches ``[pages, page_size, Hkv, 128]``.
Built for sm_90a / sm_100a / sm_103a. No workspace, no host sync, so the launch
is CUDA-graph capturable.

FP8 entry: ``flashinfer.cake_fused_qk_rope_fp8_append``
(``cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache``), JIT module
``flashinfer.jit.cake_fused_qk_rope_fp8_append``; added after the baseline by
FlashInfer PR flashinfer-ai/flashinfer#5956 (commit ``335c4eb4e``), read at
main ``e4f94f948``. Same BF16 packed ``qkv``, head configs, ``cos_sin_cache``,
int32 metadata and dense page table as the BF16 entry; the head configuration
is derived from the ``qkv`` width and the cache's kv-head count. In one launch
it RMSNorms (optional), RoPEs, quantizes Q to FP8 E4M3 (``quant_policy=1``:
dynamic per-token / per-head scale ``max(|q|, 1e-6) / upper_max`` returned in
``q_scale``; ``quant_policy=2``: static ``q_scale_inv`` multiplier, empty
``q_scale``), stores K/V as ``x / k_scale`` / ``x / v_scale`` into NHD
``float8_e4m3fn`` caches ``[pages, page_size, Hkv, 128]`` (or caller-owned
``out_k`` / ``out_v``), zeroes the unused rows of each request's last page and
validates ``q_indptr`` on the device (``split_k_flag`` 0 valid, -1 invalid; an
invalid table leaves every output untouched). Dynamic prefill
(``quant_policy=1`` and ``is_prefill``) needs ``max_seqlen > 0`` and lays
``q_scale`` out as ``[B, Hq, round_up(max_seqlen, 128)]``. Built for sm_90a /
sm_100a / sm_103a; no workspace, no host sync, CUDA-graph capturable.

Not supported here (keep the existing SGLang path): HND caches, head_dim !=
128, other head configurations, non-dense page tables, FP8 caches other than
``float8_e4m3fn``, ``quant_policy`` outside {1, 2}, ``upper_max`` outside
(0, 448], and SM120 / SM121 devices (no cubin for either entry).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

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
FI_FP8_MODULE = "flashinfer.cake_fused_qk_rope_fp8_append"
FI_FP8_JIT_MODULE = "flashinfer.jit.cake_fused_qk_rope_fp8_append"
ARCHS = (SM90, SM100, SM103)
FP8_ARCHS = (SM90, SM100, SM103)
HEAD_DIM = 128
HEAD_CONFIGS = ((8, 1), (64, 8))
FP8_MAX = 448.0
FP8_QUANT_POLICIES = (1, 2)
QK_NORM_POLICIES = (0, 1, 2)


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


# --------------------------------------------------------------------------
# FP8 E4M3 variant (flashinfer.cake_fused_qk_rope_fp8_append)
# --------------------------------------------------------------------------


def supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
    qkv: torch.Tensor,
    key_cache: torch.Tensor,
    *,
    quant_policy: int,
    qk_norm_policy: int = 0,
    upper_max: float = FP8_MAX,
) -> bool:
    """Admission check mirroring the FlashInfer FP8 contract; never raises.

    The head configuration is derived as FlashInfer does: ``Hkv`` from
    ``key_cache.shape[2]`` and ``Hq = qkv.shape[1] // 128 - 2 * Hkv``.
    """
    try:
        import torch

        if not (
            flashinfer_module_available(FI_FP8_MODULE, FI_FP8_JIT_MODULE)
            and cuda_tensor_on(qkv, FP8_ARCHS)
            and qkv.dtype == torch.bfloat16
            and qkv.ndim == 2
            and qkv.is_contiguous()
            and key_cache.dtype == torch.float8_e4m3fn
            and key_cache.ndim == 4
            and key_cache.shape[3] == HEAD_DIM
            and key_cache.is_cuda
            and key_cache.device == qkv.device
        ):
            return False
        num_kv_heads = int(key_cache.shape[2])
        width = int(qkv.shape[1])
        if width % HEAD_DIM or width // HEAD_DIM <= 2 * num_kv_heads:
            return False
        num_q_heads = width // HEAD_DIM - 2 * num_kv_heads
        return (
            (num_q_heads, num_kv_heads) in HEAD_CONFIGS
            and int(quant_policy) in FP8_QUANT_POLICIES
            and int(qk_norm_policy) in QK_NORM_POLICIES
            and math.isfinite(upper_max)
            and 0.0 < float(upper_max) <= FP8_MAX
        )
    except Exception:
        return False


def fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
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
    upper_max: float = FP8_MAX,
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
    """Forward to FlashInfer; returns ``(out_q, q_scale, split_k_flag)``.

    ``out_q`` is FP8 ``[T, Hq, 128]``; ``q_scale`` is f32 ``[T, Hq]`` (decode,
    ``quant_policy=1``), ``[B, Hq, round_up(max_seqlen, 128)]`` (dynamic
    prefill) or empty (``quant_policy=2``); ``split_k_flag`` is int32
    ``[B, Hkv]``. K/V are quantized in place into the ``float8_e4m3fn`` NHD
    ``paged_kv_cache = (key_cache, value_cache)``.
    """
    from flashinfer.cake_fused_qk_rope_fp8_append import (
        cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache,
    )

    return cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
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

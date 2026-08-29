# SPDX-License-Identifier: Apache-2.0

# KDA provenance: optimized by Codex and Kimi K3 agents through KDA-1.5.
# Task: https://github.com/radixark/KDA-1.5/pull/4 @
# 414ce456e14ae8546f77d9356d2c4d955c5bb7f1.
# Winning submission: b4181149c8884ddb.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_SUPPORTED_HEAD_TOPOLOGIES = frozenset({(12, 1), (24, 2)})
# Largest batch qualified by the extended GB10 baseline sweep.
_MAX_BATCH = 128
_MAX_SELECTED_KV = 2055
_logged_fast_path = False


def can_use_qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
) -> bool:
    """Return whether this call matches the captured Qwen3.8 SM121 contract."""
    if not q.is_cuda or q.ndim != 3 or q.dtype != torch.bfloat16:
        return False
    batch, num_q_heads, head_dim = q.shape
    if not (0 < batch <= _MAX_BATCH) or head_dim != 256:
        return False
    if k.ndim != 3 or v.shape != k.shape or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    num_kv_heads = k.shape[1]
    if (num_q_heads, num_kv_heads) not in _SUPPORTED_HEAD_TOPOLOGIES:
        return False
    if k.shape[2] != head_dim or not (0 < max_seqlen_k <= _MAX_SELECTED_KV):
        return False
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        return False
    if q.device != k.device or q.device != v.device:
        return False
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        return False
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        return False
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        return False
    if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
        return False
    if cu_seqlens_q.numel() != batch + 1 or cu_seqlens_k.numel() != batch + 1:
        return False
    properties = torch.cuda.get_device_properties(q.device)
    return (properties.major, properties.minor) == (12, 1)


def qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the KDA-generated Qwen3.8 packed QSA decode kernel."""
    global _logged_fast_path
    if not can_use_qwen38_qsa_sm121(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k):
        raise ValueError("unsupported call for the KDA Qwen3.8 SM121 QSA kernel")

    from .kernel import qwen38_qsa_sm121 as run_kernel

    if not _logged_fast_path:
        logger.info(
            "Using the Codex/Kimi K3 KDA Qwen3.8 QSA kernel on SM121 "
            "(radixark/KDA-1.5#4, submission b4181149c8884ddb)"
        )
        _logged_fast_path = True
    return run_kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)


__all__ = ["can_use_qwen38_qsa_sm121", "qwen38_qsa_sm121"]

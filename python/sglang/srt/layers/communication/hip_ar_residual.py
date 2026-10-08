"""HIP fused all-reduce + residual add via AITER custom all-reduce.

Completes a deferred o_proj all-reduce and folds the pending residual into it.
Decode M in {1, 2, 4} uses the element-parallel 1-stage kernel; M=8 (112 KiB
for Kimi-K3 TP8) uses the 2-stage kernel with the add in the all-gather
writeback. Larger M stays on split AR + add.

Kimi-K3 uses it for the attn-res prefix (AR1); CUDA folds the same add into
``k3_ar_fusion.all_reduce(x, prefix)``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

# Steady-state decode M plus the M=1 CUDA-graph drain bucket.
_RESIDUAL_BATCHES = (1, 2, 4, 8)


def enabled() -> bool:
    return is_hip() and envs.SGLANG_ROCM_AR_RESIDUAL.get()


def covers(num_tokens: int) -> bool:
    if num_tokens not in _RESIDUAL_BATCHES:
        return False
    return num_tokens <= envs.SGLANG_ROCM_AR_RESIDUAL_MAX_TOKENS.get()


def try_all_reduce_add(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """``AR(x) + residual`` in one custom AR, or ``None``.

    ``residual`` must be identical on every rank. Residual fusion runs for
    decode M covered by ``SGLANG_ROCM_AR_RESIDUAL_MAX_TOKENS``. When
    ``residual`` is ``None`` this still completes a deferred o_proj
    all-reduce (plain custom AR) at any M.
    """
    if not enabled() or x.numel() == 0:
        return None
    if residual is not None and not covers(x.shape[0]):
        return None
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        ca_comm = group.ca_comm
        if ca_comm is None or getattr(ca_comm, "disabled", True):
            return None
        if residual is None:
            if hasattr(ca_comm, "custom_all_reduce"):
                out = ca_comm.custom_all_reduce(x)
                return out
            return None
        # Kimi-K3 KDA preserves a leading singleton/head view around the residual while
        # o_proj returns the same token-major storage flattened. Aggregation
        # treats these as the same elementwise tensor; normalize the view so
        # AITER's shape guard does not send the seven KDA layers down split AR.
        if residual.shape != x.shape:
            if residual.numel() != x.numel() or not residual.is_contiguous():
                return None
            residual = residual.view_as(x)
        if not hasattr(ca_comm, "custom_all_reduce_residual"):
            return None
        return ca_comm.custom_all_reduce_residual(x, residual)
    except Exception as exc:
        logger.debug("HIP AR+residual unavailable: %s", exc)
        return None


def all_reduce_add(
    x: torch.Tensor, residual: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Complete a deferred o_proj all-reduce; returns (x, still-pending residual)."""
    fused = try_all_reduce_add(x, residual)
    if fused is not None:
        return fused, None
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(x), residual

"""HIP fused all-reduce + RMSNorm for K3 decode.

NVIDIA fuses the latent all-reduce with RMSNorm via MNNVL
``k3_ar_fusion.all_reduce_norm``. On HIP that path is unavailable.

The AITER 1-stage fused kernel is token-parallel (one block per row,
``__launch_bounds__(1024, 1)``) and loses to AITER's element-parallel 1-stage
all-reduce. Fusion only wins when 2-stage AR is already the faster collective
(TP8: ``bytes >= 80 KiB``): stage 1 is the same reduce-scatter as plain AR,
and stage 2 can fold RMSNorm onto the local load.

K3's latent RMSNorm has no residual add, so the 2-stage kernel is launched with
``skip_residual``. Combined ``[latent | shared]`` buffers pass ``num_norm_rows=N``
so only the latent rows are normalized; ``residual_out`` is the in-place AR
result (shared rows stay reduced, not normed).

Fail-closed: returns ``None`` unless HIP, the K3 fusion flag is on, the layout
is 2-stage, and AITER can service the request.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

# Match AITER custom AR on TP<=8: 1-stage below 80 KiB, 2-stage at/above.
# Combined fused-front c2 is 6*3584*2 = 43 KiB (1-stage AR); c4 is 86 KiB
# and c8 is 172 KiB (2-stage AR).
_AITER_TP8_1STAGE_BYTES = 80 * 1024

_ZEROS: dict[tuple, torch.Tensor] = {}
_OUT: dict[tuple, torch.Tensor] = {}


def aiter_ar_uses_1stage(x: torch.Tensor) -> bool:
    """True when AITER's custom AR would pick the 1-stage kernel (TP<=8)."""
    return x.numel() * x.element_size() < _AITER_TP8_1STAGE_BYTES


def _zeros_like(ref: torch.Tensor) -> torch.Tensor:
    key = (str(ref.device), ref.dtype, tuple(ref.shape))
    buf = _ZEROS.get(key)
    if buf is None or buf.shape != ref.shape or buf.device != ref.device:
        buf = torch.zeros_like(ref)
        _ZEROS[key] = buf
    return buf


def _cached_out(ref: torch.Tensor, rows: int) -> torch.Tensor:
    key = (str(ref.device), ref.dtype, rows, ref.shape[-1])
    buf = _OUT.get(key)
    if buf is None or buf.shape[0] != rows or buf.device != ref.device:
        buf = torch.empty((rows, ref.shape[-1]), dtype=ref.dtype, device=ref.device)
        _OUT[key] = buf
    return buf


def try_fused_ar_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    use_1stage: Optional[bool] = None,
    num_norm_rows: Optional[int] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(RMSNorm(AR(x))[:num_norm_rows], AR(x))`` via AITER, or ``None``.

    Default policy: skip 1-stage sizes (cannot beat element-parallel AR) and
    run 2-stage fused with ``skip_residual`` and in-place ``residual_out=x``.
    ``use_1stage=True`` still dispatches the one-shot kernel (tests / override)
    with a cached zeros residual.

    ``SGLANG_ROCM_K3_FUSED_AR_RMSNORM_MAX_TOKENS`` (0 = no cap) bounds it from
    above: past ~24 tokens the split pair wins, and the gap widens with M.
    """
    if not is_hip() or not envs.SGLANG_ROCM_K3_FUSED_AR_RMSNORM.get():
        return None
    if x.numel() == 0 or weight.numel() != x.shape[-1]:
        return None
    max_tokens = envs.SGLANG_ROCM_K3_FUSED_AR_RMSNORM_MAX_TOKENS.get()
    # Model tokens, not buffer rows: the combined front has 3 rows per token.
    tokens = x.shape[0] if num_norm_rows is None else int(num_norm_rows)
    if max_tokens > 0 and tokens > max_tokens:
        return None
    if use_1stage is None:
        if aiter_ar_uses_1stage(x):
            return None
        use_1stage = False
    try:
        from sglang.srt.distributed import (
            tensor_model_parallel_fused_allreduce_rmsnorm,
        )

        skip_residual = not use_1stage
        residual = x if skip_residual else _zeros_like(x)
        residual_out = x if skip_residual else None
        rows = x.shape[0] if num_norm_rows is None else int(num_norm_rows)
        return tensor_model_parallel_fused_allreduce_rmsnorm(
            x,
            residual,
            weight,
            eps,
            use_1stage=use_1stage,
            residual_out=residual_out,
            out=_cached_out(x, rows),
            num_norm_rows=rows,
            skip_residual=skip_residual,
        )
    except Exception as exc:
        logger.debug("K3 fused AR+RMSNorm unavailable: %s", exc)
        return None

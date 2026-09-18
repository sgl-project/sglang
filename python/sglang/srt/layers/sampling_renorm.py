"""Top-p / top-k probability renormalization with deterministic output.

flashinfer's default ``top_p_renorm_probs`` (AIR radix) and
``top_k_renorm_probs`` (radix multi-CTA) pool partial sums with float
``atomicAdd``, so byte-identical input can return probabilities differing in
the last bits (measured: 99 of 99 repeated calls on a 256 x 128256 flat
distribution; for top-k even on peaky rows). Each TP rank renormalizes
independently, and the output decides committed tokens and speculative accept
lengths, so a last-bit gap desynchronizes the ranks (#33549, #33289).

``deterministic`` defaults to ``SGLANG_RENORM_DETERMINISTIC``; the top-k
fallback goes away once the flashinfer pin includes flashinfer-ai/flashinfer#5034.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang.srt.environ import envs

try:
    import flashinfer.sampling as _flashinfer_sampling

    _HAS_FLASHINFER = True
except ImportError:  # pragma: no cover - non-CUDA builds
    _flashinfer_sampling = None
    _HAS_FLASHINFER = False

import sgl_kernel as _sgl_kernel


def _resolve(deterministic: Optional[bool]) -> bool:
    if deterministic is None:
        return envs.SGLANG_RENORM_DETERMINISTIC.get()
    return bool(deterministic)


def _split_param(x: Union[torch.Tensor, float, int]):
    """(per-row tensor or None, scalar) for the sgl_kernel op schemas."""
    if isinstance(x, torch.Tensor):
        return x, 0
    return None, x


def _single_cta_top_k(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    # torch.ops.sgl_kernel.top_k_renorm_probs is flashinfer's single-CTA kernel
    # compiled into sgl_kernel: one block per row, fixed-order block reductions.
    arr, val = _split_param(top_k)
    probs32 = probs.float()
    out = torch.empty_like(probs32)
    torch.ops.sgl_kernel.top_k_renorm_probs.default(
        probs32, out, arr.int() if arr is not None else None, int(val)
    )
    return out if out.dtype == probs.dtype else out.to(probs.dtype)


def _single_cta_top_p(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    arr, val = _split_param(top_p)
    probs32 = probs.float()
    out = torch.empty_like(probs32)
    torch.ops.sgl_kernel.top_p_renorm_probs.default(
        probs32, out, arr.float() if arr is not None else None, float(val)
    )
    return out if out.dtype == probs.dtype else out.to(probs.dtype)


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    deterministic: Optional[bool] = None,
) -> torch.Tensor:
    if not _resolve(deterministic):
        return _sgl_kernel.top_p_renorm_prob(probs, top_p)
    if _HAS_FLASHINFER and probs.is_cuda:
        # is_deterministic exists since flashinfer 0.6.7; sglang pins newer.
        return _flashinfer_sampling.top_p_renorm_probs(
            probs, top_p, is_deterministic=True
        )
    # no flashinfer (e.g. MUSA): the single-CTA kernel is deterministic already.
    return _single_cta_top_p(probs, top_p)


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    deterministic: Optional[bool] = None,
) -> torch.Tensor:
    if not _resolve(deterministic):
        return _sgl_kernel.top_k_renorm_prob(probs, top_k)
    return _single_cta_top_k(probs, top_k)


__all__ = ["top_k_renorm_prob", "top_p_renorm_prob"]

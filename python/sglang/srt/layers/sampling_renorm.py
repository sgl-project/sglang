"""Top-p / top-k probability renormalization with deterministic output.

Why this module exists
----------------------
flashinfer's default ``top_p_renorm_probs`` (AIR radix, flashinfer >= 0.6.7)
and ``top_k_renorm_probs`` (radix multi-CTA) pool partial sums across thread
blocks with float ``atomicAdd``. Float addition is not associative, so two
calls on byte-identical input can return probabilities that differ in the
last bits (measured: 99 of 99 repeated calls differ on a 256 x 128256 flat
distribution; for top-k even on peaky rows).

Every tensor-parallel rank runs these kernels independently on the same
logits, and the output feeds decisions that are committed to per-rank state:
sampled tokens (the ``min_p`` path of the sampler) and speculative-decoding
accept lengths / bonus tokens (DFlash, DSpark, EAGLE verify). A last-bit gap
between ranks flips a rejection-sampling coin on one rank only, the per-rank
radix/KV caches drift apart, and a later prefix match deadlocks an NCCL
collective (#33549, #33289; #33614 is the broadcast that papers over it).

So by default this module routes to kernels whose output is bit-identical
call to call:

* top-p: flashinfer's integer-histogram AIR variant (``is_deterministic=True``);
  without flashinfer (MUSA), the single-CTA kernel compiled into ``sgl_kernel``.
* top-k: the single-CTA kernel compiled into ``sgl_kernel`` (fixed-order block
  reductions). flashinfer has no deterministic option for its radix top-k.

Set ``SGLANG_RENORM_DETERMINISTIC=0`` (or pass ``deterministic=False``) to opt
back into the faster non-deterministic kernels, e.g. on a single rank without
speculative decoding.
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
    """Zero every token outside the top-p nucleus and renormalize.

    ``deterministic`` defaults to ``SGLANG_RENORM_DETERMINISTIC`` (on). When on,
    repeated calls on identical input return bit-identical output.
    """
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
    """Zero every token outside the top-k set and renormalize.

    ``deterministic`` defaults to ``SGLANG_RENORM_DETERMINISTIC`` (on). When on,
    repeated calls on identical input return bit-identical output.
    """
    if not _resolve(deterministic):
        return _sgl_kernel.top_k_renorm_prob(probs, top_k)
    return _single_cta_top_k(probs, top_k)


__all__ = ["top_k_renorm_prob", "top_p_renorm_prob"]

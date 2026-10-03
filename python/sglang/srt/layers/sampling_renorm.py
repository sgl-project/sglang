"""Which top-p / top-k renorm variant runs, decided per call site.

The fast flashinfer kernels pool partial sums with float atomics, so TP ranks
can disagree in the last bits on identical logits. The deterministic variants
close that at 1.2x (top-p) to 5-40x (single-CTA top-k; GB300 is the worst) the
cost. Consumers that already broadcast their decision from rank 0 (speculative
verify: EAGLE always, DFlash/DSpark via SGLANG_SPEC_TP_SYNC) keep the fast
kernels; the plain sampler has no broadcast and goes deterministic as soon as
more than one attention-TP rank has to agree. SGLANG_RENORM_DETERMINISTIC=0/1
overrides both; --enable-deterministic-inference forces both on.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_parallel

try:
    import flashinfer.sampling as _flashinfer_sampling
except ImportError:  # pragma: no cover - non-CUDA builds
    _flashinfer_sampling = None

try:
    import sgl_kernel as _sgl_kernel
except ImportError:  # pragma: no cover - CPU-only (policy tests)
    _sgl_kernel = None


def renorm_deterministic(*, ranks_agree: bool) -> bool:
    """``ranks_agree``: the consumer broadcasts its decision from rank 0 anyway."""
    if envs.SGLANG_RENORM_DETERMINISTIC.is_set():
        return envs.SGLANG_RENORM_DETERMINISTIC.get()
    if get_exec().deterministic.enable_deterministic_inference:
        return True
    if ranks_agree:
        return False
    return get_parallel().attn_tp_size > 1


def _split_param(x: Union[torch.Tensor, float, int]):
    """(per-row tensor or None, scalar) for the sgl_kernel op schemas."""
    if isinstance(x, torch.Tensor):
        return x, 0
    return None, x


def _single_cta_top_k(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    # flashinfer's single-CTA kernel compiled into sgl_kernel: fixed-order reductions.
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
    if deterministic is None:
        deterministic = renorm_deterministic(ranks_agree=False)
    if not deterministic:
        return _sgl_kernel.top_p_renorm_prob(probs, top_p)
    if _flashinfer_sampling is not None and probs.is_cuda:
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
    if deterministic is None:
        deterministic = renorm_deterministic(ranks_agree=False)
    if not deterministic:
        return _sgl_kernel.top_k_renorm_prob(probs, top_k)
    return _single_cta_top_k(probs, top_k)


def spec_top_p_renorm_prob(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Speculative verify: the accept decision is broadcast from rank 0."""
    return top_p_renorm_prob(
        probs, top_p, deterministic=renorm_deterministic(ranks_agree=True)
    )


def spec_top_k_renorm_prob(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    return top_k_renorm_prob(
        probs, top_k, deterministic=renorm_deterministic(ranks_agree=True)
    )


__all__ = [
    "renorm_deterministic",
    "spec_top_k_renorm_prob",
    "spec_top_p_renorm_prob",
    "top_k_renorm_prob",
    "top_p_renorm_prob",
]

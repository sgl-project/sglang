"""Which top-p / top-k renorm variant runs.

The fast flashinfer kernels pool partial sums with float atomics, so TP ranks
can disagree in the last bits on identical logits; a shared sampling coin that
lands in that gap commits different tokens on different ranks and the ranks
later wedge in a collective. The deterministic variants (flashinfer
``is_deterministic=True``: integer-histogram AIR top-p, fixed-point sum radix
top-k) close that, at no measurable end-to-end cost. Policy: deterministic
whenever the TP group has more than one rank, on every path (plain sampler and
speculative verify alike), and under --enable-deterministic-inference; fast on
a single rank, where there is nobody to agree with.

The fast kernels come from the per-platform table in
``sglang.kernels.ops.sampling.renorm``; this module keeps that table's
contract (a name is ``None`` where the platform has no kernel).
"""

from __future__ import annotations

import inspect
from typing import Optional, Union

import torch

from sglang.kernels.ops.sampling import renorm as _platform
from sglang.srt.runtime_context import get_exec, get_parallel

try:
    import flashinfer.sampling as _flashinfer_sampling
except ImportError:  # pragma: no cover - non-CUDA builds
    _flashinfer_sampling = None


def _flashinfer_has_deterministic(name: str) -> bool:
    # top-p has had is_deterministic since 0.6.7; top-k only since
    # flashinfer-ai/flashinfer#4831 (0.7.x). sglang pins 0.7.2, but a deployment
    # on an older flashinfer must not fail at the first sampled request.
    if _flashinfer_sampling is None:
        return False
    fn = getattr(_flashinfer_sampling, name, None)
    return fn is not None and "is_deterministic" in inspect.signature(fn).parameters


_FI_TOP_P_DETERMINISTIC = _flashinfer_has_deterministic("top_p_renorm_probs")
_FI_TOP_K_DETERMINISTIC = _flashinfer_has_deterministic("top_k_renorm_probs")


def renorm_deterministic() -> bool:
    # tp_size, not attn_tp_size: context-parallel ranks sample the same batch
    # too, and attn_tp_size divides them out.
    if get_exec().deterministic.enable_deterministic_inference:
        return True
    return get_parallel().tp_size > 1


def _top_p_renorm_prob(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    deterministic: Optional[bool] = None,
) -> torch.Tensor:
    if deterministic is None:
        deterministic = renorm_deterministic()
    if deterministic and _FI_TOP_P_DETERMINISTIC and probs.is_cuda:
        return _flashinfer_sampling.top_p_renorm_probs(
            probs, top_p, is_deterministic=True
        )
    # The platform's fast kernel: flashinfer's default on CUDA, Triton on HIP,
    # sgl_kernel's single-CTA kernel on MUSA / XPU (fixed-order block
    # reductions, deterministic already).
    return _platform.top_p_renorm_prob(probs, top_p)


def _top_k_renorm_prob(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    deterministic: Optional[bool] = None,
) -> torch.Tensor:
    if deterministic is None:
        deterministic = renorm_deterministic()
    if deterministic and _FI_TOP_K_DETERMINISTIC and probs.is_cuda:
        return _flashinfer_sampling.top_k_renorm_probs(
            probs, top_k, is_deterministic=True
        )
    return _platform.top_k_renorm_prob(probs, top_k)


top_p_renorm_prob = (
    _top_p_renorm_prob if _platform.top_p_renorm_prob is not None else None
)
top_k_renorm_prob = (
    _top_k_renorm_prob if _platform.top_k_renorm_prob is not None else None
)

__all__ = [
    "renorm_deterministic",
    "top_k_renorm_prob",
    "top_p_renorm_prob",
]

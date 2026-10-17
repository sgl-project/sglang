"""Which top-p / top-k renorm variant runs.

The fast flashinfer kernels pool partial sums with float atomics, so TP ranks
can disagree in the last bits on identical logits; a shared sampling coin that
lands in that gap commits different tokens on different ranks and the ranks
later wedge in a collective. The deterministic variants (flashinfer
``is_deterministic=True``: integer-histogram AIR top-p, fixed-point sum radix
top-k) close that, at no measurable end-to-end cost. Policy: deterministic
whenever more than one attention-TP rank has to agree, on every path (plain
sampler and speculative verify alike); fast on a single rank.
SGLANG_RENORM_DETERMINISTIC=0/1 overrides; --enable-deterministic-inference
forces on.
"""

from __future__ import annotations

import inspect
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
    if envs.SGLANG_RENORM_DETERMINISTIC.is_set():
        return envs.SGLANG_RENORM_DETERMINISTIC.get()
    if get_exec().deterministic.enable_deterministic_inference:
        return True
    return get_parallel().attn_tp_size > 1


def top_p_renorm_prob(
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
    # sgl_kernel forwards to flashinfer's default kernel when flashinfer is
    # present; without it (MUSA, XPU) it runs its own single-CTA kernel, whose
    # fixed-order block reductions are deterministic already.
    return _sgl_kernel.top_p_renorm_prob(probs, top_p)


def top_k_renorm_prob(
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
    return _sgl_kernel.top_k_renorm_prob(probs, top_k)


__all__ = [
    "renorm_deterministic",
    "top_k_renorm_prob",
    "top_p_renorm_prob",
]

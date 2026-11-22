"""Row-wise top-k of a score matrix, resolved once per platform."""

from __future__ import annotations

import functools
import logging

import torch

from sglang._lazy_import import LazyImport
from sglang.srt.environ import envs
from sglang.srt.utils import is_flashinfer_available

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    # Resolved on first call, like the renorm kernels.
    _flashinfer_top_k = LazyImport("flashinfer", "top_k")
else:
    _flashinfer_top_k = None


def radix_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest values of each row of ``scores`` and their int64 ids,
    sorted descending.

    FlashInfer's radix select reads each row once and is several times faster
    than ``torch.topk`` on a large vocabulary. It runs in deterministic mode:
    when values tie at the k-th place, which of the tied ids are returned is
    fixed for a given input, but it may differ from ``torch.topk``'s choice.
    Without FlashInfer, or for a tensor that is not on the GPU, this is
    ``torch.topk``.
    """
    if _flashinfer_top_k is not None and scores.is_cuda:
        # FlashInfer only reads dense rows.
        return _flashinfer_top_k(
            scores.contiguous(), k, sorted=True, deterministic=True
        )
    return torch.topk(scores, k, dim=-1)


@functools.cache
def prebuild_radix_topk(device: torch.device | str) -> None:
    """Build and load FlashInfer's top-k module before serving.

    Without flashinfer-jit-cache, FlashInfer compiles it on first use, which
    takes minutes on a cold cache. Reached first from a sampling step, that
    build runs inside the scheduler loop and stalls every request, health
    checks included, until it finishes.
    """
    if (
        _flashinfer_top_k is None
        or torch.device(device).type != "cuda"
        or not envs.SGLANG_FLASHINFER_PREBUILD_SAMPLING.get()
    ):
        return
    logger.info(
        "Loading FlashInfer top-k kernels; an uncached first build takes minutes "
        "(SGLANG_FLASHINFER_PREBUILD_SAMPLING=0 defers it to the first request)."
    )
    radix_topk(torch.zeros((1, 2), dtype=torch.float32, device=device), 1)


__all__ = ["prebuild_radix_topk", "radix_topk"]

"""Row-wise top-k of a score matrix, resolved once per platform."""

from __future__ import annotations

import torch

from sglang.srt.utils import is_flashinfer_available
from sglang.utils import LazyImport

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


__all__ = ["radix_topk"]

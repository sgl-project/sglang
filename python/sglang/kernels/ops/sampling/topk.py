"""Row-wise top-k of a score matrix, resolved once per platform."""

from __future__ import annotations

import torch

from sglang._lazy_import import LazyImport
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    # Resolved on first call, like the renorm kernels.
    _flashinfer_top_k = LazyImport("flashinfer", "top_k")
else:
    _flashinfer_top_k = None


def radix_topk(
    scores: torch.Tensor, k: int, *, deterministic: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest values of each row of ``scores`` and their int64 ids,
    sorted descending.

    FlashInfer's radix select reads each row once and is several times faster
    than ``torch.topk`` on a large vocabulary. In deterministic mode, when
    values tie at the k-th place, which of the tied ids are returned is fixed
    for a given input, but it may differ from ``torch.topk``'s choice;
    ``deterministic=False`` drops that guarantee for FlashInfer's faster path.
    Without FlashInfer, or for a tensor that is not on the GPU, this is
    ``torch.topk``, which on XPU also beats sgl-kernel-xpu's SYCL top-k.
    """
    if _flashinfer_top_k is not None and scores.is_cuda:
        # FlashInfer only reads dense rows.
        return _flashinfer_top_k(
            scores.contiguous(), k, sorted=True, deterministic=deterministic
        )
    return torch.topk(scores, k, dim=-1)


__all__ = ["radix_topk"]

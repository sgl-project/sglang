"""Flashinfer-accelerated top-k for large vocabulary sizes.

This module provides FlashinferTopK, a MultiPlatformOp that uses flashinfer.top_k
for radix-based top-k selection on CUDA. This is O(n) vs O(n log k) for heap-based,
providing 3-100x speedup for large vocabularies (>10k).

Used by:
- logprob.py: get_top_logprobs_raw() for top logprobs computation
- (Note: nsa_backend.py has its own flashinfer top-k integration, not using this op)
"""

from typing import Tuple

import torch

from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.srt.utils import is_cuda

# Threshold: flashinfer.top_k is faster for vocab > 10k
FLASHINFER_TOPK_THRESHOLD = 10000


_flashinfer_available = False
if is_cuda():
    try:
        import flashinfer

        _flashinfer_available = True
    except ImportError:
        pass


class FlashinferTopK(MultiPlatformOp):
    """
    Multi-platform top-k op that uses flashinfer on CUDA for large vocab.

    Uses radix-based top-k selection which is O(n) vs O(n log k) for heap-based,
    providing 3-100x speedup for large vocabularies (>10k).

    Usage:
        topk_op = FlashinferTopK()
        values, indices = topk_op(tensor, k)
    """

    def __init__(self, threshold: int = FLASHINFER_TOPK_THRESHOLD):
        super().__init__()
        self.threshold = threshold

    def forward_native(
        self,
        input: torch.Tensor,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch native implementation."""
        return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)

    def forward_cuda(
        self,
        input: torch.Tensor,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CUDA implementation using flashinfer radix top-k.

        Falls back to torch.topk when:
        - flashinfer not available
        - dim != -1 or input is not 2D (flashinfer limitation)
        - largest=False (flashinfer only supports largest)
        - vocab_size < threshold (torch.topk faster for small vocab)

        API used: flashinfer.top_k(input, k, sorted) -> (values, indices)
        """
        use_fallback = (
            not _flashinfer_available
            or dim != -1
            or input.dim() != 2
            or not largest
            or input.size(-1) < self.threshold
        )

        if use_fallback:
            return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)

        values, indices = flashinfer.top_k(input, k, sorted=sorted)
        return values, indices

    def forward_npu(
        self,
        input: torch.Tensor,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NPU implementation - fallback to native."""
        return self.forward_native(input, k, dim, largest, sorted)


# Global singleton instance for convenience
_default_topk_op = None


def get_flashinfer_topk_op(
    threshold: int = FLASHINFER_TOPK_THRESHOLD,
) -> FlashinferTopK:
    """Get or create the default FlashinferTopK instance."""
    global _default_topk_op
    if _default_topk_op is None:
        _default_topk_op = FlashinferTopK(threshold=threshold)
    return _default_topk_op


def flashinfer_topk(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function - drop-in replacement for torch.topk.

    Dispatches to flashinfer on CUDA (for large vocab), torch.topk otherwise.
    Uses radix-based selection which is O(n) vs O(n log k), providing
    3-100x speedup for large vocabularies.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch_size, d) for flashinfer acceleration,
        or any shape for torch.topk fallback.
    k : int
        Number of top elements to select.
    dim : int, optional
        Dimension to sort along. Default is -1.
        Note: flashinfer only supports dim=-1 with 2D input.
    largest : bool, optional
        If True, return largest elements. Default is True.
        Note: flashinfer only supports largest=True.
    sorted : bool, optional
        If True, return sorted results. Default is True.

    Returns
    -------
    values : torch.Tensor
        Top-k values.
    indices : torch.Tensor
        Indices of top-k values.
    """
    return get_flashinfer_topk_op()(input, k, dim, largest, sorted)

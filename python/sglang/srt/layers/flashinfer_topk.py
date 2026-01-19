"""Flashinfer-accelerated top-k for large vocabulary sizes.

Provides flashinfer_topk() - a drop-in replacement for torch.topk that uses
flashinfer.top_k for radix-based selection on CUDA when available.
This is O(n) vs O(n log k) for heap-based, providing 3-100x speedup for
large vocabularies (>10k).

Automatically dispatches to:
- flashinfer.top_k on CUDA (when available and conditions met)
- torch.topk as fallback (CPU, NPU, or when flashinfer unavailable)

Used by:
- logprob.py: get_top_logprobs_raw() for top logprobs computation
- (Note: nsa_backend.py has its own flashinfer top-k integration)
"""

from typing import Tuple

import torch

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


def flashinfer_topk(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for torch.topk with flashinfer acceleration on CUDA.

    Automatically dispatches to flashinfer.top_k when:
    - flashinfer is available
    - device is CUDA
    - input is 2D
    - dim == -1 (required by flashinfer)
    - largest == True (required by flashinfer)
    - vocab_size (input.size(-1)) > FLASHINFER_TOPK_THRESHOLD

    Falls back to torch.topk otherwise.

    Uses radix-based selection which is O(n) vs O(n log k), providing
    3-100x speedup for large vocabularies.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch_size, vocab_size) or any shape.
        flashinfer optimization works best with (batch_size, large_vocab_size).
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
    # Check if we can use flashinfer
    use_flashinfer = (
        _flashinfer_available
        and input.is_cuda
        and input.dim() == 2
        and dim == -1
        and largest
        and input.size(-1) > FLASHINFER_TOPK_THRESHOLD
    )

    if use_flashinfer:
        # flashinfer.top_k(input, k, sorted) -> (values, indices)
        return flashinfer.top_k(input, k, sorted=sorted)

    # Fallback to torch.topk
    return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)

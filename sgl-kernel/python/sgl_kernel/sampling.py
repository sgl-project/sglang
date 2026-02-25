from typing import Optional, Union

import torch
from sgl_kernel.utils import _to_tensor_scalar_tuple


def _top_k_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops.sgl_kernel.top_k_renorm_probs.default(
        probs, renorm_probs, maybe_top_k_arr, top_k_val
    )
    return renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    r"""Adapt from https://github.com/flashinfer-ai/flashinfer/flashinfer/sampling.py
    Fused GPU kernel for renormalizing probabilities by top-k thresholding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for for
        for re-normalizing probabilities, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k probabilities, set the rest to zero, and renormalize the probabilities.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Note
    ----
    This combination of ``top_k_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_k_sampling_from_probs``.
    """
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


top_k_renorm_prob = top_k_renorm_probs


def _top_p_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops.sgl_kernel.top_p_renorm_probs.default(
        probs, renorm_probs, maybe_top_p_arr, top_p_val
    )
    return renorm_probs


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    r"""Adapt from https://github.com/flashinfer-ai/flashinfer/flashinfer/sampling.py
    Fused GPU kernel for renormalizing probabilities by top-p thresholding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-p threshold for for
        re-normalizing probabilities, should be in ``(0, 1)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We mask out the probabilities less than `threshold` where the cumulative sum
        of ``probs[probs >= threshold]`` is `top_p`, and renormalize the probabilities.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Note
    ----
    This combination of ``top_p_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_p_sampling_from_probs``.

    """
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))


top_p_renorm_prob = top_p_renorm_probs


def _top_k_mask_logits_internal(
    logits: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    logits = logits.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    mask_logits = torch.empty_like(logits)
    torch.ops.sgl_kernel.top_k_mask_logits.default(
        logits, mask_logits, maybe_top_k_arr, top_k_val
    )
    return mask_logits


def top_k_mask_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    r"""Adapt from https://github.com/flashinfer-ai/flashinfer/flashinfer/sampling.py
    Fused GPU kernel for masking logits by top-k thresholding.

    Parameters
    ----------
    logits: torch.Tensor
        Logits before softmax, shape ``(batch_size, num_classes)``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for for
        for masking logits, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k logits, set the rest to negative infinity.

    Returns
    -------
    masked_logits: torch.Tensor
        Masked logits, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 3
    >>> logits = torch.randn(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> masked_logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
    >>> masked_logits
    tensor([[ 1.9269,  1.4873,  0.9007,    -inf,    -inf],
            [ 1.0783,  0.8008,  1.6806,    -inf,    -inf],
            [   -inf,  0.2415, -0.2316,  0.0418,    -inf],
            [ 0.8599, -0.3097,    -inf,  0.8034,    -inf]], device='cuda:0')

    Note
    ----
    The combination of ``top_k_mask_logits`` and ``softmax`` should be equivalent to ``top_k_renorm_probs``.

    See Also
    --------
    top_k_renorm_probs
    """
    return _top_k_mask_logits_internal(logits, *_to_tensor_scalar_tuple(top_k))

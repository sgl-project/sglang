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


def _top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    with probs.device as device:
        probs = probs.float()
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        torch.ops.sgl_kernel.top_p_sampling_from_probs.default(
            probs,
            samples,
            indices,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            generator,
        )
        return samples


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Adapt from https://github.com/flashinfer-ai/flashinfer/flashinfer/sampling.py
    Fused GPU kernel for top-p sampling (nucleus sampling) from probabilities,
    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Note
    ----
    This function expects float32 inputs, and the output is int32.

    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return _top_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(top_p), deterministic, generator
    )


def _top_k_top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    with probs.device as device:
        probs = probs.float()
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        torch.ops.sgl_kernel.top_k_top_p_sampling_from_probs.default(
            probs,
            samples,
            indices,
            maybe_top_k_arr,
            top_k_val,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            generator,
        )
        return samples


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Adapt from https://github.com/flashinfer-ai/flashinfer/flashinfer/sampling.py
    Fused GPU kernel for top-k and top-p sampling from probabilities,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    filter_apply_order: str
        The order of applying top-k and top-p sampling, should be either ``"top_k_first"`` or ``"joint"``.
        If ``"top_k_first"``, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Note
    ----
    This function expects float32 inputs, and the output is int32.

    """
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs,
            top_p,
            indices,
            deterministic,
            check_nan=check_nan,
            generator=generator,
        )
    elif filter_apply_order == "joint":
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return _top_k_top_p_sampling_from_probs_internal(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def _min_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_min_p_arr: Optional[torch.Tensor],
    min_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    with probs.device as device:
        probs = probs.float()
        maybe_min_p_arr = (
            maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        torch.ops.sgl_kernel.min_p_sampling_from_probs.default(
            probs,
            samples,
            indices,
            maybe_min_p_arr,
            min_p_val,
            deterministic,
            generator,
        )
        return samples


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    r"""Adapt from https://github.com/flashinfer-ai/flashinfer/flashinfer/sampling.py
    Fused GPU kernel for `min_p sampling <https://arxiv.org/abs/2407.01082>`_ from probabilities,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    min_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for min-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)`` that maps each output to a row in probs.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return _min_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(min_p), deterministic, generator
    )

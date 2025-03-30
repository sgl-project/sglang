from typing import Optional, Tuple, Union

import torch
from sgl_kernel.utils import _to_tensor_scalar_tuple, get_cuda_stream


def _top_k_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops.sgl_kernel.top_k_renorm_probs.default(
        probs,
        renorm_probs,
        maybe_top_k_arr,
        top_k_val,
        get_cuda_stream(),
    )
    return renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
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
        probs,
        renorm_probs,
        maybe_top_p_arr,
        top_p_val,
        get_cuda_stream(),
    )
    return renorm_probs


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))


top_p_renorm_prob = top_p_renorm_probs


def _top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with probs.device as device:
        probs = probs.float()
        uniform_samples = uniform_samples.float()
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        success = torch.empty(probs.size(0), dtype=torch.bool, device=device)
        torch.ops.sgl_kernel.top_p_sampling_from_probs.default(
            probs,
            uniform_samples,
            samples,
            success,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            get_cuda_stream(),
        )
        return samples, success


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    deterministic: bool = True,
    check_nan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return _top_p_sampling_from_probs_internal(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_p), deterministic
    )


def _top_k_top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with probs.device as device:
        probs = probs.float()
        uniform_samples = uniform_samples.float()
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        success = torch.empty(probs.size(0), dtype=torch.bool, device=device)
        torch.ops.sgl_kernel.top_k_top_p_sampling_from_probs.default(
            probs,
            uniform_samples,
            samples,
            success,
            maybe_top_k_arr,
            top_k_val,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            get_cuda_stream(),
        )
        return samples, success


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs, uniform_samples, top_p, deterministic, check_nan=check_nan
        )
    elif filter_apply_order == "joint":
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return _top_k_top_p_sampling_from_probs_internal(
            probs,
            uniform_samples,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def _min_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    maybe_min_p_arr: Optional[torch.Tensor],
    min_p_val: float,
    deterministic: bool,
) -> torch.Tensor:
    with probs.device as device:
        probs = probs.float()
        uniform_samples = uniform_samples.float()
        maybe_min_p_arr = (
            maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        torch.ops.sgl_kernel.min_p_sampling_from_probs.default(
            probs,
            uniform_samples,
            samples,
            maybe_min_p_arr,
            min_p_val,
            deterministic,
            get_cuda_stream(),
        )
        return samples


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    deterministic: bool = True,
    check_nan: bool = False,
) -> torch.Tensor:
    if uniform_samples.dim() == 2:
        # Take the first row (round) of uniform_samples
        uniform_samples = uniform_samples[0]

    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return _min_p_sampling_from_probs_internal(
        probs, uniform_samples, *_to_tensor_scalar_tuple(min_p), deterministic
    )

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.dims_spec import (
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
    apply_dim_names,
    get_dim_names,
)
from sglang.srt.debug_utils.comparator.utils import Pair

_UNNAMED_TOKEN_DIM_FALLBACK: int = 0


def execute_token_aligner_concat_steps(
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    """Concat all steps in order, then truncate to min(total_x, total_y) tokens."""
    some_tensor: torch.Tensor = next(iter(tensor_of_step_pair.x.values()))
    token_dim: int = _resolve_token_dim(some_tensor)

    concatenated: Pair[torch.Tensor] = tensor_of_step_pair.map(
        lambda d: _concat_steps(d, dim=token_dim)
    )
    common: int = min(concatenated.x.shape[token_dim], concatenated.y.shape[token_dim])
    return concatenated.map(
        lambda t: _narrow_preserving_dim_names(t, dim=token_dim, start=0, length=common)
    )


def _narrow_preserving_dim_names(
    tensor: torch.Tensor, *, dim: int, start: int, length: int
) -> torch.Tensor:
    names: tuple[Optional[str], ...] = get_dim_names(tensor)
    narrowed: torch.Tensor = tensor.narrow(dim=dim, start=start, length=length)
    if names[0] is not None:
        narrowed = apply_dim_names(narrowed, [n for n in names if n is not None])
    return narrowed


def _resolve_token_dim(tensor: torch.Tensor) -> int:
    """Find the token/seq dim index. Falls back to dim 0 for unnamed tensors or
    tensors without a recognised token/seq dim."""
    names: tuple[Optional[str], ...] = get_dim_names(tensor)
    if names[0] is None:
        return _UNNAMED_TOKEN_DIM_FALLBACK
    for candidate in (TOKEN_DIM_NAME, SEQ_DIM_NAME):
        if candidate in names:
            return list(names).index(candidate)

    return _UNNAMED_TOKEN_DIM_FALLBACK


def _concat_steps(tensor_of_step: dict[int, torch.Tensor], *, dim: int) -> torch.Tensor:
    first_tensor: torch.Tensor = next(iter(tensor_of_step.values()))
    names: tuple[Optional[str], ...] = get_dim_names(first_tensor)
    result: torch.Tensor = torch.cat(
        [tensor_of_step[s] for s in sorted(tensor_of_step)], dim=dim
    )
    if names[0] is not None:
        result = apply_dim_names(result, [n for n in names if n is not None])
    return result

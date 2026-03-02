from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.dims import (
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
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
    return concatenated.map(lambda t: t.narrow(dim=token_dim, start=0, length=common))


def _resolve_token_dim(tensor: torch.Tensor) -> int:
    """Find the token/seq dim index. Falls back to dim 0 for unnamed tensors or
    tensors without a recognised token/seq dim."""
    if tensor.names[0] is None:
        return _UNNAMED_TOKEN_DIM_FALLBACK

    names: tuple[Optional[str], ...] = tensor.names
    for candidate in (TOKEN_DIM_NAME, SEQ_DIM_NAME):
        if candidate in names:
            return list(names).index(candidate)

    return _UNNAMED_TOKEN_DIM_FALLBACK


def _concat_steps(tensor_of_step: dict[int, torch.Tensor], *, dim: int) -> torch.Tensor:
    return torch.cat([tensor_of_step[s] for s in sorted(tensor_of_step)], dim=dim)

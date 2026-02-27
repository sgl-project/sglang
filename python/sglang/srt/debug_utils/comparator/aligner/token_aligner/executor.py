from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.dims import (
    TOKEN_DIM_NAME,
    resolve_dim_by_name,
    strip_dim_names,
)
from sglang.srt.debug_utils.comparator.utils import Pair

_UNNAMED_TOKEN_DIM_FALLBACK: int = 0


def _resolve_dim_or_fallback(tensor: torch.Tensor, name: str) -> int:
    if tensor.names[0] is None:
        return _UNNAMED_TOKEN_DIM_FALLBACK
    return resolve_dim_by_name(tensor, name)


def execute_token_aligner(
    plan: TokenAlignerPlan,
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
    *,
    token_dims: Pair[int] = Pair(x=0, y=0),
) -> Pair[torch.Tensor]:
    if not plan.locators.x.steps:
        return Pair(
            x=_make_empty(tensor_of_step=tensor_of_step_pair.x),
            y=_make_empty(tensor_of_step=tensor_of_step_pair.y),
        )

    return Pair(
        x=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.x,
            locator=plan.locators.x,
            token_dim=token_dims.x,
        ),
        y=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.y,
            locator=plan.locators.y,
            token_dim=token_dims.y,
        ),
    )


def _make_empty(
    *,
    tensor_of_step: dict[int, torch.Tensor],
) -> torch.Tensor:
    dummy: torch.Tensor = next(iter(tensor_of_step.values()))
    token_dim: int = _resolve_dim_or_fallback(dummy, TOKEN_DIM_NAME)
    shape: list[int] = list(dummy.shape)
    shape[token_dim] = 0
    return torch.empty(shape, dtype=dummy.dtype)


def _extract_and_stack_tokens(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    locator: TokenLocator,
    token_dim: int,
) -> torch.Tensor:
    some_tensor: torch.Tensor = next(iter(tensor_of_step.values()))
    token_dim: int = _resolve_dim_or_fallback(some_tensor, TOKEN_DIM_NAME)

    tokens: list[torch.Tensor] = [
        strip_dim_names(tensor_of_step[s]).select(dim=token_dim, index=i)
        for s, i in zip(locator.steps, locator.token_index_in_step)
    ]
    return torch.stack(tokens, dim=token_dim)

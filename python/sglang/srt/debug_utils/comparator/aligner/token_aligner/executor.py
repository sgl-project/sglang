from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def execute_token_aligner(
    plan: TokenAlignerPlan,
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
    *,
    token_dims: Pair[int] = Pair(x=0, y=0),
) -> Pair[torch.Tensor]:
    if not plan.locators.x.steps:
        return Pair(
            x=_make_empty(tensor_of_step=tensor_of_step_pair.x, token_dim=token_dims.x),
            y=_make_empty(tensor_of_step=tensor_of_step_pair.y, token_dim=token_dims.y),
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
    *, tensor_of_step: dict[int, torch.Tensor], token_dim: int
) -> torch.Tensor:
    dummy: torch.Tensor = next(iter(tensor_of_step.values()))
    shape: list[int] = list(dummy.shape)
    shape[token_dim] = 0
    return torch.empty(shape, dtype=dummy.dtype)


def _extract_and_stack_tokens(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    locator: TokenLocator,
    token_dim: int,
) -> torch.Tensor:
    tokens: list[torch.Tensor] = [
        tensor_of_step[s].select(dim=token_dim, index=i)
        for s, i in zip(locator.steps, locator.token_index_in_step)
    ]
    return torch.stack(tokens, dim=token_dim)

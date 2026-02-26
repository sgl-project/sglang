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
) -> Pair[torch.Tensor]:
    if not plan.locators.x.steps:
        dummy: torch.Tensor = next(iter(tensor_of_step_pair.x.values()))
        empty_shape: list[int] = [0] + list(dummy.shape[1:])
        empty: torch.Tensor = torch.empty(empty_shape, dtype=dummy.dtype)
        return Pair(x=empty, y=empty.clone())

    return Pair(
        x=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.x,
            locator=plan.locators.x,
        ),
        y=_extract_and_stack_tokens(
            tensor_of_step=tensor_of_step_pair.y,
            locator=plan.locators.y,
        ),
    )


def _extract_and_stack_tokens(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    locator: TokenLocator,
) -> torch.Tensor:
    tokens: list[torch.Tensor] = [
        tensor_of_step[s][i] for s, i in zip(locator.steps, locator.token_index_in_step)
    ]
    return torch.stack(tokens)

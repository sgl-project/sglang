from __future__ import annotations

import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def execute_token_aligner(
    plan: TokenAlignerPlan,
    tensor_pair: Pair[torch.Tensor],
) -> Pair[torch.Tensor]:
    if not plan.locators.x.token_index_in_step:
        empty_shape: list[int] = [0] + list(tensor_pair.x.shape[1:])
        empty: torch.Tensor = torch.empty(empty_shape, dtype=tensor_pair.x.dtype)
        return Pair(x=empty, y=empty.clone())

    return Pair(
        x=tensor_pair.x[plan.locators.x.token_index_in_step],
        y=tensor_pair.y[plan.locators.y.token_index_in_step],
    )

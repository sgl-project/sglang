from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """Standard combine input."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)

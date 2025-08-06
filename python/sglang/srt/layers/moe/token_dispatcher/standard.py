from __future__ import annotations

from typing import NamedTuple

from sglang.srt.layers.moe.token_dispatcher.base_dispatcher import (
    DispatchOutput,
    DispatchOutputFormat,
)


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.standard


assert isinstance(StandardDispatchOutput, DispatchOutput)

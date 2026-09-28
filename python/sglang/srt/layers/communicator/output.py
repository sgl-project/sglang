# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Outputs that carry work their consumer still owes."""

from typing import Callable, Optional, Union

import msgspec
import torch

from sglang.srt.distributed import GroupCoordinator


class UnreducedOutput(msgspec.Struct, frozen=True):
    """A layer output that still owes its sum, left for the next layer's input.
    Hand it to the next layer, or pass it through reduce_output() before reading
    it any other way. The producer says what is owed: one all-reduce over
    ``group`` that keeps the layout, or ``reduce_and_redistribute``."""

    partial: torch.Tensor
    group: Optional[GroupCoordinator] = None
    # Under attention DP: the reduction that also brings ``partial`` back to this
    # rank's tokens (a reduce-scatter, or an all-reduce then a scatter).
    reduce_and_redistribute: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class HandoffOutput(msgspec.Struct, frozen=True):
    """A layer output that still owes work only its producer knows how to do (a
    MoE's finalize and sum), left for the next layer's input. A fused kernel
    there may do that work together with its own; anything else passes it
    through reduce_output(), which calls ``complete()``."""

    def complete(self) -> torch.Tensor:
        """Do the owed work, unfused, and return the complete output."""
        raise NotImplementedError


def reduce_output(
    hidden_states: Union[torch.Tensor, UnreducedOutput, HandoffOutput, None],
) -> Optional[torch.Tensor]:
    """Run the work an UnreducedOutput or a HandoffOutput still owes; pass
    anything else through."""
    if isinstance(hidden_states, UnreducedOutput):
        if hidden_states.reduce_and_redistribute is not None:
            return hidden_states.reduce_and_redistribute(hidden_states.partial)
        return hidden_states.group.all_reduce(hidden_states.partial)
    if isinstance(hidden_states, HandoffOutput):
        return hidden_states.complete()
    return hidden_states

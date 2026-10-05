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


class OutputTransform(msgspec.Struct, frozen=True):
    """Transform a producer contribution before its residual update.

    Fields:
        apply: Callable(tensor) returning the transformed contribution.
        before_reduce_scatter: Permit this operation on a partial before
            reduce-scatter. Otherwise the transform follows completion.

    This explicitly preserves an implementation's order; it does not imply
    that arbitrary transforms commute with reduction or floating-point rounding.
    """

    apply: Callable[[torch.Tensor], torch.Tensor]
    before_reduce_scatter: bool = False


class UnreducedOutput:
    """Internal adapter value describing an unfinished reduction.

    Fields:
        partial: Tensor containing this rank's contribution to the sum.
        group: All-reduce group when no redistribution callable is supplied.
        reduce_to_dp_local: Callable(partial) completing the sum and moving
            it to destination rows; takes precedence over group.

    Exits hand this form to ResidualStream.record(), which keeps it as the
    contribution's owed work and exposes an opaque OwedOutput to models.
    Low-level adapters use complete_owed() before reading it. A group is
    required when reduce_to_dp_local is absent.
    """

    __slots__ = ("partial", "group", "reduce_to_dp_local")

    def __init__(
        self,
        partial: torch.Tensor,
        group: Optional[GroupCoordinator] = None,
        # Under attention DP: the reduction that also brings ``partial`` back to this
        # rank's tokens (a reduce-scatter, or an all-reduce then a scatter).
        reduce_to_dp_local: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.partial = partial
        self.group = group
        self.reduce_to_dp_local = reduce_to_dp_local

    def complete(self) -> torch.Tensor:
        """Complete the sum, on the destination rows when it moves them."""
        if self.reduce_to_dp_local is not None:
            return self.reduce_to_dp_local(self.partial)
        return self.group.all_reduce(self.partial)


class DeferredFinalize:
    """A layer output that still owes work only its producer knows how to do (a
    MoE's finalize and sum), left for the next layer's input or for a terminal
    norm that accepts it (residual_batch.final_norm(finalize_norm=...)). A fused kernel
    there may do that work together with its own; anything else passes it
    through complete_owed(), which calls ``complete()``."""

    __slots__ = ()

    def complete(self) -> torch.Tensor:
        """Do the owed work, unfused, and return the complete output."""
        raise NotImplementedError


def complete_owed(
    hidden_states: Union[torch.Tensor, UnreducedOutput, DeferredFinalize, None],
) -> Optional[torch.Tensor]:
    """Run the work an UnreducedOutput or a DeferredFinalize still owes; pass
    anything else through."""
    if isinstance(hidden_states, (UnreducedOutput, DeferredFinalize)):
        return hidden_states.complete()
    return hidden_states

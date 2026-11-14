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


class SumImplementation(msgspec.Struct, frozen=True):
    """A producer's implementation of its declared output sum.

    ``run(partial, forward_batch)`` completes the same group the output
    contract declares. It may use a model's custom collective; it does not
    change token rows or perform the residual update. A hidden-dimension
    scatter may leave a hidden shard for the producer update to consume.
    ``can_defer(batch)`` preserves that implementation's batch eligibility;
    None uses the ordinary exit eligibility. Neither callback is retained in
    a neighbouring stage's construction facts.
    """

    run: Optional[Callable]
    can_defer: Optional[Callable] = None

    def facts(self):
        return SumImplementation(None)


class UnreducedOutput(msgspec.Struct, frozen=True):
    """Internal adapter value describing an unfinished reduction.

    Fields:
        partial: Tensor containing this rank's contribution to the sum.
        group: All-reduce group when no redistribution callable is supplied.
        reduce_to_dp_local: Callable(partial) completing the sum and moving
            it to destination rows; takes precedence over group.
        complete_sum: Custom implementation of this group's sum on the same
            token rows; mutually exclusive with reduce_to_dp_local.


    Exits hand this form to ResidualStream.record(), which keeps it as the
    contribution's owed work and exposes an opaque OwedOutput to models.
    Low-level adapters use complete_owed() before reading it. A group is
    required when reduce_to_dp_local is absent.
    """

    partial: torch.Tensor
    group: Optional[GroupCoordinator] = None
    # Under attention DP: the reduction that also brings ``partial`` back to this
    # rank's tokens (a reduce-scatter, or an all-reduce then a scatter).
    reduce_to_dp_local: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    # Custom implementation of the same group, without token redistribution.
    complete_sum: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def __post_init__(self):
        if self.complete_sum is not None and self.reduce_to_dp_local is not None:
            raise ValueError("choose a custom sum or a DP redistribution")

    def complete(self) -> torch.Tensor:
        """Complete the sum, on the destination rows when it moves them."""
        if self.complete_sum is not None:
            return self.complete_sum(self.partial)
        if self.reduce_to_dp_local is not None:
            return self.reduce_to_dp_local(self.partial)
        return self.group.all_reduce(self.partial)


class DeferredFinalize(msgspec.Struct, frozen=True):
    """A layer output that still owes work only its producer knows how to do (a
    MoE's finalize and sum), left for the next layer's input or for a terminal
    norm that accepts it (residual_batch.final_norm(finalize_norm=...)). A fused kernel
    there may do that work together with its own; anything else passes it
    through complete_owed(), which calls ``complete()``."""

    @property
    def completion_group(self):
        from sglang.srt.layers.layer_boundary.layout import SumGroup

        return SumGroup.MOE_OUTPUT

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

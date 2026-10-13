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
"""Move prepared branch inputs and merge complete branch contributions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.layers.layer_boundary.ops import move_rows
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.layer_boundary.construction import StagePlan


def branch_input(
    plan: StagePlan,
    source: StagePlan,
    hidden_states: torch.Tensor,
    stream: ResidualStream,
    forward_batch: ForwardBatch,
) -> Tuple[torch.Tensor, ResidualStream]:
    """The FFN input and residual that ``source``'s boundary read for its
    own FFN, for this layer's FFN, which branches from the same input: moved
    to the rows this FFN needs and its residual's rows."""
    rows, residual_rows, _ = source._branch_rows(forward_batch)
    to, residual_to, _ = plan._branch_rows(forward_batch)
    if stream.pending is not None:
        raise RuntimeError("a branch must start from a prepared stage input")
    residual = stream.residual
    hidden_states = move_rows(hidden_states, rows, to, forward_batch)
    residual = move_rows(residual, residual_rows, residual_to, forward_batch)
    return (
        hidden_states,
        ResidualStream(residual),
    )


def branch_output(
    plan: StagePlan, hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """This layer's complete FFN output as a branch's contribution, which
    adds to the layer's output without writing the residual: moved to the
    rows the layer hands on."""
    rows, _, to = plan._branch_rows(forward_batch)
    return move_rows(hidden_states, rows, to, forward_batch)


def merge_branch(
    plan: StagePlan,
    contribution: torch.Tensor,
    hidden_states: torch.Tensor,
    stream: ResidualStream,
    source: StagePlan,
    forward_batch: ForwardBatch,
) -> Tuple[torch.Tensor, ResidualStream]:
    """A contribution from ``branch_output`` summed with what ``source``'s
    layer hands on, ``hidden_states`` and ``residual``, moved to the rows
    this layer hands on."""
    _, _, rows = source._branch_rows(forward_batch)
    _, _, to = plan._branch_rows(forward_batch)
    stream.check(hidden_states)
    if stream.pending is None:
        raise RuntimeError("branch merge requires a pending producer contribution")
    update = stream.pending.update
    hidden_states, residual = stream.finish(hidden_states)
    hidden_states = move_rows(hidden_states, rows, to, forward_batch)
    residual = move_rows(residual, rows, to, forward_batch)
    output = contribution + hidden_states
    stream.write(residual)
    return stream.leave(output, update), stream

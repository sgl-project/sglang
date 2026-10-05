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
"""One forward's residual and its producer's outstanding contribution."""

from typing import Optional, Union

import msgspec
import torch

from sglang.srt.layers.layer_boundary.layout import SumGroup, _sum_group
from sglang.srt.layers.layer_boundary.output import DeferredFinalize, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import ResidualUpdate


class DeclaredSum(msgspec.Struct, frozen=True):
    """A sum every output of this producer owes to its declared input edge."""

    group: SumGroup

    def complete(self, value):
        return _sum_group(self.group).all_reduce(value)


class Contribution(msgspec.Struct):
    """Own a producer's output, residual update and outstanding completion.

    Fields:
        value: Contribution tensor (an unreduced output's partial), or None
            for an opaque finalize handoff.
        update: Producer operation to apply before the consumer reads input.
        owed: The exit's unreduced output, a declared sum, a producer-specific
            finalize, or None.

    Completing owed work clears owed but does not apply the residual update.
    """

    value: Optional[torch.Tensor]
    update: ResidualUpdate
    owed: Union[UnreducedOutput, DeclaredSum, DeferredFinalize, None] = None

    def for_boundary(self):
        # These forms are private inputs to the existing fused-kernel adapters.
        if isinstance(self.owed, (UnreducedOutput, DeferredFinalize)):
            return self.owed
        return self.value

    def complete(self):
        if isinstance(self.owed, DeclaredSum):
            value = self.owed.complete(self.value)
        elif self.owed is not None:
            value = self.owed.complete()
        else:
            return self.value
        self.value, self.owed = value, None
        return value

    def release(self):
        """Drop the tensors of a contribution its consumer has taken."""
        self.value = None
        self.owed = None


class OwedOutput(msgspec.Struct, frozen=True):
    """Opaque model-facing handle. Only its boundary may read the contribution."""

    contribution: Contribution


class ResidualStream:
    """Own residual state for one forward or TBO microbatch.

    Args:
        residual: Already-written residual, or None before stack entry.

    Fields:
        residual: Residual tensor retained across stages, or None initially.
        pending: Contribution awaiting a residual update, or None. Its owed
            field separately records incomplete reduction/finalize work.

    The states are initial (both None), written (residual only), and pending.
    Prepare consumes pending once and writes the resulting residual, which
    releases the consumed contribution's tensors. Completing
    communication for capture leaves the update pending. Layouts and batch-path
    selection live on the bound stage, not on tensor values.
    """

    __slots__ = ("residual", "pending")

    def __init__(self, residual=None):
        self.residual = residual
        self.pending = None

    @classmethod
    def from_handoff(cls, hidden, residual, update, *, declared_sum=None):
        """Reconstruct a stream from a tensor handoff, including a TBO microbatch.

        Args:
            hidden: Received output or already-written residual tensor.
            residual: Separate residual tensor. None means hidden already holds
                the written residual, rather than an uninitialized stack.
            update: Producer operation for a separate contribution/residual pair.
            declared_sum: Static sum owed by hidden when residual is separate.

        Returns:
            (handle, stream). Use start() for a fresh, uninitialized stack instead.
        """
        stream = cls(residual)
        if residual is None:
            return stream.write(hidden), stream
        return stream.record(hidden, update, declared_sum=declared_sum), stream

    def write(self, residual):
        if self.pending is not None:
            # The written residual consumes the pending contribution. Handles
            # to it are now stale and must not keep its tensors alive.
            self.pending.release()
        self.residual = residual
        self.pending = None
        return residual

    def record(self, output, update, *, declared_sum=None):
        """Record one producer contribution without applying its residual update.

        Args:
            output: Complete tensor, UnreducedOutput, or producer-specific DeferredFinalize.
            update: Producer operation that the next boundary must apply.
            declared_sum: SumGroup statically owed by a raw tensor, or None. Cannot
                be combined with another output wrapper's completion contract.

        Returns:
            The tensor when complete, otherwise an opaque OwedOutput. Replacing an
            unconsumed contribution is an error; retain the returned handle.
        """
        if self.pending is not None:
            raise RuntimeError("cannot replace an unconsumed producer contribution")
        if declared_sum is not None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("a declared sum cannot carry another completion")
            contribution = Contribution(output, update, DeclaredSum(declared_sum))
        elif isinstance(output, UnreducedOutput):
            contribution = Contribution(output.partial, update, output)
        elif isinstance(output, DeferredFinalize):
            contribution = Contribution(None, update, output)
        else:
            contribution = Contribution(output, update)
        self.pending = contribution
        return OwedOutput(contribution) if contribution.owed is not None else output

    def check(self, hidden):
        pending = self.pending
        if pending is None:
            if isinstance(hidden, OwedOutput):
                raise RuntimeError("an owed handle has no contribution in this stream")
            if self.residual is not None and hidden is not self.residual:
                raise RuntimeError(
                    "written residual does not match the supplied output"
                )
            return
        if pending.owed is not None:
            valid = isinstance(hidden, OwedOutput) and hidden.contribution is pending
        else:
            valid = hidden is pending.value
        if not valid:
            raise RuntimeError(
                "output does not belong to this residual stream; "
                "change layer outputs through boundary accessors"
            )

    def input(self, hidden):
        self.check(hidden)
        if self.pending is not None:
            return self.pending.for_boundary(), self.residual
        # An initialized, written stream needs read only. An uninitialized
        # stream still uses the first stage's init_residual.
        return hidden, None

    def complete(self, hidden):
        self.check(hidden)
        return self.pending.complete() if self.pending is not None else hidden

    def snapshot(self, hidden):
        self.check(hidden)
        pending = self.pending
        if pending is None:
            return hidden.clone()
        if not pending.update.is_plain_add:
            raise NotImplementedError("snapshot requires a plain residual update")
        if pending.owed is None:
            value = pending.value
        elif isinstance(pending.owed, DeclaredSum):
            value = pending.owed.complete(pending.value.clone())
        elif isinstance(pending.owed, UnreducedOutput):
            value = msgspec.structs.replace(
                pending.owed, partial=pending.value.clone()
            ).complete()
        else:
            raise NotImplementedError("a finalize handoff requires main-output capture")
        return value.clone() if self.residual is None else value + self.residual

    def export(self, hidden, *, takes_handoff=False, preserve_declared=False):
        """Export the output/residual pair without closing or consuming the stream.

        Args:
            hidden: Current stream tensor or opaque owed handle.
            takes_handoff: Pass a producer-specific finalize handoff through for a
                terminal adapter that can consume it; otherwise complete it here.
            preserve_declared: Export a raw declared partial sum for a receiver
                whose incoming contract reconstructs that sum (for example PP).

        Returns:
            (output, residual), completing outstanding work unless explicitly
            preserved. Does not apply the pending residual update.
        """
        self.check(hidden)
        if self.pending is None:
            return hidden, None
        if preserve_declared and isinstance(self.pending.owed, DeclaredSum):
            return self.pending.value, self.residual
        if takes_handoff and isinstance(self.pending.owed, DeferredFinalize):
            return self.pending.owed, self.residual
        return self.pending.complete(), self.residual

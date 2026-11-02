"""Transport a compute operand without moving its source residual."""

from contextlib import nullcontext
from typing import Callable, Mapping, Optional

import msgspec

from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup, _sum_group
from sglang.srt.layers.layer_boundary.output import (
    DeferredFinalize,
    UnreducedOutput,
    complete_owed,
)
from sglang.srt.layers.layer_boundary.residual import bind_update


class TransportRows(msgspec.Struct, frozen=True):
    """The source/read, compute, and completed output rows of one adapter.

    The residual stays on source rows. ``group`` names the compute output's
    outstanding sum; None means the computation (e.g. EP combine) completes it.
    The boundary completes that sum and returns source rows, or publishes an
    explicit completion handle on source rows. It never applies the update.
    """

    source: Layout
    compute: Layout
    returned: Layout
    group: Optional[SumGroup] = None
    decode_group: Optional[SumGroup] = None


class RowTransport(msgspec.Struct, frozen=True):
    """Declare separate compute/residual rows and bind one invocation's moves.

    ``bind(input, batch, context, rows, overlapped_rows)`` returns an invocation
    with ``input`` and ``scope()``. The boundary checks its selected rows,
    prepares its output, completes owed work and runs its return move.
    Facts retain row contracts only.
    """

    rows: Mapping[BatchVariant, TransportRows]
    bind: Optional[Callable] = None
    overlapped_rows: Optional[Mapping[BatchVariant, TransportRows]] = None

    def facts(self):
        return RowTransport(
            dict(self.rows),
            overlapped_rows=(
                dict(self.overlapped_rows) if self.overlapped_rows is not None else None
            ),
        )


class ComputeTransport:
    """One prepared operand with declared rows and output completion policy.

    ``prepare_output`` exposes the compute result's outstanding work.
    The boundary completes it before ``return_output`` moves the result back;
    a move that performs the sum declares ``sum_in_return`` before compute.
    Residual update and consumer normalization stay with the boundary.
    """

    @staticmethod
    def scope():
        return nullcontext()

    is_overlapped = False
    defers_sum = False
    sum_in_return = False

    def release(self):
        pass

    def prepare_output(self, output):
        return output

    def return_output(self, output):
        return output


def validate_invocation(invocation, ordinary, overlapped):
    expected = overlapped if invocation.is_overlapped else ordinary
    if expected is None or invocation.rows != expected:
        raise ValueError("compute transport does not match its declared rows and sum")
    if invocation.defers_sum and (
        invocation.sum_in_return or expected.compute != expected.returned
    ):
        raise ValueError("a deferred transport sum must already be on its return rows")
    if invocation.sum_in_return and expected.group is None:
        raise ValueError("a transport move cannot complete an undeclared sum")


class TransportExit:
    """Consume a prepared transport once, then publish the residual update."""

    def __init__(self, stage, batch, *, update_context=None):
        self.stage = stage
        self.batch = batch
        self.stream = batch.residual_stream
        owner, self.invocation = self.stream.transport
        if owner is not stage:
            raise RuntimeError("compute transport belongs to another stage")
        self.path = stage.plan.path_for(batch)
        self.update = bind_update(self.path.output.update, batch, update_context)
        self.publish = (
            TransportExit._write
            if self.path.exit.writes_residual
            else TransportExit._record
        )
        self.scope = self.invocation.scope()
        self.used = False

    def __enter__(self):
        try:
            self.scope.__enter__()
        except BaseException:
            self.stream.transport = None
            raise
        return self

    def __exit__(self, *exc):
        try:
            result = self.scope.__exit__(*exc)
        finally:
            self.invocation.release()
            self.stream.transport = None
        if exc[0] is None and not self.used:
            raise RuntimeError("compute transport exited without finish")
        return result

    def finish(self, output):
        if self.used:
            raise RuntimeError("compute transport was already consumed")
        self.used = True
        output = self.invocation.prepare_output(output)
        rows = self.invocation.rows
        groups = tuple(
            group for group in (rows.group, rows.decode_group) if group is not None
        )
        if isinstance(output, UnreducedOutput):
            if output.complete_sum is not None or output.reduce_to_dp_local is not None:
                raise ValueError("operand transport requires its declared sum group")
            if not any(_sum_group(group) is output.group for group in groups):
                raise ValueError("transport output owes an undeclared sum group")
        elif (
            isinstance(output, DeferredFinalize)
            and output.completion_group not in groups
        ):
            raise ValueError("transport finalize owes an undeclared sum group")
        if not self.invocation.defers_sum or self.path.exit.writes_residual:
            output = complete_owed(output)
        output = self.invocation.return_output(output)
        return self.publish(self, output)

    def _write(self, output):
        return self.stream.write(self.update.update(output, self.stream.residual))

    def _record(self, output):
        return self.stream.record(output, self.update)

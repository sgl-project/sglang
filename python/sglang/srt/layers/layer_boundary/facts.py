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
"""What binding reads of a stage it does not run: the stage before or after
a stack on another pipeline rank, and the producer a bound stage records.

A stage's read, update and output transform are objects that hold the
layer's modules and state. Binding a neighbouring stage reads only the
capabilities they declare, so a neighbour, and the producer a stage keeps,
is recorded as those values alone: no module, bound method or declaration
history stays referenced once the stack is bound."""

from typing import Optional

import msgspec


class UpdateFacts(msgspec.Struct, frozen=True):
    """The capabilities a ResidualUpdate declares (see ResidualUpdate)."""

    is_plain_add: bool
    applied_at_exit: bool
    outlives_layer: bool
    writes_stream: bool
    quantized_sum: bool = False

    @classmethod
    def of(cls, update) -> "UpdateFacts":
        if isinstance(update, cls):
            return update
        return cls(
            is_plain_add=update.is_plain_add,
            applied_at_exit=update.applied_at_exit,
            outlives_layer=update.outlives_layer,
            writes_stream=update.writes_stream,
            quantized_sum=update.quantized_sum,
        )


class ReadFacts(msgspec.Struct, frozen=True):
    """The capabilities a ResidualReadout declares (see ResidualReadout).
    Binding takes kernels only from the reads of the stages it binds, never
    from a stage recorded by its facts, so these supply none."""

    is_plain_norm: bool
    reads_before_dp_gather: bool
    reads_after_attn_tp_gather: bool
    completing_fusions = ()
    gathering_reads = ()

    @classmethod
    def of(cls, read) -> "ReadFacts":
        if isinstance(read, cls):
            return read
        return cls(
            is_plain_norm=read.is_plain_norm,
            reads_before_dp_gather=read.reads_before_dp_gather,
            reads_after_attn_tp_gather=read.reads_after_attn_tp_gather,
        )


class TransformFacts(msgspec.Struct, frozen=True):
    """What an OutputTransform declares (see OutputTransform)."""

    before_reduce_scatter: bool = False

    @classmethod
    def of(cls, transform) -> Optional["TransformFacts"]:
        if transform is None or isinstance(transform, cls):
            return transform
        return cls(before_reduce_scatter=transform.before_reduce_scatter)


def residual_facts(ops):
    """A layer's reads and updates (a LayerResidualOps) reduced to the
    capabilities they declare, in the same shape: what its stages declare
    without the modules and state the reads and updates hold."""
    return type(ops)(
        attn_readout=ReadFacts.of(ops.attn_readout),
        attn_update=UpdateFacts.of(ops.attn_update),
        ffn_readout=ReadFacts.of(ops.ffn_readout),
        ffn_update=UpdateFacts.of(ops.ffn_update),
    )


def facts_of(declaration):
    """``declaration`` with its read, update and output transform reduced to
    the capabilities they declare, and without the declarations it was
    chained to. None stays None."""
    if declaration is None:
        return None
    from dataclasses import replace

    return replace(
        declaration,
        read=ReadFacts.of(declaration.read),
        update=UpdateFacts.of(declaration.update),
        output_transform=TransformFacts.of(declaration.output_transform),
        previous=None,
        prepared_from=None,
    )

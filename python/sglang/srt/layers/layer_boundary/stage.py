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
"""A stage's model-facing input and output boundary."""

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateAccumulator
from sglang.srt.layers.layer_boundary.adapters import branch
from sglang.srt.layers.layer_boundary.adapters.lora import (
    publish_attn_lora_rows,
    publish_ffn_lora_rows,
)
from sglang.srt.layers.layer_boundary.contracts import (
    BatchVariant,
    EntryPath,
    StageKind,
    StagePath,
)
from sglang.srt.layers.layer_boundary.ops import attn_tp_gather_input
from sglang.srt.layers.layer_boundary.residual.access import buffer, from_pp
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    PLAIN_ADD,
)
from sglang.srt.layers.layer_boundary.residual.batch import stream_of
from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum, ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_forward

if TYPE_CHECKING:
    from sglang.srt.layers.layer_boundary.construction import StagePlan
    from sglang.srt.layers.layer_boundary.factories import StageDeclaration


class StageBoundary:
    """Model-facing preparation and completion for one compute stage.

    Args:
        plan: Precomputed entry/exit paths, this stage's norm and input hooks.
        declaration: Immutable stage description used to construct related
            boundaries without exposing this object's execution plan.

    The boundary does not run attention or FFN computation. Per-forward residual
    state is read from forward_batch.residual_stream, not stored here.
    """

    def __init__(
        self,
        plan: "StagePlan",
        *,
        declaration: "StageDeclaration",
    ):
        self.plan = plan
        self.declaration = declaration

    @property
    def kind(self):
        return self.declaration.kind

    @property
    def norm(self):
        return self.plan.norm

    def entry(
        self, forward_batch: ForwardBatch, steps: Optional[StagePath] = None
    ) -> EntryPath:
        if steps is None:
            steps = self.plan.path_for(forward_batch)
        return steps.entry

    def _prepare(
        self,
        hidden_states,
        stream: ResidualStream,
        forward_batch: ForwardBatch,
        steps: Optional[StagePath] = None,
        **call,
    ):
        """The stage's input and the residual, from the previous stage's output
        (``call``: what the stage's read takes, e.g. the attention's
        ``quant_format``)."""
        entry = self.entry(forward_batch, steps)
        if (
            stream.pending is None
            and stream.residual is None
            and entry.declared_sum is not None
        ):
            hidden_states = stream.record(
                hidden_states, PLAIN_ADD, declared_sum=entry.declared_sum
            )
        if stream.pending is not None:
            call["update"] = stream.pending.update
            call["pending"] = stream.pending
        if stream.pending is None and stream.residual is not None:
            call["written"] = True
        hidden_states, residual = stream.input(hidden_states)
        hidden_states, residual = entry.prepare(
            hidden_states, residual, forward_batch, self.norm, **call
        )
        if entry.input_move is not None:
            hidden_states = entry.input_move(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        if entry.attn_input_adapter is not None:
            hidden_states = entry.attn_input_adapter(
                hidden_states, forward_batch, self.plan.qkv_latent_func
            )
        stream.write(residual)
        return hidden_states, stream

    @property
    def fusions(self):
        return self.plan.fusions

    @property
    def incoming_residual_rows(self):
        return self.plan.incoming_residual_rows

    @property
    def input_on_attn_tp_slices(self):
        return self.plan.input_on_attn_tp_slices

    def prepare(self, hidden_states, forward_batch, *, cache=None, **call):
        """Finish the previous contribution and produce this stage's compute input.

        Args:
            hidden_states: Tensor or opaque owed-output handle belonging to this
                forward's residual stream. Do not unwrap or modify a partial sum.
            forward_batch: Batch owning the stream and selecting the prepared path.
            cache: FFN-only NPU weight cache prefetched on the FFN input path.
            **call: Attention-only read/adapter options: quant_format,
                post_residual_addition, capture_gathered and capture. FFN
                stages do not accept these options. capture is a callback on
                the producer's rows; capture_gathered is a collector on the
                attention's compute-input rows. A capture callback is
                called as capture(value, owned=bool): owned=True means
                value is fresh storage it may keep, and owned=False means it
                must copy before retaining.

        Returns:
            Compute input in the selected read's format (possibly quantized).
            Updates forward_batch.residual_stream with the resulting residual.
        """
        if cache is not None:
            call["cache"] = cache
        prepare = (
            self._prepare_attention
            if self.kind is StageKind.ATTENTION
            else self._prepare_input
        )
        hidden_states, forward_batch.residual_stream = prepare(
            hidden_states, stream_of(forward_batch), forward_batch, **call
        )
        return hidden_states

    def finish(self, hidden_states, forward_batch):
        """Hand this attention's actual contribution to the following stage."""
        if self.kind is not StageKind.ATTENTION or not self.plan.finishes_directly:
            raise RuntimeError(
                "use the stage exit scope for an FFN or single-stage mixer"
            )
        produced = self.plan.produced(forward_batch)
        return stream_of(forward_batch).record(
            hidden_states,
            produced.update,
            declared_sum=produced.group if produced.always_partial else None,
        )

    def exit(self, forward_batch):
        """Select and scope one FFN or single-stage mixer's output decision.

        Args:
            forward_batch: Active batch whose stream receives the compute result.

        Returns:
            A context manager publishing reduction/finalize flags during compute.
            Call its finish(output) exactly once after successful compute, including
            when compute is skipped for an empty input. It completes or carries work
            using the same decision rather than selecting a second path.
        """
        result = (
            self.plan.output.ffn_exit(forward_batch, stream=stream_of(forward_batch))
            if self.kind is StageKind.FFN
            else self.plan.output.mixer_exit(
                forward_batch, stream=stream_of(forward_batch)
            )
        )
        return result

    def from_pp(self, tensors, forward_batch, *, allow_missing_residual: bool = False):
        """Restore the residual stream from a pipeline handoff.

        Args:
            tensors: PPProxyTensors containing hidden_states and, ordinarily, residual.
            forward_batch: Batch receiving a new stream.
            allow_missing_residual: Accept a handoff without the residual key for
                model paths that explicitly support it.

        Returns:
            Tensor or owed handle for prepare. A producer-written residual and a
            declared partial sum are reconstructed from the incoming contract.
        """
        hidden_states, residual = from_pp(
            tensors,
            residual_in_hidden=(
                self.declaration.previous is not None
                and self.declaration.previous.update.applied_at_exit
            ),
            allow_missing_residual=allow_missing_residual,
        )
        declared_sum = self.entry(forward_batch).declared_sum
        hidden_states, forward_batch.residual_stream = ResidualStream.from_handoff(
            hidden_states, residual, PLAIN_ADD, declared_sum=declared_sum
        )
        return hidden_states

    def snapshot(self, hidden_states, forward_batch):
        return stream_of(forward_batch).snapshot(hidden_states)

    def capture_output(self, hidden_states, forward_batch, *, skip_empty=False):
        """Capture an output while preserving the caller's reduction timing.

        Args:
            hidden_states: Current stream output or opaque owed handle.
            forward_batch: Batch owning that stream.
            skip_empty: Return no snapshot for a known zero-row output buffer.

        Returns:
            (updated_handle, snapshot). Dynamic owed work is completed on the main
            contribution; a declared sum is reduced only on the snapshot copy.
            Always use updated_handle afterwards. Plain residual updates are required.
        """
        stream = stream_of(forward_batch)
        if skip_empty:
            storage = buffer(hidden_states)
            if storage is not None and storage.shape[0] == 0:
                return hidden_states, None
        if stream.pending is None or not isinstance(stream.pending.owed, DeclaredSum):
            hidden_states = stream.complete(hidden_states)
        return hidden_states, stream.snapshot(hidden_states)

    def finish_complete_output(self, hidden_states, forward_batch):
        """Move an FFN output that compute already completed outside exit()
        (the operation-scheduled TBO path) onto the rows the layer hands on;
        it chooses no reduction step."""
        return self.plan.output.finish_complete_output(
            hidden_states, stream_of(forward_batch), forward_batch
        )

    def branch_input(self, source, hidden_states, forward_batch):
        hidden_states, stream = branch.branch_input(
            self.plan,
            source.plan,
            hidden_states,
            stream_of(forward_batch),
            forward_batch,
        )
        forward_batch.residual_stream = stream
        return hidden_states

    def branch_output(self, hidden_states, forward_batch):
        return branch.branch_output(self.plan, hidden_states, forward_batch)

    def merge_branch(self, contribution, hidden_states, source, forward_batch):
        hidden_states, stream = branch.merge_branch(
            self.plan,
            contribution,
            hidden_states,
            stream_of(forward_batch),
            source.plan,
            forward_batch,
        )
        forward_batch.residual_stream = stream
        return hidden_states

    def _prepare_input(self, hidden_states, stream, forward_batch, **call):
        plan = self.plan
        if self.kind is StageKind.ATTENTION:
            publish_attn_lora_rows(plan._publish_lora_layout)
            if (
                plan.paths.get(BatchVariant.SEQUENCE_PARALLEL) is not None
                and plan.enters_stack
            ):
                get_forward().set(
                    "sp_active", layernorm_sp.runs_sp(forward_batch.forward_mode)
                )
                if get_forward().sp_active:
                    hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)
        else:
            publish_ffn_lora_rows(
                plan._publish_lora_layout, self.entry(forward_batch).input_rows
            )
        return self._prepare(hidden_states, stream, forward_batch, **call)

    def _prepare_attention(
        self,
        hidden_states: torch.Tensor,
        stream: ResidualStream,
        forward_batch: ForwardBatch,
        capture_gathered: Optional[AuxHiddenStateAccumulator] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        quant_format: str = "",
        capture: Optional[Callable] = None,
    ):
        # Aux consumers need a materialized output before the input norm.
        # Complete communication here, preserving the existing add+norm kernel
        # and its FP32 accumulation. Its residual result also supplies capture.
        capture_before_read = capture is not None and (
            (stream.residual is None and stream.pending is None)
            or post_residual_addition is not None
        )
        if capture is not None:
            hidden_states = stream.complete(hidden_states)
            if capture_before_read:
                # Embeddings precede enter; HF deepstack capture precedes the
                # extra addition. Neither is the residual returned by the read.
                value, previous = stream.export(hidden_states)
                if previous is None:
                    capture(value)
                else:
                    capture(value + previous, owned=True)
        hidden_states, stream = self._prepare_input(
            hidden_states,
            stream,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )
        entry = self.entry(forward_batch)
        if capture is not None and not capture_before_read:
            value = stream.residual
            move = entry.capture_move
            if move is not None:
                value = move(value, forward_batch=forward_batch)
            keeps_input = entry.capture_preserves_residual
            capture(
                value,
                owned=entry.capture_move_allocates
                or (
                    move is None
                    and keeps_input is not None
                    and keeps_input(value, forward_batch)
                ),
            )
        if capture_gathered is not None:
            residual_value = stream.residual
            move = entry.input_move
            gathered_last_layer_output = (
                residual_value
                if move is None
                else move(
                    hidden_states=residual_value,
                    forward_batch=forward_batch,
                )
            )
            # Input gathers allocate fresh DP buffers (see attn_tp_gather_input).
            # Without a gather this is the mutable residual, so retention copies.
            capture_gathered.capture(
                gathered_last_layer_output,
                owned=move is attn_tp_gather_input
                or (
                    move is None
                    and entry.capture_preserves_residual is not None
                    and entry.capture_preserves_residual(residual_value, forward_batch)
                ),
            )
        return hidden_states, stream

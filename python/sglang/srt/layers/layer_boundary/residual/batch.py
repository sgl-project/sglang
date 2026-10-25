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
"""Residual access for one layer-stack invocation or TBO microbatch."""

from typing import NamedTuple

from sglang.srt.layers.layer_boundary.output import complete_owed
from sglang.srt.layers.layer_boundary.residual import access, bind_update
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def start(forward_batch):
    """Reset forward_batch.residual_stream at entry to a fresh layer-stack call.

    Call once before the first local stage on an embedding path, not per layer.
    Pipeline reception and TBO adapters reconstruct their own streams.
    """
    forward_batch.residual_stream = ResidualStream()


def stream_of(forward_batch):
    stream = forward_batch.residual_stream
    if stream is None:
        raise RuntimeError("start the layer stack before entering a stage")
    return stream


class ReplayHandoff(NamedTuple):
    value: object
    residual: object
    update: object
    written: bool


def clear(forward_batch):
    """Release invocation state on completion or an exception."""
    stream = forward_batch.residual_stream
    if stream is not None:
        if stream.transport is not None:
            stream.transport[1].release()
        if stream.pending is not None:
            stream.pending.release()
        stream.pending = stream.residual = stream.transport = None
    forward_batch.residual_stream = None


def export_replay_handoff(hidden_states, forward_batch):
    """Export a complete operand and reusable producer state across an eager break.

    The handoff does not complete owed work or apply the residual update.
    Replay rebinds dynamic producer metadata to the live batch.
    """
    from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD

    stream = stream_of(forward_batch)
    stream.check(hidden_states)
    pending = stream.pending
    if pending is not None and pending.owed is not None:
        raise RuntimeError("complete the contribution before a graph replay handoff")
    value, residual = stream.input(hidden_states)
    update = pending.update if pending is not None else PLAIN_ADD
    reusable = update.replay_update
    result = ReplayHandoff(
        value,
        residual,
        update if reusable is None else reusable,
        pending is None and stream.residual is not None,
    )
    clear(forward_batch)
    return result


def import_replay_handoff(handoff, forward_batch, *, update_context=None):
    """Reconstruct the residual state with replay's live producer metadata."""
    if handoff.written:
        if handoff.residual is not None:
            raise ValueError("a written replay handoff has no separate residual")
        set_written(handoff.value, forward_batch)
        return handoff.value
    if handoff.residual is None:
        start(forward_batch)
        return handoff.value
    hidden, forward_batch.residual_stream = ResidualStream.from_handoff(
        handoff.value,
        handoff.residual,
        bind_update(handoff.update, forward_batch, update_context),
    )
    return hidden


def written_residual(forward_batch):
    """The residual the last prepare wrote, for a stage whose compute reads it
    too (e.g. to write the next stream at its exit). Borrowed: do not modify."""
    return stream_of(forward_batch).residual


def complete_output(hidden_states, forward_batch):
    """Complete the contribution while retaining its pending residual update."""
    return stream_of(forward_batch).complete(hidden_states)


def take_output(hidden_states, forward_batch):
    """Release a terminal output whose residual update has already been written."""
    stream = stream_of(forward_batch)
    stream.check(hidden_states)
    if stream.pending is not None:
        raise RuntimeError("write the residual update before taking the final output")
    forward_batch.residual_stream = None
    return hidden_states


def _plain_output(hidden_states, forward_batch):
    stream = stream_of(forward_batch)
    stream.check(hidden_states)
    if stream.pending is not None and not stream.pending.update.is_plain_add:
        raise NotImplementedError(
            "output/residual pair requires a plain residual update; "
            "apply the producer update before export"
        )
    return stream


def final_norm(
    hidden_states,
    forward_batch,
    layernorm,
    capture=None,
    *,
    finalize_norm=None,
    skip_empty=False,
    **read_kwargs,
):
    """Complete the layer stack and apply its terminal normalization.

    Args:
        hidden_states: Current stream output or opaque owed handle.
        forward_batch: Batch owning the residual stream.
        layernorm: Final norm supporting the model's output/residual pair.
        capture: Optional callback retaining the same updated residual;
            it must copy borrowed storage when retention requires ownership.
        finalize_norm: Optional adapter with finalize(handoff, residual, gamma)
            for a producer-specific finalize handoff, returning (normed,
            updated residual); capture then gets that residual, borrowed.
            Without it, a handoff is completed unfused first. Its gamma is
            layernorm.gemma_weight, so the final norm must then be a
            GemmaRMSNorm.
        skip_empty: Skip the ordinary norm on a zero-row completed output.
        **read_kwargs: Extra arguments forwarded to the two-input final norm.

    Returns:
        Normalized tensor. Add and norm remain together to preserve the kernel's
        accumulation/rounding order; a snapshot is not used as the norm input.
    """
    hidden_states, residual = _plain_output(hidden_states, forward_batch).export(
        hidden_states, takes_handoff=finalize_norm is not None
    )
    # The terminal consumer now owns the pair. Do not keep layer buffers alive
    # through logits processing or the next forward on this batch.
    clear(forward_batch)
    from sglang.srt.layers.layer_boundary.output import DeferredFinalize

    if isinstance(hidden_states, DeferredFinalize):
        if residual is None:
            raise RuntimeError("invalid final deferred MoE handoff")
        hidden_states, residual = finalize_norm.finalize(
            handoff=hidden_states, residual=residual, gamma=layernorm.gemma_weight
        )
        if capture is not None:
            capture(residual)
        return hidden_states
    if skip_empty and hidden_states.shape[0] == 0:
        if capture is not None:
            capture(hidden_states)
        return hidden_states
    return access.final_norm_pair(
        hidden_states, residual, layernorm, capture, **read_kwargs
    )


def read_output(hidden_states, forward_batch, read, norm, *, read_context=None):
    """Consume the stack's output with an explicitly declared final read.

    This entry accepts producer updates through ``accepts_pending_update``.
    Completion candidates see the actual update before any collective; their
    fallback completes the owed output once, then applies update_and_read.
    A bindable read receives read_context once before consuming the output.
    Ordinary model final norms continue to use final_norm().
    """
    from sglang.srt.layers.layer_boundary.layout import _sum_group
    from sglang.srt.layers.layer_boundary.output import (
        DeferredFinalize,
        UnreducedOutput,
    )
    from sglang.srt.layers.layer_boundary.prepare import _run_read_fusions
    from sglang.srt.layers.layer_boundary.residual import (
        bind_read,
        takes_pending_update,
    )
    from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum

    stream = stream_of(forward_batch)
    stream.check(hidden_states)
    pending = stream.pending
    try:
        if pending is not None and not takes_pending_update(read, pending.update):
            raise NotImplementedError(
                "final read does not accept the pending producer update"
            )
        read = bind_read(read, forward_batch, read_context)
        if pending is None:
            value = stream.residual
            if value is None:
                value = read.init_residual(hidden_states)
            return read.read(value, norm)
        if not takes_pending_update(read, pending.update):
            raise NotImplementedError(
                "final read does not accept the pending producer update"
            )
        owed = pending.owed
        candidates = tuple(
            f
            for f in read.completing_fusions
            if not f.scatters
            and (
                (isinstance(owed, DeclaredSum) and f.completes is owed.group)
                or (
                    isinstance(owed, UnreducedOutput)
                    and f.carried
                    and owed.reduce_to_dp_local is None
                    and _sum_group(f.completes) is owed.group
                )
                or (
                    isinstance(owed, DeferredFinalize)
                    and f.carried
                    and f.completes is owed.completion_group
                )
            )
        )
        result = _run_read_fusions(
            candidates,
            pending.for_boundary(),
            stream.residual,
            forward_batch,
            norm,
            read=read,
            update=pending.update,
        )
        if result is None:
            value = pending.complete()
            result = read.update_and_read(pending.update, value, stream.residual, norm)
        return result
    finally:
        # The terminal consumer owns the result, including on an exception.
        clear(forward_batch)


def to_pp(hidden_states, forward_batch):
    """Export hidden_states and residual as PPProxyTensors.

    Args:
        hidden_states: Current stream output or opaque owed handle.
        forward_batch: Batch owning the stream.

    Every sum the output owes is completed before transport, a declared one
    too: the pipeline sends each tensor as one slice per attention-TP rank and
    gathers the slices back on the next rank, which reassembles only a tensor
    that every attention-TP rank holds alike. A stream the producer already
    wrote (MHC writes its streams at the FFN exit) has no separate residual
    and is sent as hidden_states alone; the receiver's from_pp reconstructs it
    as written.
    """
    hidden_states, residual = _plain_output(hidden_states, forward_batch).export(
        hidden_states
    )
    clear(forward_batch)
    tensors = {"hidden_states": hidden_states}
    if residual is not None:
        tensors["residual"] = residual
    return PPProxyTensors(tensors)


def snapshot(hidden_states, forward_batch):
    return stream_of(forward_batch).snapshot(hidden_states)


def add_to_output(hidden_states, forward_batch, extra):
    """Complete the output before adding an extra contribution exactly once.
    Leave its residual update for the next prepare call."""
    hidden_states = stream_of(forward_batch).complete(hidden_states)
    hidden_states.add_(extra)
    return hidden_states


def fold(hidden_states, forward_batch):
    """Finish a plain layer output and fold its residual into a complete value.
    The following stage receives it with no outstanding residual addition."""
    stream = stream_of(forward_batch)
    if stream.pending is not None and not stream.pending.update.is_plain_add:
        raise NotImplementedError("fold requires a plain residual update")
    hidden_states, residual = stream.export(hidden_states)
    hidden_states = complete_owed(hidden_states)
    if residual is not None:
        hidden_states = hidden_states + residual
    return stream.write(hidden_states)


def set_written(hidden_states, forward_batch):
    forward_batch.residual_stream = ResidualStream(hidden_states)
    return hidden_states

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

from sglang.srt.layers.layer_boundary.output import complete_owed
from sglang.srt.layers.layer_boundary.residual import access
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
            for a producer-specific finalize handoff; when a handoff arrives it
            cannot be combined with capture. Without it, a handoff is completed
            unfused first. Its gamma is layernorm.gemma_weight, so the final
            norm must then be a GemmaRMSNorm.
        skip_empty: Skip the ordinary norm on a zero-row completed output.
        **read_kwargs: Extra arguments forwarded to the two-input final norm.

    Returns:
        Normalized tensor. Add and norm remain together to preserve the kernel's
        accumulation/rounding order; a snapshot is not used as the norm input.
    """
    hidden_states, residual = stream_of(forward_batch).export(
        hidden_states, takes_handoff=finalize_norm is not None
    )
    # The terminal consumer now owns the pair. Do not keep layer buffers alive
    # through logits processing or the next forward on this batch.
    forward_batch.residual_stream = None
    from sglang.srt.layers.layer_boundary.output import DeferredFinalize

    if isinstance(hidden_states, DeferredFinalize):
        if residual is None:
            raise RuntimeError("invalid final deferred MoE handoff")
        if capture is not None:
            raise RuntimeError(
                "final handoff capture requires an explicit capture adapter"
            )
        hidden_states, _ = finalize_norm.finalize(
            handoff=hidden_states, residual=residual, gamma=layernorm.gemma_weight
        )
        return hidden_states
    if skip_empty and hidden_states.shape[0] == 0:
        return hidden_states
    return access.final_norm_pair(
        hidden_states, residual, layernorm, capture, **read_kwargs
    )


def to_pp(hidden_states, forward_batch, *, preserve_declared=True):
    """Export hidden_states and residual as PPProxyTensors.

    Args:
        hidden_states: Current stream output or opaque owed handle.
        forward_batch: Batch owning the stream.
        preserve_declared: True (the default) sends a statically declared
            partial sum unreduced for the receiver's from_pp to complete; pass
            False only when the receiver does not declare that sum.

    Runtime-selected completion work is finished before transport. A stream
    the producer already wrote (MHC writes its streams at the FFN exit) has no
    separate residual and is sent as hidden_states alone; the receiver's
    from_pp reconstructs it as written.
    """
    stream = stream_of(forward_batch)
    if stream.pending is not None and not stream.pending.update.is_plain_add:
        hidden_states = stream.flush_update(hidden_states)
    hidden_states, residual = stream.export(
        hidden_states, preserve_declared=preserve_declared
    )
    forward_batch.residual_stream = None
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

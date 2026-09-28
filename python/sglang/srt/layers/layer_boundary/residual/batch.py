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

from sglang.srt.layers.layer_boundary.residual import access
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def start(forward_batch):
    forward_batch.residual_stream = ResidualStream()


def current(forward_batch):
    stream = forward_batch.residual_stream
    if stream is None:
        raise RuntimeError("start the layer stack before entering a stage")
    return stream


def complete_output(hidden_states, forward_batch):
    """Complete the contribution while retaining its pending residual update."""
    return current(forward_batch).complete(hidden_states)


def take_output(hidden_states, forward_batch):
    """Release a terminal output whose residual update has already been written."""
    stream = current(forward_batch)
    stream.check(hidden_states)
    if stream.pending is not None:
        raise RuntimeError("write the residual update before taking the final output")
    forward_batch.residual_stream = None
    return hidden_states


def norm(
    hidden_states,
    forward_batch,
    layernorm,
    capture_output=None,
    *,
    handoff_norm=None,
    skip_empty=False,
    **read_kwargs,
):
    hidden_states, residual = current(forward_batch).finish(
        hidden_states, takes_handoff=handoff_norm is not None
    )
    # The terminal consumer now owns the pair. Do not keep layer buffers alive
    # through logits processing or the next forward on this batch.
    forward_batch.residual_stream = None
    from sglang.srt.layers.layer_boundary.output import HandoffOutput

    if isinstance(hidden_states, HandoffOutput):
        if residual is None:
            raise RuntimeError("invalid final deferred MoE handoff")
        if capture_output is not None:
            raise RuntimeError(
                "final handoff capture requires an explicit capture adapter"
            )
        hidden_states, _ = handoff_norm.finalize(
            handoff=hidden_states, residual=residual, gamma=layernorm.gemma_weight
        )
        return hidden_states
    if skip_empty and hidden_states.shape[0] == 0:
        return hidden_states
    return access.norm_output(
        hidden_states, residual, layernorm, capture_output, **read_kwargs
    )


def to_pp(hidden_states, forward_batch, *, preserve_declared=True):
    hidden_states, residual = current(forward_batch).finish(
        hidden_states, preserve_declared=preserve_declared
    )
    forward_batch.residual_stream = None
    return PPProxyTensors({"hidden_states": hidden_states, "residual": residual})


def snapshot(hidden_states, forward_batch):
    return current(forward_batch).snapshot(hidden_states)


def add_to_output(hidden_states, forward_batch, extra):
    output, _ = access.add_to_output(hidden_states, current(forward_batch), extra)
    return output


def fold(hidden_states, forward_batch):
    output, _ = access.fold(hidden_states, current(forward_batch))
    return output


def written(hidden_states, forward_batch):
    forward_batch.residual_stream = ResidualStream(hidden_states)
    return hidden_states

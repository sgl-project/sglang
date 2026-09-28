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

from typing import TYPE_CHECKING, Optional

from sglang.srt.layers.communicator.boundary import BoundarySteps, StageEntry
from sglang.srt.layers.communicator.residual.add_norm import ADD
from sglang.srt.layers.communicator.residual.batch import current
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.communicator.layer import LayerCommunicator


class StageCommunicator:
    """One stage of a layer, its attention or its FFN: the boundary into it,
    run with the stage's norm on the steps the layer chose for the batch."""

    def __init__(self, layer: "LayerCommunicator", name: str, norm_name: str):
        self._layer = layer
        self._name = name
        self.norm = getattr(layer, norm_name)

    def entry(
        self, forward_batch: ForwardBatch, steps: Optional[BoundarySteps] = None
    ) -> StageEntry:
        if steps is None:
            steps = self._layer._batch_steps(forward_batch)
        entry = getattr(steps, self._name)
        if entry is None:
            raise NotImplementedError(
                f"a layer that is one stage has no {self._name} stage"
            )
        return entry

    def _prepare(
        self,
        hidden_states,
        residual,
        forward_batch: ForwardBatch,
        steps: Optional[BoundarySteps] = None,
        **call,
    ):
        """The stage's input and the residual, from the previous stage's output
        (``call``: what the stage's read takes, e.g. the attention's
        ``quant_format``)."""
        entry = self.entry(forward_batch, steps)
        stream = current(forward_batch)
        if residual is not stream:
            raise RuntimeError("residual alias belongs to a different invocation")
        if (
            stream.pending is None
            and stream.residual is None
            and entry.input_sum is not None
        ):
            hidden_states = stream.leave(
                hidden_states, ADD, declared_sum=entry.input_sum
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
        if entry.handoff is not None:
            hidden_states = entry.handoff(
                hidden_states, forward_batch, self._layer.qkv_latent_func
            )
        stream.write(residual)
        return hidden_states, stream

    @property
    def input_rows(self):
        return self._layer.input_rows

    @property
    def input_on_attention_tp_slices(self):
        return self._layer.input_on_attention_tp_slices

    def prepare(self, hidden_states, forward_batch, *, cache=None, **call):
        stream = current(forward_batch)
        if self._name == "attention":
            hidden_states, stream = (
                self._layer.prepare_attn_and_capture_last_layer_outputs(
                    hidden_states, stream, forward_batch, **call
                )
            )
        else:
            steps = self._layer._batch_steps(forward_batch)
            self._layer.publish_mlp_lora_layout(steps)
            if cache is not None:
                call["cache"] = cache
            hidden_states, stream = self._prepare(
                hidden_states, stream, forward_batch, steps, **call
            )
        forward_batch.residual_stream = stream
        return hidden_states

    def finish(self, hidden_states, forward_batch):
        """Hand this attention's actual contribution to the following stage."""
        if self._name != "attention" or self._layer.stage_edges is not None:
            raise RuntimeError(
                "use the stage exit scope for an FFN or single-stage mixer"
            )
        return current(forward_batch).leave(
            hidden_states,
            self._layer._residual.attention_update,
            declared_sum=self._layer._batch_steps(forward_batch).ffn.input_sum,
        )

    def exit(self, forward_batch):
        result = (
            self._layer.ffn_exit(forward_batch)
            if self._name == "ffn"
            else self._layer.mixer_exit(forward_batch)
        )
        result._stream = current(forward_batch)
        return result

    def from_pp(self, tensors, forward_batch, **kwargs):
        hidden_states, stream = self._layer.from_pp(tensors, forward_batch, **kwargs)
        forward_batch.residual_stream = stream
        return hidden_states

    def snapshot(self, hidden_states, forward_batch, *, at_input=False):
        return self._layer.snapshot(
            hidden_states, current(forward_batch), at_input=at_input
        )

    def capture_output(self, hidden_states, forward_batch, **kwargs):
        return self._layer.capture_output(
            hidden_states, current(forward_batch), **kwargs
        )

    def postprocess(self, hidden_states, forward_batch):
        hidden_states, stream = self._layer.postprocess_layer(
            hidden_states, current(forward_batch), forward_batch
        )
        forward_batch.residual_stream = stream
        return hidden_states

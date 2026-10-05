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
"""Gated hyper-connection residual streams."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.srt.layers.layer_boundary.residual import LayerResidualOps
from sglang.srt.runtime_context import get_parallel


@dataclass
class IHCState:
    """A layer's residual as hc_mult gated streams: a stage reads its input by
    mixing the streams down with per-stream gates, and writes its output back
    scaled by the post gates the same read produced. Unlike the hyper-connection
    streams in mhc.py there is no combination matrix, so a read produces one
    coefficient rather than a pair. ``residual_ops()`` gives the reads and
    updates the layer's two stages declare. Parameters belong to the owning
    layer's modules; this state only holds the coefficient shared between a
    stage's read and its write-back.

    Args:
        expand: Widens the layer stack's 2-D input into the stream
            representation, and passes an already widened residual through.
        attn_pre: The attention stage's read.
        ffn_pre: The FFN stage's read.
        attn_post: Writes the attention output back into the streams.
        ffn_post: Writes the FFN output back into the streams.
        post_pre: The attention write-back fused with the FFN read, or None to
            run the two in sequence.
    """

    expand: Callable
    attn_pre: Callable
    ffn_pre: Callable
    attn_post: Callable
    ffn_post: Callable
    post_pre: Optional[Callable] = None
    # Produced by a stage's read and consumed by that same stage's write-back.
    post_gate: Optional[torch.Tensor] = None

    def read_attn_input(self, residual, out_norm: Optional[torch.nn.Module] = None):
        hidden_states, self.post_gate, residual = self.attn_pre(residual, out_norm)
        return hidden_states, residual

    def update_and_read_ffn_input(
        self, hidden_states, residual, out_norm: Optional[torch.nn.Module] = None
    ):
        if self.post_pre is not None:
            hidden_states, self.post_gate, residual = self.post_pre(
                hidden_states, residual, self.post_gate, out_norm
            )
            return hidden_states, residual
        residual = self.apply_attn_post(hidden_states, residual)
        hidden_states, self.post_gate, residual = self.ffn_pre(residual, out_norm)
        return hidden_states, residual

    def apply_attn_post(self, hidden_states, residual):
        return self.attn_post(hidden_states, residual, self.post_gate)

    def apply_ffn_post(self, hidden_states, residual):
        return self.ffn_post(hidden_states, residual, self.post_gate)

    def clear_coefficients(self):
        self.post_gate = None

    def slice_residual_attn_tp(self, residual):
        parallel = get_parallel()
        rank, size = parallel.attn_tp_rank, parallel.attn_tp_size
        self.post_gate = self.post_gate.tensor_split(size)[rank]
        return residual.tensor_split(size)[rank]

    def gather_residual_attn_tp(self, residual):
        raise NotImplementedError(
            "Unsupported: iHC post-gate allgather is not implemented."
        )

    def residual_ops(self) -> LayerResidualOps:
        return LayerResidualOps(
            attn_readout=_AttnReadout(self),
            attn_update=_AttnUpdate(self),
            ffn_readout=_FfnReadout(self),
            ffn_update=_FfnUpdate(self),
        )


class _AttnReadout:
    """The gated mix of the streams and the input norm, from streams that
    already hold the previous layer's output: an iHC layer takes its input
    written back."""

    is_plain_norm = False
    reads_before_dp_gather = False

    def __init__(self, state: IHCState):
        self.state = state

    def init_residual(self, hidden_states):
        return self.state.expand(hidden_states)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        if quant_format:
            raise NotImplementedError(f"an iHC attention input in {quant_format=}")
        return self.state.read_attn_input(residual, out_norm=norm)

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        raise NotImplementedError("an iHC layer takes its input written back")


class _AttnUpdate:
    """The gated write-back with the coefficient the attention's read produced.
    It is not a plain add, so it runs only once the sum it writes in is
    complete."""

    is_plain_add = False
    applied_at_exit = False
    outlives_layer = False

    def __init__(self, state: IHCState):
        self.state = state

    def update(self, hidden_states, residual):
        return self.state.apply_attn_post(hidden_states, residual)

    def slice_residual_attn_tp(self, residual):
        return self.state.slice_residual_attn_tp(residual)

    def gather_residual_attn_tp(self, residual):
        return self.state.gather_residual_attn_tp(residual)


class _FfnReadout:
    """The attention output's write-back and the FFN input's gated mix and
    norm, fused in post_pre when the layer provides it."""

    is_plain_norm = False
    reads_before_dp_gather = False

    def __init__(self, state: IHCState):
        self.state = state

    def init_residual(self, hidden_states):
        return self.state.expand(hidden_states)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        raise NotImplementedError("an iHC FFN input read without its attention")

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        if not isinstance(update, _AttnUpdate) or update.state is not self.state:
            raise NotImplementedError(f"an iHC FFN input after {update=}")
        return self.state.update_and_read_ffn_input(
            hidden_states, residual, out_norm=norm
        )


class _FfnUpdate:
    """The gated write-back with the coefficient the FFN input's read produced,
    which this layer runs itself. The streams stay widened: unlike mhc.py the
    layer stack's terminal read is a learned head rather than a contraction,
    so it belongs to the final norm."""

    is_plain_add = False
    applied_at_exit = True
    outlives_layer = False

    def __init__(self, state: IHCState):
        self.state = state

    def update(self, hidden_states, residual):
        hidden_states = self.state.apply_ffn_post(hidden_states, residual)
        self.state.clear_coefficients()
        return hidden_states

    def slice_residual_attn_tp(self, residual):
        return self.state.slice_residual_attn_tp(residual)

    def gather_residual_attn_tp(self, residual):
        return self.state.gather_residual_attn_tp(residual)

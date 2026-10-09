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
"""Gated-injection hyper-connection residual streams."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.srt.layers.layer_boundary.residual import LayerResidualOps
from sglang.srt.runtime_context import get_parallel


@dataclass
class GatedResidualState:
    """A layer's residual as hc_count streams flattened into one tensor: a
    stage reads its input by normalizing the streams and mixing them down with
    per-stream gates, and writes its output back scaled by an injection
    coefficient computed from that same normalized residual.

    Unlike the hyper-connection streams in mhc.py and ihc.py, whose read
    produces the coefficient its write-back consumes, the coefficient here is
    computed at write time. What has to survive from a stage's read to its
    write-back is therefore the normalized residual itself, which this state
    holds. Parameters belong to the owning layer's modules.

    Args:
        expand: Widens the layer stack's input into the stream representation,
            and passes an already widened residual through.
        attn_mix: The attention stage's read. A layer whose own contribution
            joins the streams before that read (the Qwen4 PLE embedding) folds
            it in here, so this state does not need to know about it.
        ffn_mix: The FFN stage's read.
        attn_combine: Writes the attention output back into the streams.
        ffn_combine: Writes the FFN output back into the streams.
        attn_reads_every_row: Whether attn_mix needs every row of the
            attention's, as a contribution computed for all of them does.
    """

    expand: Callable
    attn_mix: Callable
    ffn_mix: Callable
    attn_combine: Callable
    ffn_combine: Callable
    attn_reads_every_row: bool = False
    # Produced by a stage's read and consumed by that same stage's write-back.
    normed: Optional[torch.Tensor] = None

    def _read(self, mix, residual, out_norm):
        if out_norm is not None:
            raise NotImplementedError(
                "a gated hyper-connection read with a separate norm; the read "
                "normalizes the streams itself"
            )
        hidden_states, (residual, self.normed) = mix(residual)
        return hidden_states, residual

    def read_attn_input(self, residual, out_norm=None):
        return self._read(self.attn_mix, residual, out_norm)

    def update_and_read_ffn_input(self, hidden_states, residual, out_norm=None):
        residual = self.apply_attn_combine(hidden_states, residual)
        return self._read(self.ffn_mix, residual, out_norm)

    def apply_attn_combine(self, hidden_states, residual):
        return self.attn_combine(hidden_states, (residual, self.normed))

    def apply_ffn_combine(self, hidden_states, residual):
        return self.ffn_combine(hidden_states, (residual, self.normed))

    def clear_coefficients(self):
        self.normed = None

    def slice_residual_attn_tp(self, residual):
        parallel = get_parallel()
        rank, size = parallel.attn_tp_rank, parallel.attn_tp_size
        self.normed = self.normed.tensor_split(size)[rank]
        return residual.tensor_split(size)[rank]

    def gather_residual_attn_tp(self, residual):
        raise NotImplementedError(
            "Unsupported: gated hyper-connection normed-residual allgather is "
            "not implemented."
        )

    def residual_ops(self) -> LayerResidualOps:
        return LayerResidualOps(
            attn_readout=_AttnReadout(self),
            attn_update=_AttnUpdate(self),
            ffn_readout=_FfnReadout(self),
            ffn_update=_FfnUpdate(self),
        )


class _AttnReadout:
    """The normalized, gated mix of the streams, from streams that already hold
    the previous layer's output: such a layer takes its input written back."""

    is_plain_norm = False
    reads_before_dp_gather = False

    def __init__(self, state: GatedResidualState):
        self.state = state

    @property
    def reads_after_attn_tp_gather(self):
        return self.state.attn_reads_every_row

    def init_residual(self, hidden_states):
        return self.state.expand(hidden_states)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        if quant_format:
            raise NotImplementedError(
                f"a gated hyper-connection attention input in {quant_format=}"
            )
        return self.state.read_attn_input(residual, out_norm=norm)

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        raise NotImplementedError(
            "a gated hyper-connection layer takes its input written back"
        )


class _AttnUpdate:
    """The gated injection of the attention output, using the normalized
    residual its read produced. The injection is not a plain add, so it runs
    only once the sum it writes in is complete."""

    is_plain_add = False
    applied_at_exit = False
    outlives_layer = False

    def __init__(self, state: GatedResidualState):
        self.state = state

    def update(self, hidden_states, residual):
        return self.state.apply_attn_combine(hidden_states, residual)

    def slice_residual_attn_tp(self, residual):
        return self.state.slice_residual_attn_tp(residual)

    def gather_residual_attn_tp(self, residual):
        return self.state.gather_residual_attn_tp(residual)


class _FfnReadout:
    """The attention output's injection and the FFN input's mix."""

    is_plain_norm = False
    reads_before_dp_gather = False

    def __init__(self, state: GatedResidualState):
        self.state = state

    def init_residual(self, hidden_states):
        return self.state.expand(hidden_states)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        raise NotImplementedError(
            "a gated hyper-connection FFN input read without its attention"
        )

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        if not isinstance(update, _AttnUpdate) or update.state is not self.state:
            raise NotImplementedError(
                f"a gated hyper-connection FFN input after {update=}"
            )
        return self.state.update_and_read_ffn_input(
            hidden_states, residual, out_norm=norm
        )


class _FfnUpdate:
    """The gated injection of the FFN output, which this layer runs itself. The
    streams stay widened: the layer stack's terminal read mixes them down."""

    is_plain_add = False
    applied_at_exit = True
    outlives_layer = False

    def __init__(self, state: GatedResidualState):
        self.state = state

    def update(self, hidden_states, residual):
        hidden_states = self.state.apply_ffn_combine(hidden_states, residual)
        self.state.clear_coefficients()
        return hidden_states

    def slice_residual_attn_tp(self, residual):
        return self.state.slice_residual_attn_tp(residual)

    def gather_residual_attn_tp(self, residual):
        return self.state.gather_residual_attn_tp(residual)

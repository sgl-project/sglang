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
"""Deferred channelwise convolution followed by residual add and normalization."""

from typing import Callable, NamedTuple, Optional

import msgspec

from sglang.srt.layers.layer_boundary.residual import PendingUpdateRead
from sglang.srt.layers.layer_boundary.residual.add_norm import NORM_READOUT, PLAIN_ADD


class _UpdateFacts:
    bind = None
    replay_update = None
    is_plain_add = False
    applied_at_exit = False
    outlives_layer = True
    writes_stream = False
    quantized_sum = False

    def slice_residual_attn_tp(self, residual):
        raise NotImplementedError("short convolution needs its original token rows")

    def gather_residual_attn_tp(self, residual):
        raise NotImplementedError("short convolution needs its original token rows")


class ShortConvUpdate(_UpdateFacts, msgspec.Struct, frozen=True):
    """Producer parameters; bind() creates state owned by one contribution.

    A scattered completion produces a hidden shard. The convolution reads
    that shard and ``gather_hidden`` reconstructs the hidden dimension before
    the residual add. These operations leave the token layout unchanged.
    """

    convolve: Callable
    group: object
    sum_group: object
    gather_hidden: Optional[Callable] = None
    can_fuse: Optional[Callable] = None

    def bind(self, forward_batch, positions=None):
        return ShortConvInvocation(
            self,
            forward_batch,
            forward_batch.positions if positions is None else positions,
        )


class ShortConvInvocation(_UpdateFacts, msgspec.Struct, frozen=True):
    spec: ShortConvUpdate
    forward_batch: object
    positions: object

    @property
    def replay_update(self):
        return self.spec

    def contribution(self, hidden_states):
        batch = self.forward_batch
        value = self.spec.convolve(
            hidden_states, self.positions[: hidden_states.shape[0]], batch
        )
        if self.spec.gather_hidden is not None:
            value = self.spec.gather_hidden(value)
        return value

    def update(self, hidden_states, residual):
        return self.contribution(hidden_states) + residual


class NormalizationHandoff(NamedTuple):
    """Completed producer contribution and residual, before add+norm."""

    value: object
    residual: object


class ShortConvRead(PendingUpdateRead):
    """Read the actual producer's convolution and keep add+norm together."""

    bind = None

    is_plain_norm = False
    reads_before_dp_gather = True
    reads_after_attn_tp_gather = True
    gathering_reads = ()

    def __init__(self, completing_fusions=()):
        self.completing_fusions = completing_fusions

    @staticmethod
    def accepts_pending_update(update):
        return update.is_plain_add or isinstance(
            update, (ShortConvUpdate, ShortConvInvocation)
        )

    def export_normalization_handoff(self, hidden_states, forward_batch):
        """Complete producer work with live metadata, leaving add+norm to capture.

        Piecewise prefill completes the unfused sum and convolution before
        exporting the unadded pair to the captured normalization.
        """
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        read = _NORMALIZATION_PAIR_READ
        return NormalizationHandoff(
            *residual_batch.read_output(hidden_states, forward_batch, read, _pair)
        )

    def read_normalization_handoff(self, handoff, norm):
        """Consume the complete pair with the original add+norm kernel."""
        if handoff.residual is None:
            return self.read(handoff.value, norm)
        return NORM_READOUT.update_and_read(
            PLAIN_ADD, handoff.value, handoff.residual, norm
        )

    def init_residual(self, hidden_states):
        return hidden_states

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        return NORM_READOUT.read(residual, norm, quant_format, post_residual_addition)

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        if update.is_plain_add:
            return NORM_READOUT.update_and_read(
                update,
                hidden_states,
                residual,
                norm,
                quant_format,
                post_residual_addition,
            )
        if not isinstance(update, ShortConvInvocation):
            raise TypeError("short-conv read requires the producer's bound invocation")
        value = update.contribution(hidden_states)
        # Normalize the contribution/residual pair without a separately rounded add.
        return NORM_READOUT.update_and_read(
            PLAIN_ADD, value, residual, norm, quant_format, post_residual_addition
        )


def _pair(value, residual=None, **kwargs):
    return value if residual is None else (value, residual)


class _NormalizationPairRead(ShortConvRead):
    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        return residual, None


_NORMALIZATION_PAIR_READ = _NormalizationPairRead()

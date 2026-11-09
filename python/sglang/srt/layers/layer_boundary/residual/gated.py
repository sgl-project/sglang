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
from functools import partial
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.layers.layer_boundary.contracts import ReadoutFusion
from sglang.srt.layers.layer_boundary.facts import residual_facts
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.residual import LayerResidualOps
from sglang.srt.runtime_context import get_exec, get_parallel


@dataclass
class GatedResidualState:
    """A layer's residual as hc_count streams flattened into one tensor: a
    stage reads its input by normalizing the streams and mixing them down with
    per-stream gates, and writes its output back scaled by an injection
    coefficient computed from that same normalized residual.

    The read carries its normalized residual to write-back. Decode may also
    prepare gate partials during the read; otherwise write-back computes the
    coefficient from that normalized residual. Both follow the residual rows
    through any slicing and are cleared after the FFN write-back.
    Parameters belong to the owning layer's modules.

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
        attn_combine_owing_sum, ffn_combine_owing_sum: Optional write-backs of
            an output that still owes its sum: ``(partial, residuals,
            coordinator)`` completes ``coordinator.all_reduce(partial)`` inside
            the write-back, or returns None having launched nothing.
    """

    expand: Callable
    attn_mix: Callable
    ffn_mix: Callable
    attn_combine: Callable
    ffn_combine: Callable
    attn_reads_every_row: bool = False
    attn_combine_owing_sum: Optional[Callable] = None
    ffn_combine_owing_sum: Optional[Callable] = None
    # Produced by a stage's read and consumed by that same stage's write-back.
    normed: Optional[torch.Tensor] = None
    gate_partials: Optional[torch.Tensor] = None

    def _read(self, mix, residual, out_norm):
        if out_norm is not None:
            raise NotImplementedError(
                "a gated hyper-connection read with a separate norm; the read "
                "normalizes the streams itself"
            )
        hidden_states, residuals = mix(residual)
        residual, self.normed, self.gate_partials = residuals
        return hidden_states, residual

    def read_attn_input(self, residual, out_norm=None):
        return self._read(self.attn_mix, residual, out_norm)

    def update_and_read_ffn_input(self, hidden_states, residual, out_norm=None):
        residual = self.apply_attn_combine(hidden_states, residual)
        return self._read(self.ffn_mix, residual, out_norm)

    def apply_attn_combine(self, hidden_states, residual):
        return self.attn_combine(
            hidden_states, (residual, self.normed, self.gate_partials)
        )

    def apply_ffn_combine(self, hidden_states, residual):
        return self.ffn_combine(
            hidden_states, (residual, self.normed, self.gate_partials)
        )

    def sum_update_and_read_ffn_input(
        self, group, hidden_states, residual, forward_batch, norm, update
    ):
        """The attention output's all-reduce, write-back and the FFN input's
        read, with the sum completed by the write-back; None, with nothing
        launched, when it cannot take it."""
        coordinator = _owed_sum_coordinator(
            group, forward_batch, may_quantize=update.quantized_sum
        )
        if coordinator is None or not isinstance(hidden_states, torch.Tensor):
            return None
        residual = self.attn_combine_owing_sum(
            hidden_states, (residual, self.normed, self.gate_partials), coordinator
        )
        if residual is None:
            return None
        return self._read(self.ffn_mix, residual, norm)

    def sum_and_apply_ffn_combine(
        self, hidden_states, residual, group, forward_batch, may_quantize
    ):
        """apply_ffn_combine of an output that still owes its sum over
        ``group``; None, with nothing launched, when it cannot take it."""
        coordinator = _owed_sum_coordinator(
            group, forward_batch, may_quantize=may_quantize
        )
        if coordinator is None:
            return None
        return self.ffn_combine_owing_sum(
            hidden_states, (residual, self.normed, self.gate_partials), coordinator
        )

    def clear_coefficients(self):
        self.normed = None
        self.gate_partials = None

    def slice_residual_attn_tp(self, residual):
        parallel = get_parallel()
        rank, size = parallel.attn_tp_rank, parallel.attn_tp_size
        self.normed = self.normed.tensor_split(size)[rank]
        if self.gate_partials is not None:
            self.gate_partials = self.gate_partials.tensor_split(size)[rank]
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

    @classmethod
    def facts(cls, *, attn_reads_every_row: bool = False) -> LayerResidualOps:
        """What residual_ops() declares (see facts_of) for a layer whose
        attention read needs every row when ``attn_reads_every_row``. It does
        not depend on the layer's modules, so the stages of a layer that is
        not built declare it too."""
        state = cls(
            expand=None,
            attn_mix=None,
            ffn_mix=None,
            attn_combine=None,
            ffn_combine=None,
            attn_reads_every_row=attn_reads_every_row,
        )
        return residual_facts(state.residual_ops())


class _AttnReadout:
    """The normalized, gated mix of the streams, from streams that already hold
    the previous layer's output: such a layer takes its input written back. A
    ``post_residual_addition`` is not applied."""

    bind = None

    is_plain_norm = False
    completing_fusions = ()
    gathering_reads = ()
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

    bind = None
    replay_update = None

    is_plain_add = False
    applied_at_exit = False
    outlives_layer = False
    writes_stream = False
    quantized_sum = False

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

    bind = None

    is_plain_norm = False
    gathering_reads = ()
    reads_before_dp_gather = False
    reads_after_attn_tp_gather = False

    def __init__(self, state: GatedResidualState):
        self.state = state

    @property
    def completing_fusions(self) -> Tuple[ReadoutFusion, ...]:
        if self.state.attn_combine_owing_sum is None:
            return ()
        return tuple(
            ReadoutFusion(
                group,
                partial(self.state.sum_update_and_read_ffn_input, group),
                reads=True,
                accepts_update=self._accepts,
            )
            for group in (SumGroup.ATTN_TP, SumGroup.TP)
        )

    def _accepts(self, update) -> bool:
        return isinstance(update, _AttnUpdate) and update.state is self.state

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

    bind = None
    replay_update = None

    is_plain_add = False
    applied_at_exit = True
    outlives_layer = False
    writes_stream = False
    quantized_sum = False

    def __init__(self, state: GatedResidualState):
        self.state = state

    def update(self, hidden_states, residual):
        hidden_states = self.state.apply_ffn_combine(hidden_states, residual)
        self.state.clear_coefficients()
        return hidden_states

    def update_owing_sum(self, hidden_states, residual, group, forward_batch):
        """update() of an output that still owes its sum over ``group``, the sum
        completed by the write-back; None, with nothing launched, when it
        cannot (or the state has no such write-back): the caller sums first."""
        if self.state.ffn_combine_owing_sum is None:
            return None
        hidden_states = self.state.sum_and_apply_ffn_combine(
            hidden_states, residual, group, forward_batch, self.quantized_sum
        )
        if hidden_states is not None:
            self.state.clear_coefficients()
        return hidden_states

    def slice_residual_attn_tp(self, residual):
        return self.state.slice_residual_attn_tp(residual)

    def gather_residual_attn_tp(self, residual):
        return self.state.gather_residual_attn_tp(residual)


def _owed_sum_coordinator(group, forward_batch, *, may_quantize):
    """The coordinator whose full-precision all_reduce completes an output's sum
    over ``group`` as the unfused steps would (prepare's step or sum_output),
    on a decode or target-verify batch; None for any other sum or batch."""
    from sglang.srt.layers.moe.utils import post_experts_output_is_complete
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )

    mode = forward_batch.forward_mode
    if (
        not (mode.is_decode() or mode.is_target_verify())
        or forward_batch.can_run_tbo
        or is_in_breakable_cuda_graph()
        or (may_quantize and get_exec().comm.enable_quant_communications)
    ):
        return None
    parallel = get_parallel()
    if group is SumGroup.ATTN_TP:
        return parallel.attn_tp_group if parallel.attn_tp_size > 1 else None
    if group is SumGroup.TP:
        return parallel.tp_group if parallel.tp_size > 1 else None
    if group is SumGroup.MOE_OUTPUT:
        if (
            parallel.moe_ep_size > 1
            or parallel.moe_tp_size == 1
            or post_experts_output_is_complete(is_tp_path=True)
        ):
            return None
        return parallel.moe_tp_group
    return None

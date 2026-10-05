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
"""A residual read that aggregates a bank of snapshots of itself."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.attn_residual import AttnResidual
from sglang.srt.layers.layer_boundary.contracts import ReadoutFusion
from sglang.srt.layers.layer_boundary.residual import LayerResidualOps
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.runtime_context import get_parallel


class AttnBank:
    """The snapshot bank of one layer-stack invocation.

    Every layer's reads aggregate the same bank, so the bank outlives a layer
    while the reads that consume it are per-layer. The stack's entry opens one
    and the layers reach it through this holder; a pipeline rank that did not
    open the stack inherits the rows the previous rank banked.
    """

    __slots__ = ("current",)

    def __init__(self):
        self.current = None

    def open(self, hidden_states, block_num, inherited=None) -> AttnResidual:
        self.current = AttnResidual(hidden_states, block_num, inherited)
        return self.current

    def close(self) -> None:
        self.current = None

    def require(self) -> AttnResidual:
        if self.current is None:
            raise RuntimeError("open the attention-residual bank before a stage")
        return self.current


def _bank_rows(bank: AttnResidual, hidden_states) -> Optional[slice]:
    """The bank rows that go with ``hidden_states``: all of them, or this
    rank's contiguous attention-TP shard of them, the rows a reduce-scatter
    over attention TP leaves it."""
    total, rows = bank.block_residual.shape[0], hidden_states.shape[0]
    if rows == total:
        return None
    parallel = get_parallel()
    if rows * parallel.attn_tp_size != total:
        raise RuntimeError(
            f"{rows} rows are neither the bank's {total} rows nor an "
            f"attention-TP shard of them over {parallel.attn_tp_size} ranks"
        )
    rank = parallel.attn_tp_rank
    return slice(rank * rows, (rank + 1) * rows)


@dataclass
class AttnBankState:
    """One layer's pair of reads into the bank an `AttnBank` holds.

    A read aggregates the banked snapshots with the running residual and
    normalizes the result; the update that precedes it is an ordinary add, so
    both stages declare `PLAIN_ADD` and each read folds the pending add into
    its aggregation kernel. A read on this rank's attention-TP shard of the
    rows aggregates the matching shard of the bank. A write layer also
    snapshots the residual it just formed into the bank's next row, where it
    now lives: the running residual restarts empty, and the next contribution
    alone is the new head."""

    bank: AttnBank
    attn_score_proj: torch.nn.Module
    attn_score_norm: torch.nn.Module
    ffn_score_proj: torch.nn.Module
    ffn_score_norm: torch.nn.Module
    writes_block: bool = False
    # Kernels that complete the attention output's sum with the pending add
    # ahead of the FFN read.
    ffn_input_fusions: Tuple[ReadoutFusion, ...] = ()

    def _aggregate(
        self, contribution, residual, norm, *, score_proj, score_norm, write
    ):
        bank = self.bank.require()
        return bank.forward(
            contribution,
            residual,
            score_proj,
            score_norm,
            norm,
            rows=_bank_rows(bank, contribution),
            write=write,
        )

    def read_attn_input(self, residual, norm):
        # No pending add: the residual alone is the head (the stack's input,
        # or the residual of a stream that was written).
        return self._attn_aggregate(residual, None, norm)

    def update_and_read_attn_input(self, contribution, residual, norm):
        return self._attn_aggregate(contribution, residual, norm)

    def _attn_aggregate(self, contribution, residual, norm):
        normed, residual = self._aggregate(
            contribution,
            residual,
            norm,
            score_proj=self.attn_score_proj,
            score_norm=self.attn_score_norm,
            write=self.writes_block,
        )
        return normed, None if self.writes_block else residual

    def update_and_read_ffn_input(self, contribution, residual, norm):
        return self._aggregate(
            contribution,
            residual,
            norm,
            score_proj=self.ffn_score_proj,
            score_norm=self.ffn_score_norm,
            write=False,
        )

    def residual_ops(self) -> LayerResidualOps:
        return LayerResidualOps(
            attn_readout=_AttnReadout(self),
            attn_update=PLAIN_ADD,
            ffn_readout=_FfnReadout(self),
            ffn_update=PLAIN_ADD,
        )


class _BankReadout:
    """A read that mixes the banked snapshots into the residual's norm, so it
    is not the residual's plain norm and no add+norm kernel may claim it. It
    reads this rank's own rows, which the bank holds, before any DP gather."""

    is_plain_norm = False
    reads_before_dp_gather = True

    def __init__(self, state: AttnBankState):
        self.state = state

    def init_residual(self, hidden_states):
        return hidden_states

    def _reject(self, update, quant_format):
        if quant_format:
            raise NotImplementedError(
                f"an attention-residual bank read in {quant_format=}"
            )
        if update is not None and not update.is_plain_add:
            # The read folds the pending add into its aggregation, which only
            # reproduces an ordinary add.
            raise NotImplementedError(
                f"an attention-residual bank read after {update=}"
            )


class _AttnReadout(_BankReadout):
    """The attention input: the bank aggregation and this layer's input norm.
    A write layer snapshots the residual this read forms."""

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        self._reject(None, quant_format)
        if post_residual_addition is not None:
            raise NotImplementedError(
                "an attention-residual bank read with a separate addition"
            )
        return self.state.read_attn_input(residual, norm)

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        self._reject(update, kwargs.get("quant_format", ""))
        return self.state.update_and_read_attn_input(hidden_states, residual, norm)


class _FfnReadout(_BankReadout):
    """The FFN input: the attention output's add folded into the bank
    aggregation, and this layer's post-attention norm."""

    @property
    def completing_fusions(self) -> Tuple[ReadoutFusion, ...]:
        return self.state.ffn_input_fusions

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        # The attention read of a write layer leaves the residual empty, so the
        # attention output alone is the head.
        self._reject(None, quant_format)
        return self.state.update_and_read_ffn_input(residual, None, norm)

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        self._reject(update, kwargs.get("quant_format", ""))
        return self.state.update_and_read_ffn_input(hidden_states, residual, norm)


class AttnBankOutputRead:
    """The layer stack's terminal read: the bank aggregation with the output
    side's scoring parameters, then the final norm. A callable final norm for
    `residual_batch.final_norm`."""

    def __init__(self, bank: AttnBank, score_proj, score_norm, norm):
        self.bank = bank
        self.score_proj = score_proj
        self.score_norm = score_norm
        self.norm = norm

    def __call__(self, hidden_states, residual=None):
        bank = self.bank.require()
        normed, head = bank.forward(
            hidden_states,
            residual,
            self.score_proj,
            self.score_norm,
            self.norm,
            rows=_bank_rows(bank, hidden_states),
        )
        return normed if residual is None else (normed, head)

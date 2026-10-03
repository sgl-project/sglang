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

from dataclasses import dataclass, field
from typing import Optional

import torch

from sglang.srt.layers.attn_residual import AttnResidual
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


@dataclass
class AttnBankState:
    """One layer's pair of reads into the bank an `AttnBank` holds.

    A read aggregates the banked snapshots with the running residual and
    normalizes the result; the update that precedes it is an ordinary add, so
    both stages declare `PLAIN_ADD` and each read folds the pending add into
    its aggregation kernel. A write layer also snapshots the residual it just
    formed into the bank's next row.

    Args:
        bank: The stack's bank holder, shared by every layer.
        attn_score_proj / attn_score_norm: This layer's scoring parameters for
            the attention stage's read.
        ffn_score_proj / ffn_score_norm: The same for the FFN stage's read.
        writes_block: Whether this layer snapshots its residual into the bank.
    """

    bank: AttnBank
    attn_score_proj: torch.nn.Module
    attn_score_norm: torch.nn.Module
    ffn_score_proj: torch.nn.Module
    ffn_score_norm: torch.nn.Module
    writes_block: bool = False
    # The rows this rank owns once the boundary has sliced the residual over
    # attention TP; None while the stage runs on every row.
    rows: Optional[slice] = field(default=None, repr=False)

    def _aggregate(
        self, contribution, residual, norm, *, score_proj, score_norm, write
    ):
        return self.bank.require().forward(
            contribution,
            residual,
            score_proj,
            score_norm,
            norm,
            rows=self.rows,
            write=write,
        )

    def read_attn_input(self, residual, norm):
        # The stack's first read has nothing banked and no pending add; its
        # residual is the value that entered the stack.
        return self._aggregate(
            residual,
            None,
            norm,
            score_proj=self.attn_score_proj,
            score_norm=self.attn_score_norm,
            write=self.writes_block,
        )

    def update_and_read_attn_input(self, contribution, residual, norm):
        return self._aggregate(
            contribution,
            residual,
            norm,
            score_proj=self.attn_score_proj,
            score_norm=self.attn_score_norm,
            write=self.writes_block,
        )

    def update_and_read_ffn_input(self, contribution, residual, norm):
        return self._aggregate(
            contribution,
            residual,
            norm,
            score_proj=self.ffn_score_proj,
            score_norm=self.ffn_score_norm,
            write=False,
        )

    def slice_rows_attn_tp(self, residual):
        """Take this rank's shard and remember which bank rows go with it.

        The shards are equal and contiguous, which is what the bank rows and
        the all-gather that reassembles them both assume. A row count the
        group does not divide has no such shard, so the boundary must complete
        the sum over every row instead of sharding.
        """
        parallel = get_parallel()
        rank, size = parallel.attn_tp_rank, parallel.attn_tp_size
        rows, remainder = divmod(residual.shape[0], size)
        if remainder:
            raise NotImplementedError(
                "an attention-residual bank sharded over attention TP needs a "
                f"row count the group divides, got {residual.shape[0]} over {size}"
            )
        self.rows = slice(rank * rows, (rank + 1) * rows)
        return residual[self.rows]

    def gather_rows_attn_tp(self, residual):
        raise NotImplementedError(
            "Unsupported: gathering an attention-residual bank over attention TP"
        )

    def residual_ops(self) -> LayerResidualOps:
        add = _BankAdd(self)
        return LayerResidualOps(
            attn_readout=_AttnReadout(self),
            attn_update=add,
            ffn_readout=_FfnReadout(self),
            ffn_update=add,
        )


class _BankAdd:
    """An ordinary residual add that also moves the bank's rows.

    The add itself is `PlainAdd`'s, so every order and fusion an ordinary add
    allows stays available; what this adds is that a residual sliced over
    attention TP takes the bank rows with it, which the shared `PLAIN_ADD`
    singleton could not record for one layer.
    """

    is_plain_add = True
    applied_at_exit = False
    outlives_layer = True

    def __init__(self, state: "AttnBankState"):
        self.state = state

    def update(self, hidden_states, residual):
        return PLAIN_ADD.update(hidden_states, residual)

    def slice_residual_attn_tp(self, residual):
        return self.state.slice_rows_attn_tp(residual)

    def gather_residual_attn_tp(self, residual):
        return self.state.gather_rows_attn_tp(residual)


class _BankReadout:
    """A read that mixes the banked snapshots into the residual's norm, so it
    is not the residual's plain norm and no add+norm kernel may claim it."""

    is_plain_norm = False
    reads_before_dp_gather = False

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

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        raise NotImplementedError(
            "an attention-residual bank FFN input read without its attention"
        )

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        self._reject(update, kwargs.get("quant_format", ""))
        return self.state.update_and_read_ffn_input(hidden_states, residual, norm)

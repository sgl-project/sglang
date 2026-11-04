"""The FFN read of Kimi-K3's gfx950 row-sharded latent-MoE tail
(``AttnBankState.ffn_row_shards``): the attention output's sum completed onto
this rank's 1/TP of the rows and read there, leaving the written residual on the
shard."""

from functools import partial
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.layers.layer_boundary.contracts import ReadoutFusion
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.layer_boundary.residual.attn_bank import AttnBankState


def row_shard(num_tokens: int, world: int, rank: int) -> Tuple[int, int]:
    """(first row, number of real rows) of rank's 1/world of num_tokens padded to a
    multiple of world; 0 real rows when the shard lies past the last token."""
    rows = -(-num_tokens // world)
    start = rank * rows
    return start, max(0, min(rows, num_tokens - start))


def reduce_scatter_rows_and_read_ffn_input(
    state: "AttnBankState", hidden_states, residual, forward_batch, norm
):
    """The attention output's sum onto this rank's 1/TP of the rows (padded
    to a multiple of TP), the FFN read there, and the all-gather of the
    normed input. Returns (normed input on every row, written residual on
    this rank's padded shard, zero past the last token)."""
    num_tokens, hidden_size = hidden_states.shape
    if not state.ffn_row_shards(num_tokens):
        return None
    group = get_parallel().attn_tp_group
    world, rank = group.world_size, group.rank_in_group
    padded = -(-num_tokens // world) * world
    rows = padded // world
    if padded != num_tokens:
        hidden_states = torch.nn.functional.pad(
            hidden_states, (0, 0, 0, padded - num_tokens)
        )
    reduced = hidden_states.new_empty(rows, hidden_size)
    group.reduce_scatter_tensor(reduced, hidden_states.contiguous())
    start, valid = row_shard(num_tokens, world, rank)
    if valid == 0:
        # this rank's padded rows all lie past the last token
        normed = reduced.new_zeros(rows, hidden_size)
        written = reduced.new_zeros(rows, hidden_size)
    else:
        normed, written = state.bank.require().forward(
            reduced[:valid],
            None if residual is None else residual[start : start + valid],
            state.ffn_score_proj,
            state.ffn_score_norm,
            norm,
            rows=slice(start, start + valid),
        )
        if valid < rows:
            pad = (0, 0, 0, rows - valid)
            normed = torch.nn.functional.pad(normed, pad)
            written = torch.nn.functional.pad(written, pad)
    gathered = normed.new_empty(padded, hidden_size)
    group.all_gather_into_tensor(gathered, normed.contiguous())
    return gathered[:num_tokens], written


def row_sharded_ffn_read(state: "AttnBankState") -> ReadoutFusion:
    """The FFN readout's kernel that takes the batches ``state.ffn_row_shards``
    accepts."""
    return ReadoutFusion(
        SumGroup.ATTN_TP,
        partial(reduce_scatter_rows_and_read_ffn_input, state),
        reads=True,
    )

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


def gather_row_shards(rows: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """All num_tokens rows from every rank's row_shard of them (each rank's
    shard padded to the equal all-gather width)."""
    group = get_parallel().attn_tp_group
    width = -(-num_tokens // group.world_size)
    if rows.shape[0] < width:
        rows = torch.nn.functional.pad(rows, (0, 0, 0, width - rows.shape[0]))
    gathered = rows.new_empty(width * group.world_size, rows.shape[1])
    group.all_gather_into_tensor(gathered, rows.contiguous())
    return gathered[:num_tokens]


def _padded_rows(hidden_states: torch.Tensor, rows: int) -> torch.Tensor:
    """hidden_states padded to ``rows`` rows for an equal-shard reduce-scatter:
    in place when its storage already extends that far (an o_proj that wrote
    into padded rows), else a padded copy. The pad rows only reach rows past
    the last token, which no read keeps."""
    hidden_size = hidden_states.shape[1]
    if hidden_states.is_contiguous() and (
        hidden_states.untyped_storage().nbytes() // hidden_states.element_size()
        >= hidden_states.storage_offset() + rows * hidden_size
    ):
        return hidden_states.as_strided(
            (rows, hidden_size), hidden_states.stride(), hidden_states.storage_offset()
        )
    return torch.nn.functional.pad(
        hidden_states, (0, 0, 0, rows - hidden_states.shape[0])
    )


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
    reduced = None if state.ffn_attn_rows is None else state.ffn_attn_rows(num_tokens)
    if reduced is None:
        if padded != num_tokens:
            hidden_states = _padded_rows(hidden_states, padded)
        reduced = hidden_states.new_empty(rows, hidden_size)
        group.reduce_scatter_tensor(reduced, hidden_states.contiguous())
    start, valid = row_shard(num_tokens, world, rank)
    if valid == 0:
        # this rank's padded rows all lie past the last token
        normed = reduced.new_zeros(rows, hidden_size)
        written = reduced.new_zeros(rows, hidden_size)
    else:
        if residual is not None and residual.shape[0] == num_tokens:
            residual = residual[start : start + valid]
        # else a row-sharded stream already left the residual on this rank's rows
        normed, written = state.bank.require().forward(
            reduced[:valid],
            residual,
            state.ffn_score_proj,
            state.ffn_score_norm,
            norm,
            rows=slice(start, start + valid),
        )
        if valid < rows:
            pad = (0, 0, 0, rows - valid)
            normed = torch.nn.functional.pad(normed, pad)
            written = torch.nn.functional.pad(written, pad)
    if state.ffn_shard_front is not None and state.ffn_shard_front(normed, num_tokens):
        # the FFN keeps what it gathered; its input rows are never read
        return normed.new_empty(1, 1).expand(num_tokens, hidden_size), written
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

"""The FFN read of Kimi-K3's gfx950 row-sharded latent-MoE tail
(``AttnBankState.ffn_row_shard``): the attention output's sum completed onto
this rank's 1/TP of the rows and read there, leaving the written residual on the
shard. With SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM the stream stays on those
(possibly short) row shards, and the next read aggregates the bank there and
gathers what it read."""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.contracts import ReadoutFusion
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.attn_residual import AttnResidual
    from sglang.srt.layers.layer_boundary.residual.attn_bank import AttnBankState


@dataclass(frozen=True)
class RowShardRead:
    """A layer's row-sharded FFN read (``AttnBankState.ffn_row_shard``)."""

    # Whether a batch of this many tokens completes the attention output's sum
    # onto this rank's 1/TP of the rows, reads the FFN input there and gathers
    # it, leaving the written residual on the shard for an FFN that consumes
    # it there (K3's row-sharded MoE tail on gfx950).
    takes: Callable[[int], bool]
    # The FFN's own work on this rank's normed rows and the gather of what it
    # needs, in place of gathering the normed input. Takes the padded rows and
    # the token count; False leaves the gather to the read.
    shard_front: Optional[Callable[[torch.Tensor, int], bool]] = None
    # The attention output's complete sum on this rank's padded rows when the
    # attention produced it there, else None (the read reduce-scatters its
    # TP-partial output).
    attn_rows: Optional[Callable[[int], Optional[torch.Tensor]]] = None


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


def stream_rows(rows: int, total: int, parallel) -> Optional[slice]:
    """_bank_rows of a row-sharded stream: the bank rows of this rank's row_shard
    when ``rows`` is its (possibly short or empty) row count, else None."""
    if not envs.SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM.get():
        return None
    start, valid = row_shard(total, parallel.attn_tp_size, parallel.attn_tp_rank)
    return slice(start, start + valid) if rows == valid else None


def gather_stream_read(bank: "AttnResidual", normed: torch.Tensor) -> torch.Tensor:
    """An attention read of a stream the MoE tail left on row shards, read there:
    its normed input gathered onto every row (the residual stays on the shard).
    Any other read as it is."""
    total = bank.block_residual.shape[0]
    if (
        not envs.SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM.get()
        or normed.shape[0] == total
    ):
        return normed
    return gather_row_shards(normed, total)


def gather_stream_output(
    bank: "AttnResidual", read: tuple, rows: Optional[slice]
) -> Tuple[tuple, Optional[slice]]:
    """The output read of a stream the MoE tail left on (possibly short) row
    shards: (the read with its normed rows gathered, None). Any other read
    returns (read, rows) for AttnBankOutputRead's own gather."""
    if rows is None or not envs.SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM.get():
        return read, rows
    total = bank.block_residual.shape[0]
    return (gather_row_shards(read[0], total), read[1]), None


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
    read = state.ffn_row_shard
    if not read.takes(num_tokens):
        return None
    group = get_parallel().attn_tp_group
    world, rank = group.world_size, group.rank_in_group
    padded = -(-num_tokens // world) * world
    rows = padded // world
    reduced = None if read.attn_rows is None else read.attn_rows(num_tokens)
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
    if read.shard_front is not None and read.shard_front(normed, num_tokens):
        # the FFN keeps what it gathered; its input rows are never read
        return normed.new_empty(1, 1).expand(num_tokens, hidden_size), written
    gathered = normed.new_empty(padded, hidden_size)
    group.all_gather_into_tensor(gathered, normed.contiguous())
    return gathered[:num_tokens], written


def row_sharded_ffn_read(state: "AttnBankState") -> ReadoutFusion:
    """The FFN readout's kernel that takes the batches ``state.ffn_row_shard``
    accepts."""
    return ReadoutFusion(
        SumGroup.ATTN_TP,
        partial(reduce_scatter_rows_and_read_ffn_input, state),
        reads=True,
    )

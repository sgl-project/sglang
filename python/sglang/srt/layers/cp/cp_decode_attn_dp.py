"""Query-row DP for target decode/verify inside a prefill-CP replica group."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_parallel


def use_cp_decode_attn_dp(forward_batch) -> bool:
    parallel = get_parallel()
    mode = forward_batch.forward_mode
    return getattr(parallel, "cp_decode_attn_mode", None) == "dp" and (
        mode.is_decode() or mode.is_target_verify()
    )


@dataclass(frozen=True)
class CPDecodeQueryLayout:
    """A graph-stable contiguous query shard; padded ranks still gather output."""

    rows: int
    rank: int
    size: int

    @property
    def local_rows(self) -> int:
        return (self.rows + self.size - 1) // self.size

    @property
    def start(self) -> int:
        return min(self.rank * self.local_rows, self.rows)

    @property
    def stop(self) -> int:
        return min((self.rank + 1) * self.local_rows, self.rows)

    @property
    def query_rows(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def actual_rows(self) -> int:
        return self.stop - self.start

    def select(self, tensor: torch.Tensor, *, value=0, pad: bool = True):
        assert tensor.shape[0] >= self.rows
        local = tensor[self.query_rows]
        if not pad or local.shape[0] == self.local_rows:
            return local
        padding = [0, 0] * (tensor.ndim - 1) + [0, self.local_rows - local.shape[0]]
        return F.pad(local, padding, value=value)

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather

        assert local.shape[0] == self.local_rows
        # The existing helper gathers in rank order without interleave reordering.
        return attn_cp_interleave_gather(local)[: self.rows]


def cp_decode_query_layout(forward_batch, rows: int):
    if not use_cp_decode_attn_dp(forward_batch):
        return None
    parallel = get_parallel()
    return CPDecodeQueryLayout(rows, parallel.attn_cp_rank, parallel.attn_cp_size)

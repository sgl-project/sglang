"""The logit_bias stage: each request's entries are written once at admission into
a per-slot store, gathered by slot at each launch, and added to the logits at
their token ids."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

import numpy as np
import torch

from sglang.srt.utils.device import async_h2d_tensor

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class LogitBiasEntries(NamedTuple):
    """A request's logit_bias, parsed once. Raw bytes, so an admission joins every
    request's entries in one call."""

    token_ids: bytes  # int32
    values: bytes  # fp32

    @classmethod
    def parse(cls, logit_bias: Dict[str, float]) -> LogitBiasEntries:
        # Dedup on int(key) first ("1" and "01" collide): duplicate indices make
        # the index_put in apply_logit_bias nondeterministic on CUDA.
        row_bias = {int(key): value for key, value in logit_bias.items()}
        return cls(
            token_ids=array("i", row_bias.keys()).tobytes(),
            values=array("f", row_bias.values()).tobytes(),
        )


class LogitBiasRows(NamedTuple):
    """One launch's logit_bias entries, gathered by slot: entry i adds values[i]
    to batch row rows[i] at token_ids[i]."""

    rows: torch.Tensor  # [n] int64
    token_ids: torch.Tensor  # [n] int64
    values: torch.Tensor  # [n] fp32


class LogitBiasStore:
    """Each request slot's logit_bias entries on the device. A request's entries
    are one contiguous run of a flat buffer; its slot's offset and count live in
    `SamplingParamsTable`, so a launch reads them with the other fields.

    Runs are appended at admission. A rewritten slot drops its old run, and a
    request whose slot is not yet reused keeps its run, so when the new runs do not
    fit, `write` compacts the live runs into a new buffer at least twice their size
    and returns the slots it moved, whose offsets the table must take."""

    def __init__(self, num_rows: int, device):
        # Row 0 token ids, row 1 fp32 value bits.
        self.entries = torch.empty((2, 0), dtype=torch.int32, device=device)
        self.offsets_host = np.zeros(num_rows, dtype=np.int64)
        self.counts_host = np.zeros(num_rows, dtype=np.int64)
        # A request with an empty logit_bias counts: the stage runs for it.
        self.has_logit_bias_host = np.zeros(num_rows, dtype=bool)
        self.end = 0

    def write(
        self, slots: List[int], entries: List[Optional[LogitBiasEntries]]
    ) -> np.ndarray:
        """Store each slot's entries (None for none) and set its offset and count.
        Returns the other slots whose runs a compaction moved (empty without one)."""
        self.counts_host[slots] = 0
        self.has_logit_bias_host[slots] = [e is not None for e in entries]
        counts = np.array(
            [0 if e is None else len(e.values) // 4 for e in entries], dtype=np.int64
        )
        num_new = int(counts.sum())
        moved = np.empty(0, dtype=np.int64)
        if self.end + num_new > self.entries.shape[1]:
            moved = self._compact(num_new)
        self.offsets_host[slots] = self.end + np.cumsum(counts) - counts
        self.counts_host[slots] = counts
        if num_new:
            biased = [e for e in entries if e is not None]
            # One copy of 32-bit words: every token id, then every value.
            block = bytearray().join(
                [e.token_ids for e in biased] + [e.values for e in biased]
            )
            staged = async_h2d_tensor(
                torch.frombuffer(block, dtype=torch.int32), self.entries.device
            )
            self.entries[:, self.end : self.end + num_new] = staged.view(2, num_new)
            self.end += num_new
        return moved

    def _compact(self, num_new: int) -> np.ndarray:
        """Move the live runs to the front of a buffer with room for `num_new` more."""
        live = np.flatnonzero(self.counts_host)
        counts = self.counts_host[live]
        num_live = int(counts.sum())
        # Half the buffer stays free, so steady admissions compact rarely.
        capacity = max(self.entries.shape[1], 2 * (num_live + num_new))
        entries = torch.empty(
            (2, capacity), dtype=torch.int32, device=self.entries.device
        )
        starts = self.offsets_host[live]
        new_starts = np.cumsum(counts) - counts
        if num_live:
            positions = np.arange(num_live) + np.repeat(starts - new_starts, counts)
            positions = async_h2d_tensor(torch.from_numpy(positions), entries.device)
            entries[:, :num_live] = self.entries[:, positions]
            self.offsets_host[live] = new_starts
        self.entries = entries
        self.end = num_live
        return live[starts != new_starts]

    def gather(
        self, slots: np.ndarray, offsets: torch.Tensor, counts: torch.Tensor
    ) -> Optional[LogitBiasRows]:
        """The entries of the batch rows in `slots`, from their `offsets` and
        `counts` gathered from the table ([bs] int32), with no host-to-device copy;
        None when no row has a logit_bias."""
        if not self.has_logit_bias_host[slots].any():
            return None
        num_entries = int(self.counts_host[slots].sum())
        device = offsets.device
        counts = counts.long()
        rows = torch.repeat_interleave(counts, output_size=num_entries)
        # Entry j of row r is the r-th run's j-th: its offset minus the entries
        # before row r, plus its index among all entries.
        starts = offsets.long() - (counts.cumsum(0) - counts)
        positions = torch.arange(num_entries, device=device) + starts[rows]
        entries = self.entries[:, positions]
        return LogitBiasRows(
            rows=rows,
            token_ids=entries[0].long(),
            values=entries[1].view(torch.float32),
        )


def apply_logit_bias(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    rows_per_request: int = 1,
) -> None:
    """Add each request's bias to its `rows_per_request` consecutive logits rows."""
    bias = sampling_info.logit_bias
    if bias is None:
        return
    logits = logits.unflatten(0, (-1, rows_per_request))
    index = (bias.rows, slice(None), bias.token_ids)
    # In the bias's own dtype, so the sum rounds once to the logits dtype.
    logits[index] = (logits[index] + bias.values[:, None]).to(logits.dtype)

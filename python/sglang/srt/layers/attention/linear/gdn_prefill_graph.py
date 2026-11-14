"""GDN extend captured inside breakable prefill CUDA graphs.

Without this module a breakable prefill CUDA graph (BCG) breaks at every GDN
layer and runs the extend chain eagerly: about ten Triton launches plus small
torch ops, roughly a millisecond of host time per layer. The eager chain derives
its launch grids and index tables from host-side sequence lengths
(``prepare_chunk_indices``, ``causal_conv1d_fn``'s ``seq_lens_cpu``, the
prefix-cache track plan); here they come from static per-bucket tables instead,
as in kda_prefill_graph.py:

* ``max_seqs`` sequences (``min(bucket, cap)``; batches with more sequences run
  eagerly, through the runner's replay vote on
  ``AttentionBackend.prefill_cuda_graph_max_batch_size``),
* ``max_chunks = cdiv(bucket, 64) + max_seqs`` delta-rule chunks,
* ``max_conv_blocks = cdiv(bucket, BLOCK_M) + max_seqs`` conv blocks.

Before every replay ``fill`` rebuilds the tables from the batch's CPU lengths
(one host-to-device copy plus a few device copies, no synchronization). The
captured kernels are the eager ones over fixed grids; padded rows point past
their sequence or carry ``PAD_SLOT_ID`` and return, so every real chunk computes
exactly what the eager break computes.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import numpy as np
import torch
import triton

from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    CAUSAL_CONV1D_FWD_BLOCK_M,
    PAD_SLOT_ID,
)
from sglang.kernels.ops.mamba.gdn_prefill_track import (
    TRACK_CHUNK_STATE,
    TRACK_FINAL_STATE,
)
from sglang.srt.layers.attention.linear.kda_prefill_graph import KDAPrefillTrackRows

# Delta-rule chunk size of chunk_gated_delta_rule (fla/chunk.py).
GDN_CHUNK_SIZE = 64


def chunk_o_block_size(num_tokens: int) -> int:
    """chunk_fwd_o's chunk size for a varlen batch of ``num_tokens`` rows."""
    return min(GDN_CHUNK_SIZE, max(16, triton.next_power_of_2(num_tokens)))


def buckets_keep_chunk_o_block_size(capture_num_tokens: list[int]) -> bool:
    """Whether every batch replays with the chunk_fwd_o chunk size its eager
    break uses: the break runs on the real token count, the graph on the bucket."""
    buckets = sorted(capture_num_tokens)
    for n in range(1, GDN_CHUNK_SIZE + 1):
        bucket = next((b for b in buckets if b >= n), None)
        if bucket is not None and chunk_o_block_size(bucket) != chunk_o_block_size(n):
            return False
    return True


class GDNPrefillGraphMetadata(msgspec.Struct, frozen=True):
    """Static per-bucket tables read by the captured GDN extend."""

    num_tokens: int
    max_seqs: int
    max_chunks: int
    max_conv_blocks: int
    # Allocates and records the prefix-cache snapshot copies.
    track: bool

    # One int32 device buffer, refreshed by a single copy, and its views.
    packed: torch.Tensor
    cu_seqlens: torch.Tensor  # [max_seqs + 1]
    chunk_indices: torch.Tensor  # [max_chunks, 2] (seq, chunk), sequence-major
    chunk_offsets: torch.Tensor  # [max_seqs + 1]
    conv_block_table: torch.Tensor  # [max_conv_blocks, 2] (seq, block)
    track_mode: torch.Tensor  # [max_seqs]
    track_h_row: torch.Tensor  # [max_seqs] row of the per-chunk states h
    track_conv_start: torch.Tensor  # [max_seqs]
    num_logical_tokens: torch.Tensor  # [1]
    host: np.ndarray  # int32 staging for `packed`

    # Filled on the device from the batch.
    cache_indices: torch.Tensor  # [max_seqs], PAD_SLOT_ID past the batch
    has_initial_state: torch.Tensor  # [max_seqs] bool
    track_dst: torch.Tensor  # [max_seqs] int64

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        max_seqs_cap: int,
        track: bool,
        cache_index_dtype: torch.dtype,
        device: torch.device,
    ) -> GDNPrefillGraphMetadata:
        max_seqs = max(1, min(num_tokens, max_seqs_cap))
        max_chunks = triton.cdiv(num_tokens, GDN_CHUNK_SIZE) + max_seqs
        max_conv_blocks = triton.cdiv(num_tokens, CAUSAL_CONV1D_FWD_BLOCK_M) + max_seqs
        sizes = [
            max_seqs + 1,  # cu_seqlens
            2 * max_chunks,  # chunk_indices
            max_seqs + 1,  # chunk_offsets
            2 * max_conv_blocks,  # conv_block_table
            max_seqs,  # track_mode
            max_seqs,  # track_h_row
            max_seqs,  # track_conv_start
            1,  # num_logical_tokens
        ]
        packed = torch.zeros(sum(sizes), dtype=torch.int32, device=device)
        views = torch.split(packed, sizes)
        return cls(
            num_tokens=num_tokens,
            max_seqs=max_seqs,
            max_chunks=max_chunks,
            max_conv_blocks=max_conv_blocks,
            track=track,
            packed=packed,
            cu_seqlens=views[0],
            chunk_indices=views[1].view(max_chunks, 2),
            chunk_offsets=views[2],
            conv_block_table=views[3].view(max_conv_blocks, 2),
            track_mode=views[4],
            track_h_row=views[5],
            track_conv_start=views[6],
            num_logical_tokens=views[7],
            host=np.zeros(sum(sizes), dtype=np.int32),
            cache_indices=torch.full(
                (max_seqs,), PAD_SLOT_ID, dtype=cache_index_dtype, device=device
            ),
            has_initial_state=torch.zeros(max_seqs, dtype=torch.bool, device=device),
            track_dst=torch.zeros(max_seqs, dtype=torch.int64, device=device),
        )

    def can_fit(self, extend_seq_lens: list[int]) -> bool:
        return (
            len(extend_seq_lens) <= self.max_seqs
            and sum(extend_seq_lens) <= self.num_tokens
        )

    def fill(
        self,
        *,
        extend_seq_lens: list[int],
        cache_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        track_rows: Optional[KDAPrefillTrackRows],
    ) -> None:
        """Rebuild the tables for this batch. ``cache_indices`` and
        ``extend_prefix_lens`` are the batch's device tensors."""
        bs = len(extend_seq_lens)
        assert self.can_fit(extend_seq_lens), (
            f"batch of {bs} sequences / {sum(extend_seq_lens)} tokens exceeds the "
            f"captured bounds ({self.max_seqs} / {self.num_tokens})"
        )
        self._write_host_tables(
            lens=np.asarray(extend_seq_lens, dtype=np.int64), track_rows=track_rows
        )
        # A pinned block from the caching host allocator keeps the copy
        # asynchronous and is not reused before the copy completes.
        self.packed.copy_(torch.from_numpy(self.host).pin_memory(), non_blocking=True)
        self.cache_indices[:bs].copy_(cache_indices[:bs])
        self.cache_indices[bs:].fill_(PAD_SLOT_ID)
        torch.gt(extend_prefix_lens[:bs], 0, out=self.has_initial_state[:bs])
        self.has_initial_state[bs:].fill_(False)
        if track_rows is not None:
            self.track_dst[:bs].copy_(track_rows.dst[:bs])

    def _write_host_tables(
        self, *, lens: np.ndarray, track_rows: Optional[KDAPrefillTrackRows]
    ) -> None:
        """Write every host-built table into ``self.host`` in ``packed`` order."""
        bs = lens.shape[0]
        total = int(lens.sum())

        # Padded sequences are empty and sit at the end.
        cu = np.empty(self.max_seqs + 1, dtype=np.int64)
        cu[0] = 0
        np.cumsum(lens, out=cu[1 : bs + 1])
        cu[bs + 1 :] = total
        o = self._write(o=0, values=cu)

        # Sequence-major (seq, chunk) rows: chunk_fwd_o reads the per-chunk states
        # by row, chunk_gated_delta_rule_fwd_h writes them at chunk_offsets[seq] + chunk.
        # Padded rows name the chunk just past sequence 0, which every chunk
        # kernel masks or returns on.
        n_chunks = (lens + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE
        o = self._write_pairs(
            o=o, counts=n_chunks, rows=self.max_chunks, pad=(0, int(n_chunks[0]))
        )
        offsets = np.empty(self.max_seqs + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(n_chunks, out=offsets[1 : bs + 1])
        offsets[bs + 1 :] = offsets[bs]
        o = self._write(o=o, values=offsets)

        n_blocks = (lens + CAUSAL_CONV1D_FWD_BLOCK_M - 1) // CAUSAL_CONV1D_FWD_BLOCK_M
        o = self._write_pairs(
            o=o, counts=n_blocks, rows=self.max_conv_blocks, pad=(PAD_SLOT_ID, 0)
        )

        track_mode = np.zeros(self.max_seqs, dtype=np.int64)
        track_h_row = np.zeros(self.max_seqs, dtype=np.int64)
        track_conv_start = np.zeros(self.max_seqs, dtype=np.int64)
        if track_rows is not None:
            # Same source rows as the eager plan: h_offsets[row] + chunk.
            for row, chunk in track_rows.chunk_rows:
                track_h_row[row] = offsets[row] + chunk
                track_mode[row] = TRACK_CHUNK_STATE
            for row in track_rows.final_rows:
                track_mode[row] = TRACK_FINAL_STATE
            for row, aligned_len in track_rows.conv_rows:
                track_conv_start[row] = cu[row] + aligned_len - track_rows.conv_len
        o = self._write(o=o, values=track_mode)
        o = self._write(o=o, values=track_h_row)
        o = self._write(o=o, values=track_conv_start)
        o = self._write(o=o, values=np.array([total]))
        assert o == self.host.shape[0]

    def _write(self, *, o: int, values: np.ndarray) -> int:
        self.host[o : o + values.shape[0]] = values
        return o + values.shape[0]

    def _write_pairs(
        self, *, o: int, counts: np.ndarray, rows: int, pad: tuple[int, int]
    ) -> int:
        """Write (seq, index) rows, `counts[seq]` indices per sequence, into a
        [rows, 2] table at `o`, padding the tail with `pad`."""
        n = int(counts.sum())
        table = self.host[o : o + 2 * rows].reshape(rows, 2)
        if n:
            table[:n, 0] = np.repeat(np.arange(counts.shape[0]), counts)
            table[:n, 1] = np.arange(n) - np.repeat(np.cumsum(counts) - counts, counts)
        table[n:, 0] = pad[0]
        table[n:, 1] = pad[1]
        return o + 2 * rows

"""KDA extend captured inside breakable prefill CUDA graphs.

A breakable prefill CUDA graph (BCG) is captured once per token bucket and
replayed for any batch that pads into that bucket. The eager KDA extend derives
its launch grids and index tables from host-side sequence lengths
(``prepare_chunk_indices``, ``causal_conv1d_fn``'s ``seq_lens_cpu``, the
prefix-cache track plan), so without this module BCG breaks the graph at every
KDA layer and runs the whole chain -- about sixteen kernels per layer plus their
Python -- eagerly.

For each bucket this module allocates static tables sized by the bucket's
bounds:

* ``max_seqs`` sequences (``min(bucket, cap)``; batches with more sequences run
  eagerly, see ``KDAAttnBackend.can_run_prefill_graph_extend``),
* ``max_chunks = cdiv(bucket, 64) + max_seqs`` delta-rule chunks,
* ``max_conv_blocks = cdiv(bucket, 8) + max_seqs`` conv blocks.

Before every replay ``fill`` rebuilds the tables from the batch's CPU lengths
(one host-to-device copy plus a few device gathers, no synchronization). The
captured kernels launch fixed grids over the bounds; padded rows point past
their sequence (chunk tables) or carry ``PAD_SLOT_ID`` (conv blocks, state
slots) and return immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch
import triton

from sglang.kernels.ops.attention.fla.kda_prefill_graph_track import (
    TRACK_CHUNK_STATE,
    TRACK_FINAL_STATE,
    kda_track_conv_window,
    kda_track_ssm_state,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    CAUSAL_CONV1D_FWD_BLOCK_M,
    PAD_SLOT_ID,
    causal_conv1d_fn,
)
from sglang.srt.layers.attention.mamba.prefill_track_metadata import (
    build_prefill_track_plan,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

# Delta-rule chunk size of chunk_kda (fla/kda.py, fla/chunk_delta_h.py).
KDA_CHUNK_SIZE = 64
# Chunk index of padded chunk_indices rows: past the end of any sequence, so
# every varlen chunk kernel returns on its `i_t * BT >= T` guard.
_PAD_CHUNK = 1 << 20


class KDAPrefillTrackRows(msgspec.Struct, frozen=True):
    """One batch's prefix-cache track plan, in batch rows."""

    # (row, chunk): the snapshot is the state at that chunk boundary.
    chunk_rows: list[tuple[int, int]]
    # Rows whose snapshot is the post-extend state (chunk-aligned length).
    final_rows: list[int]
    # (row, aligned_len) of every tracked row: its conv window ends there.
    conv_rows: list[tuple[int, int]]
    # Conv window length (kernel_size - 1).
    conv_len: int
    # [bs] physical track slots, on the device.
    dst: torch.Tensor


class KDAPrefillGraphMetadata(msgspec.Struct, frozen=True):
    """Static per-bucket tables read by the captured KDA extend."""

    num_tokens: int
    max_seqs: int
    max_chunks: int
    max_conv_blocks: int
    # Allocates and records the prefix-cache snapshot copies.
    track: bool

    # One int32 device buffer, refreshed by a single copy, and its views.
    packed: torch.Tensor
    cu_seqlens: torch.Tensor  # [max_seqs + 1]
    chunk_indices: torch.Tensor  # [max_chunks, 2] (seq, chunk)
    chunk_offsets: torch.Tensor  # [max_seqs + 1]
    conv_block_table: torch.Tensor  # [max_conv_blocks, 2] (seq, block)
    track_chunk_idx: torch.Tensor  # [max_seqs]
    track_mode: torch.Tensor  # [max_seqs]
    track_conv_start: torch.Tensor  # [max_seqs]
    num_logical_tokens: torch.Tensor  # [1]
    host: np.ndarray  # int32 staging for `packed`

    # Filled on the device from the batch.
    cache_indices: torch.Tensor  # [max_seqs], PAD_SLOT_ID past the batch
    has_initial_state: torch.Tensor  # [max_seqs] bool
    track_dst: torch.Tensor  # [max_seqs] int64
    token_index: torch.Tensor  # [num_tokens] int32 arange

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        max_seqs_cap: int,
        track: bool,
        cache_index_dtype: torch.dtype,
        device: torch.device,
    ) -> KDAPrefillGraphMetadata:
        max_seqs = max(1, min(num_tokens, max_seqs_cap))
        max_chunks = triton.cdiv(num_tokens, KDA_CHUNK_SIZE) + max_seqs
        max_conv_blocks = triton.cdiv(num_tokens, CAUSAL_CONV1D_FWD_BLOCK_M) + max_seqs
        sizes = [
            max_seqs + 1,  # cu_seqlens
            2 * max_chunks,  # chunk_indices
            max_seqs + 1,  # chunk_offsets
            2 * max_conv_blocks,  # conv_block_table
            max_seqs,  # track_chunk_idx
            max_seqs,  # track_mode
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
            track_chunk_idx=views[4],
            track_mode=views[5],
            track_conv_start=views[6],
            num_logical_tokens=views[7],
            host=np.zeros(sum(sizes), dtype=np.int32),
            cache_indices=torch.full(
                (max_seqs,), PAD_SLOT_ID, dtype=cache_index_dtype, device=device
            ),
            has_initial_state=torch.zeros(max_seqs, dtype=torch.bool, device=device),
            track_dst=torch.zeros(max_seqs, dtype=torch.int64, device=device),
            token_index=torch.arange(num_tokens, dtype=torch.int32, device=device),
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
        host = self.host

        # Padded sequences are empty and sit at the end.
        cu = np.empty(self.max_seqs + 1, dtype=np.int64)
        cu[0] = 0
        np.cumsum(lens, out=cu[1 : bs + 1])
        cu[bs + 1 :] = total
        o = self._write(o=0, values=cu)

        # Sequence-major (seq, chunk) pairs: chunk_gla_fwd_o reads the per-chunk
        # states by row, chunk_gated_delta_rule_fwd_h writes them at
        # chunk_offsets[seq] + chunk.
        n_chunks = (lens + KDA_CHUNK_SIZE - 1) // KDA_CHUNK_SIZE
        o = self._write_pairs(
            o=o, counts=n_chunks, rows=self.max_chunks, pad=(0, _PAD_CHUNK)
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

        track_chunk_idx = np.full(self.max_seqs, -1, dtype=np.int64)
        track_mode = np.zeros(self.max_seqs, dtype=np.int64)
        track_conv_start = np.zeros(self.max_seqs, dtype=np.int64)
        if track_rows is not None:
            for row, chunk in track_rows.chunk_rows:
                track_chunk_idx[row] = chunk
                track_mode[row] = TRACK_CHUNK_STATE
            for row in track_rows.final_rows:
                track_mode[row] = TRACK_FINAL_STATE
            for row, aligned_len in track_rows.conv_rows:
                track_conv_start[row] = cu[row] + aligned_len - track_rows.conv_len
        o = self._write(o=o, values=track_chunk_idx)
        o = self._write(o=o, values=track_mode)
        o = self._write(o=o, values=track_conv_start)
        o = self._write(o=o, values=np.array([total]))
        assert o == host.shape[0]

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


def build_track_rows(
    *,
    track_mask: list[bool],
    track_seq_lens: list[int],
    extend_seq_lens: list[int],
    extend_prefix_lens: list[int],
    state_chunk_size: int,
    cache_chunk_size: int,
    conv_len: int,
    dst: torch.Tensor,
) -> Optional[KDAPrefillTrackRows]:
    """The eager track selection (_init_track_ssm_indices_from_cpu and
    _init_track_conv_indices) as per-row lists; None when nothing is tracked."""
    if not any(track_mask):
        return None
    plan = build_prefill_track_plan(
        track_mask,
        track_seq_lens,
        extend_seq_lens,
        extend_prefix_lens,
        state_chunk_size,
        mamba2=False,
    )
    conv_rows = [
        (
            row,
            (track_seq_lens[row] - extend_prefix_lens[row])
            // cache_chunk_size
            * cache_chunk_size,
        )
        for row in plan.tracked_rows
    ]
    return KDAPrefillTrackRows(
        chunk_rows=[(row, plan.chunk_indices[row]) for row in plan.unaligned_rows],
        final_rows=list(plan.final_rows),
        conv_rows=conv_rows,
        conv_len=conv_len,
        dst=dst,
    )


def kda_extend_in_graph(
    *,
    layer: RadixLinearAttention,
    meta: KDAPrefillGraphMetadata,
    conv_pool: torch.Tensor,
    ssm_states: torch.Tensor,
    kernel_dispatcher: KDAKernelDispatcher,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """KDAAttnBackend.forward_extend with launch grids that depend only on the
    bucket. ``conv_pool`` is the layer's [slots, kernel-1, dim] conv cache.
    Returns [1, num_tokens, H, V]; rows past the batch's tokens are zero."""
    if meta.track:
        # Snapshot the conv window before the conv kernel updates the cache;
        # the source is the raw input, the destinations are track slots.
        kda_track_conv_window(
            conv_pool=conv_pool,
            x=mixed_qkv,
            track_mode=meta.track_mode,
            track_conv_start=meta.track_conv_start,
            track_dst=meta.track_dst,
            num_tokens=meta.num_logical_tokens,
        )

    qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation="silu",
        conv_states=conv_pool.transpose(-1, -2),
        has_initial_state=meta.has_initial_state,
        cache_indices=meta.cache_indices,
        query_start_loc=meta.cu_seqlens,
        seq_lens_cpu=None,
        block_table=meta.conv_block_table,
    ).transpose(0, 1)
    q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
    q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
    k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
    v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

    gate_was_flat = a.ndim == 3
    if gate_was_flat:
        a = a.unflatten(-1, (-1, layer.head_k_dim))

    h_track = None
    if meta.track:
        h_track = torch.empty(
            (meta.max_seqs, *ssm_states.shape[1:]),
            dtype=torch.float32,
            device=ssm_states.device,
        )
    core_attn_out = kernel_dispatcher.extend(
        q=q,
        k=k,
        v=v,
        g=a,
        beta=b,
        ssm_states=ssm_states,
        cache_indices=meta.cache_indices,
        query_start_loc=meta.cu_seqlens,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        lower_bound=layer.lower_bound,
        beta_is_raw=gate_was_flat,
        extend_seq_lens_cpu=None,
        is_spec_decode=False,
        return_intermediate_states=False,
        track_state=h_track,
        track_chunk_idx=meta.track_chunk_idx if meta.track else None,
        chunk_indices=meta.chunk_indices,
        chunk_offsets=meta.chunk_offsets,
    )

    if meta.track:
        kda_track_ssm_state(
            ssm_pool=ssm_states,
            h_track=h_track,
            track_mode=meta.track_mode,
            track_dst=meta.track_dst,
            cache_indices=meta.cache_indices,
        )

    # Rows past the batch belong to no sequence and were never written; keep the
    # layer's padded output finite for the MoE and the collectives.
    core_attn_out.masked_fill_(
        (meta.token_index >= meta.num_logical_tokens).view(1, -1, 1, 1), 0
    )
    return core_attn_out

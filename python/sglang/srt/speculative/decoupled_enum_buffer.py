"""Verifier-side GPU landing buffer for decoupled enumeration spec.

The drafter ships one DraftEnumerationBufferBatch per round (one round ahead),
pre-enumerating every (accept_case, bonus_guess) chain the verifier could select
next; this is where those blocks land on the verifier's GPU for the verify
forward to consume.

Pool-indexed by req_pool_idx like FutureMap's relays (managers/overlap_utils.py),
so the forward gathers it with the same batch.req_pool_indices as every other
per-request tensor; leading dim = req_to_token.shape[0] (max_running + 1; seat 0
is the harmless cuda-graph padding row). Each seat carries a base_committed_len
stamp; the verify forward (phase 4a) compares it against the request's live
committed length for fresh-vs-fallback, and a never-written / reset seat holds a
sentinel that always falls back.

Phase 1b is SYNC (buf_count == 1: land scatters on the current stream, the caller
synchronizes before gather). The async overlap form (pinned staging + a private
copy stream + a double-buffer swap fence) lands in phase 6.3; the API is shaped so
6.3 only flips buf_count and moves the H2D onto the copy stream.
"""

from __future__ import annotations

import numpy as np
import torch

from sglang.srt.speculative.decoupled_slot_table import (
    DecoupledSlotTable,
    plan_landing,
)
from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch

# Stamp for a seat with no valid block (never written, or reset on realloc); a
# negative committed length never matches a real one, so the seat falls back.
_STAMP_EMPTY = -1


class DecoupledEnumBuffer:
    """GPU landing buffer for enumeration blocks, indexed by req_pool_idx."""

    def __init__(
        self,
        *,
        device: str,
        req_to_token_pool,
        num_steps: int,  # K = draft chain length per case
        fanout: int,  # F = bonus-token guesses per accept case
        verifier_rank: int,
        enable_overlap: bool,
    ) -> None:
        if enable_overlap:
            raise NotImplementedError(
                "overlap landing needs the async copy-stream + swap fence "
                "(phase 6.3); run with enable_overlap=False for now"
            )
        self.device = device
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        # From DecoupledSpecIpcConfig.rank in phase 5b; land() rejects a block
        # routed to a different verifier.
        self.verifier_rank = int(verifier_rank)
        # (K+1) accept cases * F bonus guesses * K chain steps, flat per row.
        self.row_width = (self.num_steps + 1) * self.fanout * self.num_steps
        # req_to_token.shape[0] == max_running + 1, so seat 0 stays the harmless
        # cuda-graph padding row; never size to bare max_running.
        self.seats = int(req_to_token_pool.req_to_token.shape[0])
        # Double-buffer (mamba idiom, memory_pool.py:849) is the phase 6.3 form;
        # buf_count is always 1 here (enable_overlap is rejected above).
        self.buf_count = 2 if enable_overlap else 1
        self._write_slot = 0

        # int64 matches the forward's token-id convention (input_ids,
        # FutureMap.output_tokens_buf, EagleDraftInput.draft_token are all int64);
        # int32 is numerically enough but would force an up-cast. Not
        # req_to_token's int32, which is a KV-slot index pool, not vocab ids.
        self.enum_tokens = [
            torch.zeros((self.seats, self.row_width), dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]
        # Per-seat freshness stamp; starts at the sentinel so an unwritten seat
        # falls back. int64 to match the committed length it is compared against.
        self.enum_base_committed_lens = [
            torch.full((self.seats,), _STAMP_EMPTY, dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]

    @property
    def _read_slot(self) -> int:
        # Single buffer: daemon and forward share one slot (SYNC). Double buffer:
        # the forward reads the slot the daemon is not writing.
        return self._write_slot if self.buf_count == 1 else 1 - self._write_slot

    def land(
        self,
        block: DraftEnumerationBufferBatch,
        slot_table: DecoupledSlotTable,
    ) -> None:
        """Scatter a block's surviving rows + stamps into their seats.

        Called by the recv daemon; routing is decided by plan_landing. Phase 1b
        scatters on the current stream (SYNC); phase 6.3 moves it onto a private
        copy stream with pinned staging.

        NOTE: like plan_landing, the raise below runs on the recv daemon thread,
        whose loop dies on an uncaught exception. TODO(phase 5c): quarantine.
        """
        if int(block.num_steps) != self.num_steps or int(block.fanout) != self.fanout:
            raise RuntimeError(
                "enumeration block dims differ from the buffer's config "
                "(mismatched K/F shape-errors the scatter, or silently mis-lays "
                "out the flat [accept_case][guess][step] layout if the products "
                f"coincide): block=({block.num_steps}, {block.fanout}) "
                f"buffer=({self.num_steps}, {self.fanout})"
            )
        plan = plan_landing(block, slot_table, verifier_rank=self.verifier_rank)
        if not plan.writes:
            return

        # Reshape the block's flat token tuple once (C-order, so rows_host[i] ==
        # block.row_tokens(i)); select survivors; blocking H2D from pageable host
        # (the copy stream + pinned staging arrive in phase 6.3).
        pool_indices = torch.tensor(
            [w.pool_idx for w in plan.writes], dtype=torch.int64, device=self.device
        )
        rows_host = torch.from_numpy(
            np.asarray(block.tokens, dtype=np.int64).reshape(
                block.batch_size, block.row_stride
            )
        )
        if len(plan.writes) == block.batch_size:
            rows_selected = rows_host  # nothing dropped; row order preserved
        else:
            row_indices = torch.tensor(
                [w.row_index for w in plan.writes], dtype=torch.int64
            )
            rows_selected = rows_host[row_indices]
        rows = rows_selected.to(device=self.device)
        base_committed_lens = torch.tensor(
            [w.base_committed_len for w in plan.writes],
            dtype=torch.int64,
            device=self.device,
        )

        slot = self._write_slot
        self.enum_tokens[slot][pool_indices] = rows
        self.enum_base_committed_lens[slot][pool_indices] = base_committed_lens

    def gather(
        self, req_pool_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this batch's rows + freshness stamps from the read-side slot:
        (rows [B, (K+1)*F*K], base_committed_lens [B]). The verify forward (phase
        4a) compares base_committed_lens against the live committed length for
        fresh-vs-fallback, then selects the winning chain by (accept_case,
        bonus_guess).
        """
        slot = self._read_slot
        rows = self.enum_tokens[slot][req_pool_indices]
        base_committed_lens = self.enum_base_committed_lens[slot][req_pool_indices]
        return rows, base_committed_lens

    def reset_slot(self, pool_idx: int) -> None:
        # Invalidate a seat's stamp when it is (re)assigned, so the reused seat
        # falls back until its new occupant's own block lands. Called by the
        # scheduler at prefill alloc / retraction re-admit.
        for slot in range(self.buf_count):
            self.enum_base_committed_lens[slot][pool_idx] = _STAMP_EMPTY

    def swap(self) -> None:
        # Advance the write/read double-buffer at a round boundary; no-op under
        # buf_count == 1 (phase 6.3 pairs this with the copy-stream event fence).
        if self.buf_count > 1:
            self._write_slot = 1 - self._write_slot

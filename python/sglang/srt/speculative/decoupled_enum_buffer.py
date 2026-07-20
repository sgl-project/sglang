"""Verifier-side GPU landing buffer for decoupled enumeration spec.

The drafter ships one :class:`DraftEnumerationBufferBatch` per round, one round
ahead: for each request it pre-enumerates every ``(accept_case, bonus_guess)``
chain the verifier could select next. This buffer is where those blocks land on
the verifier's GPU so the verify forward can consume them.

Design (see also :mod:`decoupled_slot_table`):

* **Pool-indexed.** The buffer is indexed by ``req_pool_idx`` (the seat), exactly
  like ``FutureMap``'s relays (``managers/overlap_utils.py``), so the verify
  forward gathers it with the same ``batch.req_pool_indices`` it uses for every
  other per-request tensor. Leading dim = ``req_to_token.shape[0]``
  (``max_running + 1``; seat 0 is the cuda-graph padding row, kept harmless).
  Trailing dim = ``(K+1) * F * K`` = the flat enumeration block per request.
* **Sole writer = the recv daemon** (:meth:`land`); **reader = the verify
  forward** (:meth:`gather`). They run on different threads/streams, so under
  overlap the device buffers are double-buffered (write side vs read side).
* **Stamp = (base_committed_len, generation).** The consumer (verify worker,
  phase 4a) compares the stamp against the request's live committed length /
  ``req_generation`` to decide fresh-vs-fallback on the GPU; a stale / reused /
  never-written seat fails the check and falls back to a plain bonus decode.

Phase boundary: phase 1b ships the SYNC form (``buf_count == 1``: :meth:`land`
scatters on the current stream, the caller synchronizes before :meth:`gather`).
The async overlap path -- pinned staging + a private copy stream + the
double-buffer swap fence so the daemon can write one round ahead while a forward
reads the other slot -- lands in phase 6.3; the API here is shaped so 6.3 only
flips ``buf_count`` and moves the H2D onto the copy stream.
"""

from __future__ import annotations

import numpy as np
import torch

from sglang.srt.speculative.decoupled_slot_table import (
    DecoupledSlotTable,
    plan_landing,
)
from sglang.srt.speculative.decoupled_spec_io import DraftEnumerationBufferBatch

# Stamp sentinel for a seat that holds no valid block (never written, or just
# reset on (re)allocation). A negative base can never equal a real committed
# length, so the downstream freshness check always falls back for such a seat.
_STAMP_EMPTY = -1


class DecoupledEnumBuffer:
    """GPU landing buffer for enumeration blocks, indexed by ``req_pool_idx``."""

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
                "DecoupledEnumBuffer overlap landing is not implemented yet: "
                "double-buffered landing requires the async copy-stream + swap "
                "fence and arrives in phase 6.3. Run with enable_overlap=False "
                "(single-buffer SYNC mode) until then."
            )
        self.device = device
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        # verifier_rank is init-static (it will come from
        # DecoupledSpecIpcConfig.rank in phase 5b); land() rejects any block
        # routed to a different verifier.
        self.verifier_rank = int(verifier_rank)
        # (K+1) accept cases * F bonus guesses * K chain steps, flattened per row.
        self.row_width = (self.num_steps + 1) * self.fanout * self.num_steps
        # Size to the pool's row count (== max_running + 1) so seat 0 stays the
        # harmless cuda-graph padding row; never size to bare max_running.
        self.seats = int(req_to_token_pool.req_to_token.shape[0])
        # Double-buffering (mamba idiom, memory_pool.py:849) is the phase 6.3
        # overlap form; until enable_overlap is accepted above, buf_count is
        # always 1 and the daemon and forward share one slot in SYNC mode.
        self.buf_count = 2 if enable_overlap else 1
        self._write_slot = 0

        # int64 (torch.long) matches the forward's token-id convention: these
        # rows are consumed as input_ids (int64) and are the analog of
        # FutureMap.output_tokens_buf / EagleDraftInput.draft_token (both int64).
        # int32 would suffice numerically (vocab < 2^31) but would force an
        # up-cast on the forward path -- do NOT model this on req_to_token
        # (int32), which is a KV-slot index pool, not vocab ids.
        self.enum_tokens = [
            torch.zeros((self.seats, self.row_width), dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]
        # Stamp travels as two parallel per-seat arrays; both start at the
        # sentinel so an unwritten seat reads as fallback. int64 matches the
        # values compared downstream: ReqToTokenPool.req_generation and the
        # request's committed length / seq_lens (all int64).
        self.enum_base_committed_lens = [
            torch.full((self.seats,), _STAMP_EMPTY, dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]
        self.enum_generations = [
            torch.full((self.seats,), _STAMP_EMPTY, dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]

    @property
    def _read_slot(self) -> int:
        # Under a single buffer the daemon and forward share one slot (SYNC);
        # under double-buffering the forward reads the slot the daemon is not
        # currently writing.
        return self._write_slot if self.buf_count == 1 else 1 - self._write_slot

    def land(
        self,
        block: DraftEnumerationBufferBatch,
        slot_table: DecoupledSlotTable,
    ) -> None:
        """Scatter an incoming block's rows into their seats on the GPU.

        Called by the recv daemon. Routing (which rows survive, which are dropped
        because their request left) is decided host-side by
        :func:`plan_landing`; the surviving rows plus their stamps are scattered
        into the write-side buffer at their seats.

        Phase 1b: the scatter runs on the current stream (SYNC). Phase 6.3 moves
        it onto a private copy stream with pinned staging + non_blocking H2D so
        the daemon can write ahead of the forward.

        HAZARD: the geometry raise below (and every raise inside
        :func:`plan_landing`) executes on the decoupled recv daemon thread, whose
        loop does not survive an uncaught exception (same hazard documented on
        ``VerifierCommitSegment.append_message`` in ``decoupled_spec_io``). A
        single malformed / misrouted peer message therefore stops landing for ALL
        requests. TODO(phase 5c): quarantine peer-data violations (drop + close
        the offending request) instead of crashing the thread.
        """
        if int(block.num_steps) != self.num_steps or int(block.fanout) != self.fanout:
            raise RuntimeError(
                "Enumeration block dims differ from the buffer's config: a "
                "mismatched (K, F) either shape-errors on the scatter or, if the "
                "products coincidentally match, silently mis-lays out the "
                "[accept_case][guess][step] flat layout: "
                f"block=(num_steps={block.num_steps}, fanout={block.fanout}) "
                f"buffer=(num_steps={self.num_steps}, fanout={self.fanout})"
            )
        plan = plan_landing(block, slot_table, verifier_rank=self.verifier_rank)
        if not plan.writes:
            return

        # Build the host-side scatter tensors from the surviving rows and copy
        # them to the device. Every host->device transfer below (the .to(device)
        # and each torch.tensor(..., device=self.device)) is a BLOCKING copy from
        # pageable host memory in phase 1b, on the recv daemon thread. Phase 6.3
        # moves these onto a private copy stream with pinned staging +
        # non_blocking H2D so the daemon can write ahead of the forward.
        # rows: reshape the block's whole flat token tuple ONCE (single C-level
        # pass); C-order reshape makes rows_host[i] exactly equal to
        # block.row_tokens(i).
        pool_indices = torch.tensor(
            [w.pool_idx for w in plan.writes], dtype=torch.int64, device=self.device
        )
        rows_host = torch.from_numpy(
            np.asarray(block.tokens, dtype=np.int64).reshape(
                block.batch_size, block.row_stride
            )
        )
        if len(plan.writes) == block.batch_size:
            # No rows dropped: plan_landing preserves row order, so
            # writes[i].row_index == i and every row survives -- use rows_host
            # directly with no gather.
            rows_selected = rows_host
        else:
            row_indices = torch.tensor(
                [w.row_index for w in plan.writes], dtype=torch.int64
            )
            rows_selected = rows_host[row_indices]
        rows = rows_selected.to(device=self.device)  # blocking H2D (pageable src)
        base_committed_lens = torch.tensor(
            [w.base_committed_len for w in plan.writes],
            dtype=torch.int64,
            device=self.device,
        )
        generations = torch.tensor(
            [w.generation for w in plan.writes],
            dtype=torch.int64,
            device=self.device,
        )

        slot = self._write_slot
        self.enum_tokens[slot][pool_indices] = rows
        self.enum_base_committed_lens[slot][pool_indices] = base_committed_lens
        self.enum_generations[slot][pool_indices] = generations

    def gather(
        self, req_pool_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather this batch's enumeration rows + stamps from the read-side slot.

        Returns ``(rows, base_committed_lens, generations)`` where ``rows`` is
        ``[B, (K+1)*F*K]`` and ``base_committed_lens`` / ``generations`` are
        ``[B]``. This is only the first (pool-index) gather layer; the verify
        worker (phase 4a) compares base_committed_lens/generations against the
        live committed length / ``req_generation`` for the freshness decision,
        then selects the winning chain by ``(accept_case, bonus_guess)``.
        """
        slot = self._read_slot
        rows = self.enum_tokens[slot][req_pool_indices]
        base_committed_lens = self.enum_base_committed_lens[slot][req_pool_indices]
        generations = self.enum_generations[slot][req_pool_indices]
        return rows, base_committed_lens, generations

    def reset_slot(self, pool_idx: int) -> None:
        """Invalidate a seat's stamp when it is (re)assigned to a request.

        Called by the scheduler at prefill alloc / retraction re-admit, so a
        reused seat reads as fallback until its new occupant's own block lands.
        Resets both double-buffer sides.
        """
        for slot in range(self.buf_count):
            self.enum_base_committed_lens[slot][pool_idx] = _STAMP_EMPTY
            self.enum_generations[slot][pool_idx] = _STAMP_EMPTY

    def swap(self) -> None:
        """Advance the write/read double-buffer at a round boundary.

        No-op under ``buf_count == 1`` (phase 1b SYNC). Phase 6.3 pairs this with
        the copy-stream event fence so the forward never reads a half-written
        slot.
        """
        if self.buf_count > 1:
            self._write_slot = 1 - self._write_slot

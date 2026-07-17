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

import torch

from sglang.srt.speculative.decoupled_slot_table import (
    DecoupledSlotTable,
    LandingPlan,
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
        enable_overlap: bool,
    ) -> None:
        self.device = device
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        # (K+1) accept cases * F bonus guesses * K chain steps, flattened per row.
        self.row_width = (self.num_steps + 1) * self.fanout * self.num_steps
        # Size to the pool's row count (== max_running + 1) so seat 0 stays the
        # harmless cuda-graph padding row; never size to bare max_running.
        self.seats = int(req_to_token_pool.req_to_token.shape[0])
        # Double-buffer only under overlap (mamba idiom, memory_pool.py:849);
        # phase 1b runs a single buffer in SYNC mode.
        self.buf_count = 2 if enable_overlap else 1
        self._write_slot = 0

        self.enum_tok = [
            torch.zeros((self.seats, self.row_width), dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]
        # Stamp travels as two parallel per-seat arrays; both start at the
        # sentinel so an unwritten seat reads as fallback.
        self.enum_base = [
            torch.full((self.seats,), _STAMP_EMPTY, dtype=torch.int64, device=device)
            for _ in range(self.buf_count)
        ]
        self.enum_gen = [
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
    ) -> LandingPlan:
        """Scatter an incoming block's rows into their seats on the GPU.

        Called by the recv daemon. Routing (which rows survive, which are dropped
        because their request left) is decided host-side by
        :func:`plan_landing`; the surviving rows plus their stamps are scattered
        into the write-side buffer at their seats. Returns the plan so the caller
        can observe / log drops.

        Phase 1b: the scatter runs on the current stream (SYNC). Phase 6.3 moves
        it onto a private copy stream with pinned staging + non_blocking H2D so
        the daemon can write ahead of the forward.
        """
        plan = plan_landing(block, slot_table)
        if not plan.writes:
            return plan

        # Build the host-side scatter tensors from the surviving rows. (Phase 6.3
        # will stage these through a reusable pinned buffer instead of a fresh
        # allocation per land.)
        pool_idx = torch.tensor(
            [w.pool_idx for w in plan.writes], dtype=torch.int64, device=self.device
        )
        rows = torch.tensor(
            [block.row_tokens(w.row_index) for w in plan.writes],
            dtype=torch.int64,
            device=self.device,
        )
        base = torch.tensor(
            [w.base_committed_len for w in plan.writes],
            dtype=torch.int64,
            device=self.device,
        )
        gen = torch.tensor(
            [w.generation for w in plan.writes],
            dtype=torch.int64,
            device=self.device,
        )

        slot = self._write_slot
        self.enum_tok[slot][pool_idx] = rows
        self.enum_base[slot][pool_idx] = base
        self.enum_gen[slot][pool_idx] = gen
        return plan

    def gather(
        self, req_pool_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather this batch's enumeration rows + stamps from the read-side slot.

        Returns ``(rows, base, gen)`` where ``rows`` is ``[B, (K+1)*F*K]`` and
        ``base`` / ``gen`` are ``[B]``. This is only the first (pool-index) gather
        layer; the verify worker (phase 4a) compares base/gen against the live
        committed length / ``req_generation`` for the freshness decision, then
        selects the winning chain by ``(accept_case, bonus_guess)``.
        """
        slot = self._read_slot
        rows = self.enum_tok[slot][req_pool_indices]
        base = self.enum_base[slot][req_pool_indices]
        gen = self.enum_gen[slot][req_pool_indices]
        return rows, base, gen

    def reset_slot(self, pool_idx: int) -> None:
        """Invalidate a seat's stamp when it is (re)assigned to a request.

        Called by the scheduler at prefill alloc / retraction re-admit, so a
        reused seat reads as fallback until its new occupant's own block lands.
        Resets both double-buffer sides.
        """
        for slot in range(self.buf_count):
            self.enum_base[slot][pool_idx] = _STAMP_EMPTY
            self.enum_gen[slot][pool_idx] = _STAMP_EMPTY

    def swap(self) -> None:
        """Advance the write/read double-buffer at a round boundary.

        No-op under ``buf_count == 1`` (phase 1b SYNC). Phase 6.3 pairs this with
        the copy-stream event fence so the forward never reads a half-written
        slot.
        """
        if self.buf_count > 1:
            self._write_slot = 1 - self._write_slot

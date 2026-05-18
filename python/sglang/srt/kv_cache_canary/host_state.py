from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sglang.jit_kernel.kv_cache_canary import VIOLATION_FIELDS
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.fingerprint import mix_step


@dataclass(frozen=True, slots=True, kw_only=True)
class _LastCommitted:
    """Last committed write for a request — used as the (single) verify target.

    Recording the slot index lets the verify entry read the *same physical
    location* the kernel wrote last step, independent of any later
    ``req_to_token_pool`` updates (perturbation, paging, etc.).
    """

    token_id: int
    position: int
    slot_idx: int
    prev_hash_at_write: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _RequestState:
    prev_hash_tail: int
    k_req: int
    last_committed: Optional[_LastCommitted] = None


class CanaryHostState:
    """Per-request canary high-water-mark, chain-hash tail, and verify target.

    Indexed by ``req_pool_idx`` (the stable per-request index assigned by
    ``ReqToTokenPool``). The GPU side only sees the expected fingerprints we
    compute here; this class owns the host bookkeeping.
    """

    def __init__(self, *, config: CanaryConfig, num_req_slots: int) -> None:
        self._config = config
        self._states: Dict[int, _RequestState] = {}
        self._lock = threading.Lock()

    def _initial_state(self) -> _RequestState:
        return _RequestState(prev_hash_tail=self._config.seed, k_req=0)

    def reset_request(self, req_pool_idx: int) -> None:
        with self._lock:
            self._states.pop(req_pool_idx, None)

    def reset_request_for_slot(self, slot_idx: int) -> None:
        """Drop any host state whose last_committed lives in this physical slot.

        Used by SWA window-slide eviction: when a slot is evicted from the
        sliding window, any request that thought its verify-target lived in
        that slot must forget it (otherwise next plan_batch builds a stale
        verify entry).
        """
        with self._lock:
            stale_req_pool_idxs: List[int] = []
            for req_pool_idx, state in self._states.items():
                if state.last_committed is None:
                    continue
                if state.last_committed.slot_idx == slot_idx:
                    stale_req_pool_idxs.append(req_pool_idx)
            for req_pool_idx in stale_req_pool_idxs:
                self._states.pop(req_pool_idx, None)

    def export_pd_snapshot(self, req_pool_idx: int) -> Optional["PDSnapshot"]:
        """Return ``(k_req, prev_hash_tail)`` for PD transport.

        Prefill side calls this when packing ``MetadataBuffers``. ``None``
        means the request was never seen (e.g. zero-length prefill, canary
        kernel didn't run yet) — caller stores 0/0 in the metadata buffer and
        decode side treats it as a fresh chain.
        """
        with self._lock:
            state = self._states.get(req_pool_idx)
            if state is None:
                return None
            return PDSnapshot(k_req=state.k_req, prev_hash_tail=state.prev_hash_tail)

    def import_pd_snapshot(
        self,
        *,
        req_pool_idx: int,
        k_req: int,
        prev_hash_tail: int,
    ) -> None:
        """Decode side: rebuild host state from PD-transported metadata.

        Drops any prior state for this slot (decode never shares req_pool_idx
        with itself; if it did, we'd just have stale data). ``last_committed``
        is None because we don't have the last-token snapshot — the chain
        continues from ``prev_hash_tail`` at position ``k_req``, but the very
        first decode forward pure-writes (no verify entry).
        """
        with self._lock:
            self._states[req_pool_idx] = _RequestState(
                prev_hash_tail=prev_hash_tail & ((1 << 64) - 1),
                k_req=int(k_req),
                last_committed=None,
            )

    def reset_all_last_committed(self) -> None:
        """Drop ``last_committed`` for every tracked request.

        Conservative SWA window-slide fallback: when we can't enumerate which
        slots were just evicted (e.g. the free hook only fires with no
        argument), wiping last_committed on all requests prevents stale
        verify entries. The next forward will pure-write and re-anchor.
        """
        with self._lock:
            for req_pool_idx, state in list(self._states.items()):
                if state.last_committed is None:
                    continue
                self._states[req_pool_idx] = _RequestState(
                    prev_hash_tail=state.prev_hash_tail,
                    k_req=state.k_req,
                    last_committed=None,
                )

    def reset_request_to(self, *, req_pool_idx: int, k_req: int) -> None:
        """Roll a request's high-water mark back to ``k_req`` (spec reject path).

        Spec decoding rejects drafted tokens after the target verifies them;
        the rejected tokens' slots get freed and reused. The canary chain
        must rewind ``K_req`` to the new low-water mark so the next batch's
        verify target is the last *accepted* token (not the rejected ones).
        """
        with self._lock:
            existing = self._states.get(req_pool_idx)
            if existing is None:
                return
            if k_req <= 0:
                self._states.pop(req_pool_idx, None)
                return
            # We don't have a per-position prev_hash snapshot, so the safest
            # behaviour is to drop last_committed (no verify entry until a
            # fresh write commits) but keep prev_hash_tail at the new k_req
            # boundary. Callers can pass k_req = old k_req to no-op.
            new_state = _RequestState(
                prev_hash_tail=existing.prev_hash_tail,
                k_req=min(existing.k_req, k_req),
                last_committed=None,
            )
            self._states[req_pool_idx] = new_state

    def has_state(self, req_pool_idx: int) -> bool:
        with self._lock:
            return req_pool_idx in self._states

    def plan_batch(
        self,
        *,
        req_pool_indices: List[int],
        req_token_counts: List[int],
        req_start_positions: List[int],
        input_tokens_per_req: List[List[int]],
        write_slot_indices_per_req: List[List[int]],
    ) -> "BatchPlan":
        """Compute expected (req_id, token_id, position, prev_hash) for the batch.

        The batch the kernel sees is laid out as ``[verify_entries..., write_entries...]``:

        - **verify entries**: one per request that already has a committed write
          (``state.last_committed is not None``). The kernel reads
          ``src_buf[last_slot_idx]`` and checks the stored triple against the
          stored expected values + monotonic-position constraint (see
          ``verify_seq_position`` below).
        - **write entries**: ``sum(req_token_counts)`` slots laid out the same
          order as the caller's flattened ``input_ids`` / ``out_cache_loc``.

        ``write_slot_indices_per_req`` is the list of physical slot indices for
        this batch's writes, partitioned by request. We do not actually use them
        for verify-target indexing here (we use ``state.last_committed.slot_idx``),
        but we DO use them to update ``state.last_committed`` in ``commit_plan``.
        """
        if not (
            len(req_pool_indices)
            == len(req_token_counts)
            == len(req_start_positions)
            == len(input_tokens_per_req)
            == len(write_slot_indices_per_req)
        ):
            raise RuntimeError(
                "kv-canary: plan_batch input lists have mismatched lengths"
            )

        verify_req_ids: List[int] = []
        verify_token_ids: List[int] = []
        verify_positions: List[int] = []
        verify_prev_hashes: List[int] = []
        verify_seq_positions: List[int] = []
        verify_slot_indices: List[int] = []

        write_req_ids: List[int] = []
        write_token_ids: List[int] = []
        write_positions: List[int] = []
        write_prev_hashes: List[int] = []
        next_state: Dict[int, _RequestState] = {}

        with self._lock:
            for req_pool_idx, count, start_pos, tokens, write_slots in zip(
                req_pool_indices,
                req_token_counts,
                req_start_positions,
                input_tokens_per_req,
                write_slot_indices_per_req,
            ):
                if count != len(tokens) or count != len(write_slots):
                    raise RuntimeError(
                        "kv-canary: per-request count must match tokens and write_slots length"
                    )
                state = self._states.get(req_pool_idx) or self._initial_state()

                if state.last_committed is not None:
                    lc = state.last_committed
                    verify_req_ids.append(req_pool_idx)
                    verify_token_ids.append(lc.token_id)
                    verify_positions.append(lc.position)
                    verify_prev_hashes.append(to_signed_int64(lc.prev_hash_at_write))
                    verify_seq_positions.append(lc.position)
                    verify_slot_indices.append(lc.slot_idx)

                prev_hash = state.prev_hash_tail
                last_token_id = 0
                last_position = -1
                last_slot_idx = -1
                last_prev_hash_at_write = state.prev_hash_tail
                for offset in range(count):
                    pos = start_pos + offset
                    token_id = tokens[offset]
                    slot_idx = write_slots[offset]
                    write_req_ids.append(req_pool_idx)
                    write_token_ids.append(token_id)
                    write_positions.append(pos)
                    write_prev_hashes.append(to_signed_int64(prev_hash))
                    last_token_id = token_id
                    last_position = pos
                    last_slot_idx = slot_idx
                    last_prev_hash_at_write = prev_hash
                    prev_hash = mix_step(prev_hash, token_id, pos)

                if count > 0:
                    new_last_committed = _LastCommitted(
                        token_id=last_token_id,
                        position=last_position,
                        slot_idx=last_slot_idx,
                        prev_hash_at_write=last_prev_hash_at_write,
                    )
                    next_state[req_pool_idx] = _RequestState(
                        prev_hash_tail=prev_hash,
                        k_req=max(state.k_req, start_pos + count),
                        last_committed=new_last_committed,
                    )
                else:
                    next_state[req_pool_idx] = state

        num_verify = len(verify_req_ids)
        num_write = len(write_req_ids)
        write_slot_indices_flat: List[int] = []
        for w in write_slot_indices_per_req:
            write_slot_indices_flat.extend(int(s) for s in w)

        expected_req_ids = verify_req_ids + write_req_ids
        expected_token_ids = verify_token_ids + write_token_ids
        expected_positions = verify_positions + write_positions
        expected_prev_hashes = verify_prev_hashes + write_prev_hashes
        verify_mask = [1] * num_verify + [0] * num_write
        verify_seq_positions_full = verify_seq_positions + [-1] * num_write

        return BatchPlan(
            expected_req_ids=expected_req_ids,
            expected_token_ids=expected_token_ids,
            expected_positions=expected_positions,
            expected_prev_hashes=expected_prev_hashes,
            verify_mask=verify_mask,
            verify_seq_positions=verify_seq_positions_full,
            verify_slot_indices=verify_slot_indices,
            write_slot_indices=write_slot_indices_flat,
            num_verify=num_verify,
            num_write=num_write,
            next_state=next_state,
        )

    def commit_plan(self, plan: "BatchPlan") -> None:
        with self._lock:
            for req_pool_idx, new_state in plan.next_state.items():
                self._states[req_pool_idx] = new_state


@dataclass(frozen=True, slots=True, kw_only=True)
class PDSnapshot:
    """K_req + prev_hash_tail packed into MetadataBuffers for PD transport."""

    k_req: int
    prev_hash_tail: int


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchPlan:
    expected_req_ids: List[int]
    expected_token_ids: List[int]
    expected_positions: List[int]
    expected_prev_hashes: List[int]
    verify_mask: List[int]
    verify_seq_positions: List[int]
    verify_slot_indices: List[int]
    write_slot_indices: List[int]
    num_verify: int
    num_write: int
    next_state: Dict[int, _RequestState]


def to_signed_int64(unsigned_value: int) -> int:
    assert (
        0 <= unsigned_value < (1 << 64)
    ), f"kv-canary: prev_hash out of unsigned-64 range: {unsigned_value:#x}"
    mask = (1 << 64) - 1
    value = unsigned_value & mask
    if value >= (1 << 63):
        value -= 1 << 64
    return value


@dataclass(slots=True)
class CanaryDeviceState:
    """GPU-resident state shared across head/tail kernel invocations."""

    violation_ring: torch.Tensor
    violation_ring_valid: torch.Tensor
    violation_write_index: torch.Tensor
    first_violation: torch.Tensor
    first_violation_set: torch.Tensor
    is_errored: torch.Tensor
    slot_run_counter_head: torch.Tensor
    slot_run_counter_tail: torch.Tensor
    kernel_run_counter_head: torch.Tensor
    kernel_run_counter_tail: torch.Tensor

    @classmethod
    def allocate(
        cls, *, device: torch.device, ring_capacity: int
    ) -> "CanaryDeviceState":
        return cls(
            violation_ring=torch.zeros(
                ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            violation_ring_valid=torch.zeros(
                ring_capacity, dtype=torch.int32, device=device
            ),
            violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
            first_violation=torch.zeros(
                VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            first_violation_set=torch.zeros(1, dtype=torch.int32, device=device),
            is_errored=torch.zeros(1, dtype=torch.int32, device=device),
            slot_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            slot_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
        )

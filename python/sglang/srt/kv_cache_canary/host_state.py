from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import VIOLATION_FIELDS
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.fingerprint import mix_step


@dataclass(frozen=True, slots=True, kw_only=True)
class _HistoryEntry:
    """One historical write for a request — a single verify target.

    Recording the slot index here (independently of ``req_to_token_pool``)
    is what makes the position-monotonic check (README §3 (b)) capable of
    catching map perturbation: the kernel reads ``src_buf[slot_idx]`` and
    checks that the slot's stored ``position`` field equals this entry's
    ``position``. Any redirect of the table to a different slot will produce
    a different ``position`` field and trip the check.
    """

    token_id: int
    position: int
    slot_idx: int
    prev_hash_at_write: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _RequestState:
    prev_hash_tail: int
    k_req: int
    # Full history of committed writes for this req in position order
    # (history[i].position == i). Verify entries cover ``[0, k_req)`` so
    # any prior slot's content can be re-checked, not just the last one.
    history: Tuple["_HistoryEntry", ...] = ()


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
        """Drop any host state whose history references this physical slot.

        Used by SWA window-slide eviction: when a slot is evicted from the
        sliding window, any request whose verify history touches that slot
        must forget it (otherwise the next plan_batch builds a verify entry
        that reads a slot now reused by a stranger).
        """
        with self._lock:
            stale_req_pool_idxs: List[int] = []
            for req_pool_idx, state in self._states.items():
                if any(entry.slot_idx == slot_idx for entry in state.history):
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
        with itself; if it did, we'd just have stale data). ``history`` is
        empty because we don't have per-position snapshots — the chain
        continues from ``prev_hash_tail`` at position ``k_req``, but the very
        first decode forward pure-writes (no verify entries) until enough
        history accumulates on the decode side.
        """
        with self._lock:
            self._states[req_pool_idx] = _RequestState(
                prev_hash_tail=prev_hash_tail & ((1 << 64) - 1),
                k_req=int(k_req),
                history=(),
            )

    def reset_all_last_committed(self) -> None:
        """Drop the verify history of every tracked request.

        Conservative SWA window-slide / spec-reject fallback: when we can't
        enumerate which slots were just evicted (e.g. the free hook only
        fires with no argument), wiping history prevents stale verify
        entries pointing at slots a stranger may now own. The next forward
        will pure-write and re-anchor; ``prev_hash_tail`` and ``k_req`` are
        kept so chain continuity is preserved.

        Method name retained for backwards source-compat; the entire
        history (not just the last-committed entry) is dropped.
        """
        with self._lock:
            for req_pool_idx, state in list(self._states.items()):
                if not state.history:
                    continue
                self._states[req_pool_idx] = _RequestState(
                    prev_hash_tail=state.prev_hash_tail,
                    k_req=state.k_req,
                    history=(),
                )

    def reset_request_to(self, *, req_pool_idx: int, k_req: int) -> None:
        """Roll a request's high-water mark back to ``k_req`` (spec reject path).

        Spec decoding rejects drafted tokens after the target verifies them;
        the rejected tokens' slots get freed and reused. The canary chain
        must rewind ``K_req`` and truncate the history so the next batch's
        verify entries don't read slots that were just returned to the
        allocator.
        """
        with self._lock:
            existing = self._states.get(req_pool_idx)
            if existing is None:
                return
            if k_req <= 0:
                self._states.pop(req_pool_idx, None)
                return
            new_k_req = min(existing.k_req, int(k_req))
            new_history = tuple(
                entry for entry in existing.history if entry.position < new_k_req
            )
            self._states[req_pool_idx] = _RequestState(
                prev_hash_tail=existing.prev_hash_tail,
                k_req=new_k_req,
                history=new_history,
            )

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

        The batch the kernel sees is laid out as
        ``[verify_entries..., write_entries...]``:

        - **verify entries**: for every request with prior committed writes,
          ONE verify entry per historical position ``[0, k_req)`` is emitted.
          Each verify entry binds a ``slot_idx`` (the physical slot the
          historical write landed in) to the expected
          ``(req_id, token_id, position, prev_hash)`` triple, plus the
          ``verify_seq_position`` for README §3 (b) — that field is just the
          historical ``position`` index itself, NOT derived from any
          ``req_to_token_pool`` lookup. So a map perturbation that redirects
          a slot pointer cannot fake-match the position-monotonic check.
        - **write entries**: ``sum(req_token_counts)`` slots laid out in the
          caller's flattened ``input_ids`` / ``out_cache_loc`` order.

        The verify range grows with k_req — covering the entire prefix means
        a perturbation anywhere along the chain (not just the last slot) is
        observable on the next forward.
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

                # Bound the per-forward verify cost: full [0, K_req) is
                # quadratic over a long req's lifetime. Verify the most
                # recent ``max_verify_per_req_per_forward`` entries instead.
                # 0 = unbounded.
                cap = self._config.max_verify_per_req_per_forward
                verify_window = state.history if cap <= 0 else state.history[-cap:]
                for entry in verify_window:
                    verify_req_ids.append(req_pool_idx)
                    verify_token_ids.append(entry.token_id)
                    verify_positions.append(entry.position)
                    verify_prev_hashes.append(to_signed_int64(entry.prev_hash_at_write))
                    verify_seq_positions.append(entry.position)
                    verify_slot_indices.append(entry.slot_idx)

                prev_hash = state.prev_hash_tail
                new_entries: List[_HistoryEntry] = []
                for offset in range(count):
                    pos = start_pos + offset
                    token_id = tokens[offset]
                    slot_idx = write_slots[offset]
                    write_req_ids.append(req_pool_idx)
                    write_token_ids.append(token_id)
                    write_positions.append(pos)
                    write_prev_hashes.append(to_signed_int64(prev_hash))
                    new_entries.append(
                        _HistoryEntry(
                            token_id=token_id,
                            position=pos,
                            slot_idx=slot_idx,
                            prev_hash_at_write=prev_hash,
                        )
                    )
                    prev_hash = mix_step(prev_hash, token_id, pos)

                if count > 0:
                    combined_history = state.history + tuple(new_entries)
                    # Keep at most max_verify_per_req_per_forward entries —
                    # older positions become unverifiable but free host memory
                    # and avoid unbounded growth. 0 = keep all history.
                    if cap > 0 and len(combined_history) > cap:
                        combined_history = combined_history[-cap:]
                    next_state[req_pool_idx] = _RequestState(
                        prev_hash_tail=prev_hash,
                        k_req=max(state.k_req, start_pos + count),
                        history=combined_history,
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

    def reset_violation_state(self) -> None:
        """Zero out is_errored / first_violation / ring buffer state.

        Called from the LOG-mode violation handler after the first-violation
        row has been pulled to host. Without this reset the GPU-side
        ``is_errored`` flag stays at 1 forever, ``first_violation_set``
        latches the first row permanently (new violations are silently
        masked), and ``violation_ring_valid`` fills up so subsequent CAS
        attempts all fail (= permanent ring deadlock after capacity rows).

        Counters (slot/kernel run counters) are intentionally NOT reset:
        the §5 health monitor uses their monotonic growth to detect "canary
        stopped running".
        """
        self.is_errored.zero_()
        self.first_violation_set.zero_()
        self.first_violation.zero_()
        self.violation_ring_valid.zero_()
        self.violation_write_index.zero_()


@dataclass(slots=True)
class CanaryLaunchBuffers:
    """Fixed-address per-launch tensors for cuda-graph-safe kernel launches.

    These tensors are allocated once at runner-init time and their addresses
    are baked into any captured cuda graph. Each forward fills the prefix
    ``[0, capacity)`` via in-place ``copy_``; rows beyond the active slot
    count are kept with ``verify_mask = -1`` so the kernel skips them.

    The kernel always launches with ``num_slots = capacity`` — replay path
    sees the SAME tensor sizes, same pointers, same launch grid as capture.
    """

    capacity: int
    slot_indices: torch.Tensor
    expected_req_ids: torch.Tensor
    expected_token_ids: torch.Tensor
    expected_positions: torch.Tensor
    expected_prev_hashes: torch.Tensor
    verify_mask: torch.Tensor
    verify_seq_positions: torch.Tensor

    @classmethod
    def allocate(cls, *, device: torch.device, capacity: int) -> "CanaryLaunchBuffers":
        if capacity <= 0:
            raise RuntimeError(
                f"kv-canary: CanaryLaunchBuffers capacity must be positive, got {capacity}"
            )
        slot_indices = torch.zeros(capacity, dtype=torch.int64, device=device)
        expected_req_ids = torch.zeros(capacity, dtype=torch.int64, device=device)
        expected_token_ids = torch.zeros(capacity, dtype=torch.int64, device=device)
        expected_positions = torch.zeros(capacity, dtype=torch.int64, device=device)
        expected_prev_hashes = torch.zeros(capacity, dtype=torch.int64, device=device)
        # Default ``-1`` = skip-sentinel: an unfilled launch must be a no-op.
        verify_mask = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        verify_seq_positions = torch.full(
            (capacity,), -1, dtype=torch.int64, device=device
        )
        return cls(
            capacity=int(capacity),
            slot_indices=slot_indices,
            expected_req_ids=expected_req_ids,
            expected_token_ids=expected_token_ids,
            expected_positions=expected_positions,
            expected_prev_hashes=expected_prev_hashes,
            verify_mask=verify_mask,
            verify_seq_positions=verify_seq_positions,
        )

    def fill_from_plan(self, plan: "BatchPlan") -> int:
        """Copy a host-side ``BatchPlan`` into the fixed GPU tensors in place.

        Returns the number of active slots (verify + write) actually copied.
        The prefix ``[0, total)`` carries real data; ``[total, capacity)`` is
        reset to the skip-sentinel so prior content from a larger batch
        cannot leak.

        Padding semantics: ``verify_mask = -1`` makes the kernel exit before
        any I/O; ``verify_seq_positions = -1`` is the existing "no position
        check" sentinel and is kept consistent. Other expected_* fields'
        residual values don't matter once ``verify_mask`` is -1.

        When ``num_verify + num_write`` exceeds capacity (e.g. an eager
        prefill batch larger than what the install-time sizing anticipated),
        the verify entries are TRUNCATED to whatever fits: writes are
        prioritized because they advance the chain hash; dropping writes
        would force every subsequent forward to see a broken chain.
        Truncating verifies merely reduces this forward's verify coverage —
        the next forward's verify entries re-cover the same history positions
        from the host-side ``history`` tuple.
        """
        total = plan.num_verify + plan.num_write
        if plan.num_write > self.capacity:
            raise RuntimeError(
                f"kv-canary: write slot count {plan.num_write} exceeds launch "
                f"capacity {self.capacity}. The fixed buffer is sized for "
                "cuda_graph_max_bs * num_tokens_per_bs writes; this eager "
                "forward exceeds that. Raise the canary launch capacity or "
                "disable the canary for this deployment."
            )
        if total > self.capacity:
            # Prefer writes over verifies (see docstring). Truncate the
            # verify section from the FRONT so the most recent history
            # entries (the right end of the verify list, since plan_batch
            # appends in order) survive.
            max_verify = max(0, self.capacity - plan.num_write)
            drop_verify = plan.num_verify - max_verify
            verify_slot_indices = plan.verify_slot_indices[drop_verify:]
            expected_req_ids = (
                plan.expected_req_ids[drop_verify : plan.num_verify]
                + plan.expected_req_ids[plan.num_verify :]
            )
            expected_token_ids = (
                plan.expected_token_ids[drop_verify : plan.num_verify]
                + plan.expected_token_ids[plan.num_verify :]
            )
            expected_positions = (
                plan.expected_positions[drop_verify : plan.num_verify]
                + plan.expected_positions[plan.num_verify :]
            )
            expected_prev_hashes = (
                plan.expected_prev_hashes[drop_verify : plan.num_verify]
                + plan.expected_prev_hashes[plan.num_verify :]
            )
            verify_mask = (
                plan.verify_mask[drop_verify : plan.num_verify]
                + plan.verify_mask[plan.num_verify :]
            )
            verify_seq_positions = (
                plan.verify_seq_positions[drop_verify : plan.num_verify]
                + plan.verify_seq_positions[plan.num_verify :]
            )
            write_slot_indices = plan.write_slot_indices
            total = max_verify + plan.num_write
        else:
            verify_slot_indices = plan.verify_slot_indices
            expected_req_ids = plan.expected_req_ids
            expected_token_ids = plan.expected_token_ids
            expected_positions = plan.expected_positions
            expected_prev_hashes = plan.expected_prev_hashes
            verify_mask = plan.verify_mask
            verify_seq_positions = plan.verify_seq_positions
            write_slot_indices = plan.write_slot_indices

        device = self.slot_indices.device
        slot_src = torch.tensor(
            verify_slot_indices + write_slot_indices,
            dtype=torch.int64,
            device=device,
        )
        req_src = torch.tensor(expected_req_ids, dtype=torch.int64, device=device)
        token_src = torch.tensor(expected_token_ids, dtype=torch.int64, device=device)
        pos_src = torch.tensor(expected_positions, dtype=torch.int64, device=device)
        hash_src = torch.tensor(expected_prev_hashes, dtype=torch.int64, device=device)
        mask_src = torch.tensor(verify_mask, dtype=torch.int32, device=device)
        seq_src = torch.tensor(verify_seq_positions, dtype=torch.int64, device=device)

        self.slot_indices[:total].copy_(slot_src)
        self.expected_req_ids[:total].copy_(req_src)
        self.expected_token_ids[:total].copy_(token_src)
        self.expected_positions[:total].copy_(pos_src)
        self.expected_prev_hashes[:total].copy_(hash_src)
        self.verify_mask[:total].copy_(mask_src)
        self.verify_seq_positions[:total].copy_(seq_src)
        if total < self.capacity:
            self.verify_mask[total:].fill_(-1)
            self.verify_seq_positions[total:].fill_(-1)
        return total

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import VIOLATION_FIELDS
from sglang.srt.kv_cache_canary.config import CanaryConfig

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchPlan:
    """Per-forward layout the canary kernel consumes.

    All per-req information is derived from the sglang ``ForwardBatch`` at
    plan time — there is no host-side per-request state. The host emits raw
    ``(req_id, token_id, position, slot_idx)`` arrays plus per-write-req
    chain-seed pointers; the kernel walks the splitmix64 chain itself.

    Layout:

    - **Per-verify-entry** (length ``num_verify``):
      ``verify_slot_indices`` / ``verify_positions`` / ``verify_req_ids`` /
      ``verify_prev_slot_indices``. ``verify_prev_slot_indices[i] == -1``
      means "this is position 0 of its req; expected prev_hash = kSeed".
      Otherwise it's the slot index for position ``verify_positions[i] - 1``
      of the same req.
    - **Per-write-entry** (length ``num_write``):
      ``write_slot_indices`` / ``write_token_ids`` / ``write_positions`` /
      ``write_req_ids``.
    - **Per-write-req** (length ``num_write_reqs``):
      ``write_req_seed_slot_indices`` (-1 = K_req_old == 0 -> kSeed),
      ``write_req_entry_starts`` (offset into per-write-entry arrays),
      ``write_req_entry_counts``.

    The kernel grid spans ``num_verify + num_write_reqs`` threads — one
    verify thread per entry plus one driver thread per write-req that walks
    its chain sequentially.
    """

    verify_req_ids: List[int]
    verify_positions: List[int]
    verify_slot_indices: List[int]
    verify_prev_slot_indices: List[int]

    write_req_ids: List[int]
    write_token_ids: List[int]
    write_positions: List[int]
    write_slot_indices: List[int]

    write_req_seed_slot_indices: List[int]
    write_req_entry_starts: List[int]
    write_req_entry_counts: List[int]

    num_verify: int
    num_write: int
    num_write_reqs: int


def plan_batch_from_forward_batch(
    *,
    forward_batch: "ForwardBatch",
    config: CanaryConfig,
) -> Optional[BatchPlan]:
    """Translate a ``ForwardBatch`` into per-slot kernel expectations.

    Every per-req field is read fresh from ``forward_batch`` — no canary
    host-side state is maintained. The chain hash is computed by the kernel
    on device using slot[i-1]'s stored ``(prev_hash, token, position)``.

    K_req (already-written token count per req) is recovered as
    ``extend_prefix_lens`` (chunked prefill) or ``seq_lens - 1`` (decode /
    target_verify). The verify range is ``[0, K_req)``, capped by
    ``config.max_verify_per_req_per_forward``. Slot indices for the verify
    range are pulled from ``forward_batch.req_to_token_pool.req_to_token``
    (the canary trusts this map; cross-checks come from the kernel side
    via req_id and position fields stored in the slot itself).

    Returns ``None`` when the plan would be empty (no out_cache_loc, unknown
    forward mode, etc.).
    """
    if forward_batch.out_cache_loc is None or forward_batch.out_cache_loc.numel() == 0:
        return None

    forward_mode = forward_batch.forward_mode
    if forward_mode is None:
        return None

    req_pool_indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    input_ids_list: List[int] = forward_batch.input_ids.detach().cpu().tolist()
    out_cache_loc_list: List[int] = forward_batch.out_cache_loc.detach().cpu().tolist()
    positions_list: Optional[List[int]] = (
        forward_batch.positions.detach().cpu().tolist()
        if forward_batch.positions is not None
        else None
    )

    is_extend = forward_mode.is_extend() or forward_mode.is_mixed()
    if is_extend:
        if (
            forward_batch.extend_seq_lens is None
            or forward_batch.extend_prefix_lens is None
        ):
            return None
        seq_lens = forward_batch.extend_seq_lens.detach().cpu().tolist()
        prefix_lens = forward_batch.extend_prefix_lens.detach().cpu().tolist()
    elif forward_mode.is_decode() or forward_mode.is_target_verify():
        seq_lens = [1] * len(req_pool_indices)
        full_seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
        prefix_lens = [int(s) - 1 for s in full_seq_lens]
    else:
        return None

    num_real_tokens = _num_real_tokens(forward_batch, len(input_ids_list))
    if sum(seq_lens) != num_real_tokens:
        return None
    if len(out_cache_loc_list) != num_real_tokens:
        return None

    req_to_token_pool = forward_batch.req_to_token_pool
    if req_to_token_pool is None:
        return None
    cap = int(config.max_verify_per_req_per_forward)

    return _build_plan(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        input_ids_list=input_ids_list,
        out_cache_loc_list=out_cache_loc_list,
        positions_list=positions_list,
        req_to_token_table=req_to_token_pool.req_to_token,
        verify_cap=cap,
    )


def _build_plan(
    *,
    req_pool_indices: List[int],
    seq_lens: List[int],
    prefix_lens: List[int],
    input_ids_list: List[int],
    out_cache_loc_list: List[int],
    positions_list: Optional[List[int]],
    req_to_token_table: torch.Tensor,
    verify_cap: int,
) -> Optional[BatchPlan]:
    verify_req_ids: List[int] = []
    verify_positions: List[int] = []
    verify_slot_indices: List[int] = []
    verify_prev_slot_indices: List[int] = []

    write_req_ids: List[int] = []
    write_token_ids: List[int] = []
    write_positions: List[int] = []
    write_slot_indices: List[int] = []

    write_req_seed_slot_indices: List[int] = []
    write_req_entry_starts: List[int] = []
    write_req_entry_counts: List[int] = []

    cursor = 0
    for req_pool_idx, n, k_req in zip(req_pool_indices, seq_lens, prefix_lens):
        next_cursor = cursor + n
        req_pool_idx_int = int(req_pool_idx)
        # Padding row in ReqToTokenPool lives at index 0 (cuda-graph padded
        # batches set padding rows' req_pool_indices to 0). Skipping avoids
        # writing synthetic data into the padding slot's shadow.
        if req_pool_idx_int == 0:
            cursor = next_cursor
            continue

        k_req_int = int(k_req)
        if k_req_int > 0:
            slot_indices_for_verify = _pull_verify_slot_indices(
                req_to_token_table=req_to_token_table,
                req_pool_idx=req_pool_idx_int,
                k_req=k_req_int,
                cap=verify_cap,
            )
            window_start = k_req_int - len(slot_indices_for_verify)
            for j, slot_idx in enumerate(slot_indices_for_verify):
                pos = window_start + j
                verify_req_ids.append(req_pool_idx_int)
                verify_positions.append(pos)
                verify_slot_indices.append(int(slot_idx))
                if pos == 0:
                    verify_prev_slot_indices.append(-1)
                elif j > 0:
                    verify_prev_slot_indices.append(
                        int(slot_indices_for_verify[j - 1])
                    )
                else:
                    # Truncated window head: prev slot lives at column
                    # (pos - 1) of the same req in req_to_token.
                    prev_slot = int(req_to_token_table[req_pool_idx_int, pos - 1])
                    verify_prev_slot_indices.append(prev_slot)

        if n > 0:
            seed_slot = -1
            if k_req_int > 0:
                seed_slot = int(
                    req_to_token_table[req_pool_idx_int, k_req_int - 1]
                )
            entry_start = len(write_slot_indices)
            write_req_seed_slot_indices.append(seed_slot)
            write_req_entry_starts.append(entry_start)
            write_req_entry_counts.append(n)

            for offset in range(n):
                pos = k_req_int + offset
                token_id = input_ids_list[cursor + offset]
                slot_idx = out_cache_loc_list[cursor + offset]
                write_req_ids.append(req_pool_idx_int)
                write_token_ids.append(int(token_id))
                # ForwardBatch.positions carries the canonical position for
                # each new token. Fall back to the prefix+offset derivation
                # when the tensor is unavailable (e.g. some test paths).
                if positions_list is not None:
                    write_positions.append(int(positions_list[cursor + offset]))
                else:
                    write_positions.append(pos)
                write_slot_indices.append(int(slot_idx))

        cursor = next_cursor

    num_verify = len(verify_req_ids)
    num_write = len(write_req_ids)
    num_write_reqs = len(write_req_seed_slot_indices)
    if num_verify == 0 and num_write == 0:
        return None

    return BatchPlan(
        verify_req_ids=verify_req_ids,
        verify_positions=verify_positions,
        verify_slot_indices=verify_slot_indices,
        verify_prev_slot_indices=verify_prev_slot_indices,
        write_req_ids=write_req_ids,
        write_token_ids=write_token_ids,
        write_positions=write_positions,
        write_slot_indices=write_slot_indices,
        write_req_seed_slot_indices=write_req_seed_slot_indices,
        write_req_entry_starts=write_req_entry_starts,
        write_req_entry_counts=write_req_entry_counts,
        num_verify=num_verify,
        num_write=num_write,
        num_write_reqs=num_write_reqs,
    )


def _pull_verify_slot_indices(
    *,
    req_to_token_table: torch.Tensor,
    req_pool_idx: int,
    k_req: int,
    cap: int,
) -> List[int]:
    """Return slot indices for the verify range of one req.

    Bounds the per-forward verify cost: the full ``[0, K_req)`` range grows
    linearly with the req's lifetime. When capped, the TAIL ``cap``
    positions are verified (most likely to surface a fresh-write bug); the
    older positions become unverifiable this forward but the next forward
    re-walks the same window.
    """
    if cap > 0 and k_req > cap:
        window_start = k_req - cap
    else:
        window_start = 0
    row = req_to_token_table[req_pool_idx, window_start:k_req]
    return [int(x) for x in row.detach().cpu().tolist()]


def _num_real_tokens(forward_batch: "ForwardBatch", total_input_len: int) -> int:
    """Strip cuda-graph padding from token-aligned arrays.

    ``num_token_non_padded_cpu`` (when present) tells us how many leading
    tokens of ``input_ids`` / ``out_cache_loc`` are real; the remainder is
    cuda-graph tail padding.
    """
    if hasattr(forward_batch, "num_token_non_padded_cpu"):
        value = forward_batch.num_token_non_padded_cpu
        if value is not None:
            try:
                return int(value)
            except TypeError:
                pass
    return total_input_len


VIOLATION_KIND_HEAD_K: str = "head_k"
VIOLATION_KIND_HEAD_V: str = "head_v"
VIOLATION_KIND_TAIL_K: str = "tail_k"
VIOLATION_KIND_TAIL_V: str = "tail_v"
VIOLATION_KINDS: Tuple[str, str, str, str] = (
    VIOLATION_KIND_HEAD_K,
    VIOLATION_KIND_HEAD_V,
    VIOLATION_KIND_TAIL_K,
    VIOLATION_KIND_TAIL_V,
)


@dataclass(slots=True)
class CanaryViolationSlot:
    """One independent violation-state set (ring + first_violation + is_errored).

    Four instances live on ``CanaryDeviceState`` — one per
    (head|tail) x (K-half|V-half) kernel launch. Keeping them disjoint stops
    the K-half launch from CAS-latching ``first_violation_set`` and
    silently masking every V-half mismatch (and vice versa), so V-half page
    table bugs become visible instead of being shadowed by an earlier K-half
    mismatch in the same forward.
    """

    violation_ring: torch.Tensor
    violation_ring_valid: torch.Tensor
    violation_write_index: torch.Tensor
    first_violation: torch.Tensor
    first_violation_set: torch.Tensor
    is_errored: torch.Tensor

    @classmethod
    def allocate(
        cls, *, device: torch.device, ring_capacity: int
    ) -> "CanaryViolationSlot":
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
        )

    def reset(self) -> None:
        self.is_errored.zero_()
        self.first_violation_set.zero_()
        self.first_violation.zero_()
        self.violation_ring_valid.zero_()
        self.violation_write_index.zero_()


@dataclass(slots=True)
class CanaryDeviceState:
    """GPU-resident state shared across head/tail kernel invocations.

    Violation state is split four ways — one ``CanaryViolationSlot`` per
    ``(head|tail) x (K-half|V-half)`` kernel launch — so K-half and V-half
    detection paths never share a ``first_violation`` latch or an
    ``is_errored`` flag. Counters stay split only by head/tail (K/V launches
    of the same kernel kind share write/verify work over the same slot
    indices, so per-kernel-kind counters are enough for health monitoring).
    """

    violation_slots: Dict[str, CanaryViolationSlot]
    slot_run_counter_head: torch.Tensor
    slot_run_counter_tail: torch.Tensor
    kernel_run_counter_head: torch.Tensor
    kernel_run_counter_tail: torch.Tensor

    @classmethod
    def allocate(
        cls, *, device: torch.device, ring_capacity: int
    ) -> "CanaryDeviceState":
        return cls(
            violation_slots={
                kind: CanaryViolationSlot.allocate(
                    device=device, ring_capacity=ring_capacity
                )
                for kind in VIOLATION_KINDS
            },
            slot_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            slot_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
        )

    def get_violation_slot(self, kind: str) -> CanaryViolationSlot:
        return self.violation_slots[kind]

    def reset_violation_state(self) -> None:
        """Zero out every per-kind violation slot.

        Called from the LOG-mode violation handler after each kind's
        first-violation row has been pulled to host. Without this reset the
        GPU-side ``is_errored`` flag stays at 1 forever,
        ``first_violation_set`` latches the first row permanently (new
        violations are silently masked), and ``violation_ring_valid`` fills
        up so subsequent CAS attempts all fail (= permanent ring deadlock
        after capacity rows).

        Counters (slot/kernel run counters) are intentionally NOT reset:
        the §5 health monitor uses their monotonic growth to detect "canary
        stopped running".
        """
        for slot in self.violation_slots.values():
            slot.reset()


@dataclass(slots=True)
class CanaryLaunchBuffers:
    """Fixed-address per-launch tensors for cuda-graph-safe kernel launches.

    Holds three tile sets:

    1. **Per-verify-entry** (capacity ``verify_capacity``):
       ``verify_slot_indices`` / ``verify_positions`` / ``verify_req_ids`` /
       ``verify_prev_slot_indices`` / ``verify_active_mask``.
    2. **Per-write-entry** (capacity ``write_capacity``):
       ``write_slot_indices`` / ``write_token_ids`` / ``write_positions`` /
       ``write_req_ids``.
    3. **Per-write-req** (capacity ``write_req_capacity``):
       ``write_req_seed_slot_indices`` / ``write_req_entry_starts`` /
       ``write_req_entry_counts`` / ``write_req_active_mask``.

    Per-write-entry rows are pure data driven by the write-req chains; the
    grid is sized by ``verify_capacity + write_req_capacity``.
    """

    verify_capacity: int
    write_capacity: int
    write_req_capacity: int
    verify_slot_indices: torch.Tensor
    verify_positions: torch.Tensor
    verify_req_ids: torch.Tensor
    verify_prev_slot_indices: torch.Tensor
    verify_active_mask: torch.Tensor
    write_slot_indices: torch.Tensor
    write_token_ids: torch.Tensor
    write_positions: torch.Tensor
    write_req_ids: torch.Tensor
    write_req_seed_slot_indices: torch.Tensor
    write_req_entry_starts: torch.Tensor
    write_req_entry_counts: torch.Tensor
    write_req_active_mask: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        device: torch.device,
        verify_capacity: int,
        write_capacity: int,
        write_req_capacity: int,
    ) -> "CanaryLaunchBuffers":
        for name, cap in [
            ("verify_capacity", verify_capacity),
            ("write_capacity", write_capacity),
            ("write_req_capacity", write_req_capacity),
        ]:
            if cap <= 0:
                raise RuntimeError(
                    f"kv-canary: CanaryLaunchBuffers {name} must be positive, got {cap}"
                )

        def zeros_i64(n: int) -> torch.Tensor:
            return torch.zeros(n, dtype=torch.int64, device=device)

        def zeros_i32(n: int) -> torch.Tensor:
            return torch.zeros(n, dtype=torch.int32, device=device)

        return cls(
            verify_capacity=int(verify_capacity),
            write_capacity=int(write_capacity),
            write_req_capacity=int(write_req_capacity),
            verify_slot_indices=zeros_i64(verify_capacity),
            verify_positions=zeros_i64(verify_capacity),
            verify_req_ids=zeros_i64(verify_capacity),
            verify_prev_slot_indices=zeros_i64(verify_capacity),
            verify_active_mask=zeros_i32(verify_capacity),
            write_slot_indices=zeros_i64(write_capacity),
            write_token_ids=zeros_i64(write_capacity),
            write_positions=zeros_i64(write_capacity),
            write_req_ids=zeros_i64(write_capacity),
            write_req_seed_slot_indices=zeros_i64(write_req_capacity),
            write_req_entry_starts=zeros_i64(write_req_capacity),
            write_req_entry_counts=zeros_i64(write_req_capacity),
            write_req_active_mask=zeros_i32(write_req_capacity),
        )

    def fill_from_plan(self, plan: "BatchPlan") -> Tuple[int, int]:
        """Copy a host-side ``BatchPlan`` into the fixed GPU tensors in place.

        Returns ``(num_active_verify, num_active_write_reqs)``. Rows past
        the active count are reset to inactive (mask = 0) so the kernel
        skips them. The write-entry tile is also reset past
        ``plan.num_write`` so a write-req driver that reads
        ``write_*[entry_start + j]`` cannot pick up stale data.

        When the verify plan exceeds capacity, the TAIL is kept and the
        head is truncated: writes are prioritised (they advance the chain);
        the next forward's verify entries re-cover the older history.
        """
        if plan.num_write > self.write_capacity:
            raise RuntimeError(
                f"kv-canary: write entry count {plan.num_write} exceeds "
                f"write_capacity {self.write_capacity}. Raise canary launch "
                "capacity or disable the canary for this deployment."
            )
        if plan.num_write_reqs > self.write_req_capacity:
            raise RuntimeError(
                f"kv-canary: write req count {plan.num_write_reqs} exceeds "
                f"write_req_capacity {self.write_req_capacity}."
            )

        num_active_verify = min(plan.num_verify, self.verify_capacity)
        drop = plan.num_verify - num_active_verify
        v = slice(drop, plan.num_verify)

        device = self.verify_slot_indices.device

        def to_i64(values: List[int], sl: slice) -> torch.Tensor:
            return torch.tensor(values[sl], dtype=torch.int64, device=device)

        if num_active_verify > 0:
            self.verify_slot_indices[:num_active_verify].copy_(
                to_i64(plan.verify_slot_indices, v)
            )
            self.verify_positions[:num_active_verify].copy_(
                to_i64(plan.verify_positions, v)
            )
            self.verify_req_ids[:num_active_verify].copy_(
                to_i64(plan.verify_req_ids, v)
            )
            self.verify_prev_slot_indices[:num_active_verify].copy_(
                to_i64(plan.verify_prev_slot_indices, v)
            )
            self.verify_active_mask[:num_active_verify].fill_(1)
        if num_active_verify < self.verify_capacity:
            self.verify_active_mask[num_active_verify:].fill_(0)

        nw = plan.num_write
        if nw > 0:
            self.write_slot_indices[:nw].copy_(
                to_i64(plan.write_slot_indices, slice(0, nw))
            )
            self.write_token_ids[:nw].copy_(
                to_i64(plan.write_token_ids, slice(0, nw))
            )
            self.write_positions[:nw].copy_(
                to_i64(plan.write_positions, slice(0, nw))
            )
            self.write_req_ids[:nw].copy_(
                to_i64(plan.write_req_ids, slice(0, nw))
            )
        if nw < self.write_capacity:
            self.write_slot_indices[nw:].zero_()
            self.write_token_ids[nw:].zero_()
            self.write_positions[nw:].zero_()
            self.write_req_ids[nw:].zero_()

        nwr = plan.num_write_reqs
        if nwr > 0:
            self.write_req_seed_slot_indices[:nwr].copy_(
                to_i64(plan.write_req_seed_slot_indices, slice(0, nwr))
            )
            self.write_req_entry_starts[:nwr].copy_(
                to_i64(plan.write_req_entry_starts, slice(0, nwr))
            )
            self.write_req_entry_counts[:nwr].copy_(
                to_i64(plan.write_req_entry_counts, slice(0, nwr))
            )
            self.write_req_active_mask[:nwr].fill_(1)
        if nwr < self.write_req_capacity:
            self.write_req_active_mask[nwr:].fill_(0)

        return num_active_verify, nwr

    def reset_to_skip_sentinel(self) -> None:
        """Reset all active masks so the recorded kernel becomes a no-op.

        Used at capture time and at replay-time when no valid plan exists.
        """
        self.verify_active_mask.zero_()
        self.write_req_active_mask.zero_()

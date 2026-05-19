"""Python reference for the canary plan kernel + shared GPU dataclass.

Defines :class:`BatchPlanGpu`, the device-side fixed-shape plan-tensor
bundle that connects the plan kernel to :func:`canary_step
<sglang.jit_kernel.kv_cache_canary.canary_step>`, and the Python
reference implementation of the plan accumulator (:class:`BatchPlan` +
:func:`plan_batch_from_forward_batch`).

The plan kernel produces one :class:`BatchPlanGpu` per forward;
``canary_step`` consumes it as a single dataclass argument instead of
unpacking ~13 individual tensors at every call site. The reference
impl in this module is consumed in two ways:

1. As the executable baseline for differential testing the Triton plan
   kernel (:mod:`sglang.jit_kernel.kv_cache_canary_plan`).
2. As the current production hot-path planner — host-side Python loop
   over reqs that fills a ``BatchPlan`` then copies it into a
   ``BatchPlanGpu`` via :func:`fill_batch_plan_gpu_from_plan`. The
   Triton kernel will replace this hot-path usage once wired in.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_EXPECTED_SKIP_SENTINEL as _SKIP_SENTINEL,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchPlanGpu:
    """Device-side fixed-shape plan tensors connecting plan kernel → canary_step.

    Produced by :func:`sglang.jit_kernel.kv_cache_canary_plan.plan_batch_from_forward_batch`
    once per forward and consumed verbatim by
    :func:`sglang.jit_kernel.kv_cache_canary.canary_step`.

    Plan layout — three parallel tile sets describing the verify + write work for one canary
    launch:

    - **Per-verify-entry** (active length ``verify_num_valid[0]``; capacity =
      ``verify_slot_indices.shape[0]``): ``(slot_idx, position, prev_slot_idx)`` for each slot
      the kernel verifies. ``prev_slot_idx == -1`` flags a chain-seed entry (position 0 of a
      req, or the head of an SWA window — anchor on
      :data:`sglang.jit_kernel.kv_cache_canary.CANARY_CHAIN_ANCHOR` instead of a predecessor
      slot).
    - **Per-write-entry** (active length implied by per-write-req ``entry_start`` /
      ``entry_count``; capacity = ``write_slot_indices.shape[0]``): ``(slot_idx, token_id,
      position)`` for each slot the kernel fingerprints, flattened across reqs in the order
      they appear in the source ``ForwardBatch.req_pool_indices``. ``(expected_write_token_ids,
      expected_write_positions)`` are the pseudo-mode oracle predictions per write entry; both
      filled with ``-1`` skip-sentinel in real-mode (the kernel pays no per-entry cost on the
      skip path).
    - **Per-write-req** (active length ``write_req_num_valid[0]``; capacity =
      ``write_req_seed_slot_indices.shape[0]``): ``(seed_slot_idx, entry_start, entry_count)``
      per req that has at least one write entry. ``seed_slot_idx == -1`` flags
      ``K_req_old == 0`` (chain anchors on
      :data:`sglang.jit_kernel.kv_cache_canary.CANARY_CHAIN_ANCHOR`). ``entry_start`` is the
      exclusive prefix-sum offset into the per-write-entry arrays.

    Every per-entry tensor is sized to a **cuda-graph-captured fixed maximum**; the active
    prefix is reported via the ``verify_num_valid`` / ``write_req_num_valid`` scalars. Padding
    tail entries (indices ``>= num_valid``) are unspecified — ``canary_step`` early-exits via
    ``tid >= num_valid[0]`` before reading them. The per-write-entry tile has no separate
    ``write_num_valid`` because writes are gated through the per-write-req tile via
    ``entry_start`` / ``entry_count``.

    Lifecycle:

    - Allocate once at runner init via :meth:`allocate`; the buffer addresses are baked into
      cuda-graph capture.
    - Refilled in-place every forward by
      :func:`plan_batch_from_forward_batch <sglang.jit_kernel.kv_cache_canary_plan.plan_batch_from_forward_batch>`;
      never reallocated.
    - Read in-place by ``canary_step`` (head + tail launches both consume the same instance,
      and the K-half / V-half launches share the same plan as well).
    - Reset to all-zero / skip-sentinel via :meth:`reset_to_skip_sentinel` at capture-time
      teardown / re-init.

    All fields are ``int32`` so kernel atomic / scan operations stay in int32 throughout;
    callers must not change dtypes without updating both plan kernel and ``canary_step``.

    Fields:
        verify_slot_indices:         ``int32 [verify_capacity]`` — canary buffer slot index of
                                     each verify entry. Already SWA-translated for the SWA
                                     group; see plan kernel docstring for the LUT-ordering rule.
        verify_positions:            ``int32 [verify_capacity]`` — expected sequence position
                                     for each verify entry.
        verify_prev_slot_indices:    ``int32 [verify_capacity]`` — slot index of the chain
                                     predecessor, or ``-1`` for chain-seed entries.
        write_slot_indices:          ``int32 [write_capacity]`` — canary buffer slot index of
                                     each write entry. Already SWA-translated.
        write_token_ids:             ``int32 [write_capacity]`` — token id to fingerprint into
                                     each write slot.
        write_positions:             ``int32 [write_capacity]`` — sequence position to
                                     fingerprint into each write slot.
        expected_write_token_ids:    ``int32 [write_capacity]`` — pseudo-mode oracle's expected
                                     token id; ``-1`` skip-sentinel in real-mode.
        expected_write_positions:    ``int32 [write_capacity]`` — pseudo-mode oracle's expected
                                     sequence position; ``-1`` skip-sentinel in real-mode.
        write_req_seed_slot_indices: ``int32 [write_req_capacity]`` — chain-seed slot per write
                                     req, or ``-1`` when the req has no prefix
                                     (``K_req_old == 0``).
        write_req_entry_starts:      ``int32 [write_req_capacity]`` — exclusive prefix-sum
                                     offset of each write req into the per-write-entry arrays.
        write_req_entry_counts:      ``int32 [write_req_capacity]`` — number of write entries
                                     owned by each write req.
        verify_num_valid:            ``int32 [1]`` — count of active per-verify-entry rows.
                                     ``canary_step`` skips threads with ``tid >=
                                     verify_num_valid[0]``.
        write_req_num_valid:         ``int32 [1]`` — count of active per-write-req rows.
                                     ``canary_step`` skips threads with ``req_tid >=
                                     write_req_num_valid[0]``.
    """

    verify_slot_indices: torch.Tensor
    verify_positions: torch.Tensor
    verify_prev_slot_indices: torch.Tensor

    write_slot_indices: torch.Tensor
    write_token_ids: torch.Tensor
    write_positions: torch.Tensor
    expected_write_token_ids: torch.Tensor
    expected_write_positions: torch.Tensor

    write_req_seed_slot_indices: torch.Tensor
    write_req_entry_starts: torch.Tensor
    write_req_entry_counts: torch.Tensor

    verify_num_valid: torch.Tensor
    write_req_num_valid: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchPlan:
    """Per-forward layout the canary kernel consumes.

    Three parallel tile sets — per-verify-entry, per-write-entry, and
    per-write-req — built from the live ``ForwardBatch``. The kernel
    grid spans ``num_verify + num_write_reqs`` threads (one verify
    thread per entry plus one driver thread per write-req).

    ``verify_prev_slot_indices[i] == -1`` flags position 0 of a req
    (chain seed = :data:`sglang.jit_kernel.kv_cache_canary.CANARY_CHAIN_ANCHOR`).
    ``write_req_seed_slot_indices[i] == -1`` flags ``K_req_old == 0`` for the same reason.
    """

    verify_positions: List[int]
    verify_slot_indices: List[int]
    verify_prev_slot_indices: List[int]

    write_token_ids: List[int]
    write_positions: List[int]
    write_slot_indices: List[int]

    write_req_seed_slot_indices: List[int]
    write_req_entry_starts: List[int]
    write_req_entry_counts: List[int]
    # Per-write-req ``req_pool_idx`` of the req that contributed each row.
    # Length == num_write_reqs. Host-only bookkeeping (the canary kernel
    # never reads it); the pseudo-mode oracle uses it to look up the
    # logical req id from a write-req row when emitting expected_*.
    write_req_pool_indices: List[int]

    num_verify: int
    num_write: int
    num_write_reqs: int

    # Pseudo-mode oracle predictions for the write entries. Set as a
    # paired non-None tuple in pseudo-mode; both None otherwise.
    expected_write_token_ids: Optional[List[int]] = None
    expected_write_positions: Optional[List[int]] = None

    def __post_init__(self) -> None:
        tokens = self.expected_write_token_ids
        positions = self.expected_write_positions
        if (tokens is None) != (positions is None):
            raise ValueError(
                "kv-canary: expected_write_token_ids and expected_write_positions "
                "must both be set or both be None"
            )
        if tokens is not None:
            if len(tokens) != self.num_write:
                raise ValueError(
                    f"kv-canary: expected_write_token_ids length {len(tokens)} "
                    f"!= num_write {self.num_write}"
                )
            if len(positions) != self.num_write:
                raise ValueError(
                    f"kv-canary: expected_write_positions length {len(positions)} "
                    f"!= num_write {self.num_write}"
                )

    @classmethod
    def empty(cls) -> "BatchPlan":
        return cls(
            verify_positions=[],
            verify_slot_indices=[],
            verify_prev_slot_indices=[],
            write_token_ids=[],
            write_positions=[],
            write_slot_indices=[],
            write_req_seed_slot_indices=[],
            write_req_entry_starts=[],
            write_req_entry_counts=[],
            write_req_pool_indices=[],
            num_verify=0,
            num_write=0,
            num_write_reqs=0,
        )


def plan_batch_from_forward_batch(
    *,
    forward_batch: "ForwardBatch",
    config: CanaryConfig,
    swa_index_lut: Optional[torch.Tensor] = None,
) -> Optional[BatchPlan]:
    """Translate a ``ForwardBatch`` into a :class:`BatchPlan`.

    Verify range is full ``[0, K_req)`` for non-SWA pools (every historical
    position re-verified each forward); SWA pools clip to the most recent
    ``swa_window_size`` slots because older positions in ``req_to_token``
    point at slots that have been evicted to other reqs.

    When ``swa_index_lut`` is supplied (SWA canary group on a pool with a
    distinct swa sub-pool slot index space) the plan's slot-index fields
    are remapped through ``lut[idx]`` before return; chain-head sentinel
    (-1) and skip-chain sentinel (-2) values pass through unchanged.

    Returns ``None`` for empty / unsupported batches (no out_cache_loc,
    unknown forward mode, missing extend lens).
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

    plan = _build_plan(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        input_ids_list=input_ids_list,
        out_cache_loc_list=out_cache_loc_list,
        positions_list=positions_list,
        req_to_token_table=req_to_token_pool.req_to_token,
        swa_window_size=config.swa_window_size,
    )
    if plan is None or swa_index_lut is None:
        return plan
    return _translate_plan_slot_indices(plan=plan, lut=swa_index_lut)


def _translate_plan_slot_indices(
    *,
    plan: BatchPlan,
    lut: torch.Tensor,
) -> BatchPlan:
    """Remap the four slot-index fields of ``plan`` through ``lut``.

    ``lut`` is the pool's ``full_to_swa_index_mapping``: a ``[size_full + 1]``
    int tensor whose final entry is ``-1`` (sentinel for "not in window /
    unmapped"). Sentinel values already present in the plan — ``-1``
    (chain-head, used by ``verify_prev_slot_indices`` and
    ``write_req_seed_slot_indices``) and ``SKIP_CHAIN_SENTINEL`` (-2,
    used by sweep) — pass through unchanged; only non-negative full-pool
    indices are translated.
    """
    cpu_lut = lut.detach().cpu().to(torch.int64)
    lut_len = int(cpu_lut.shape[0])

    def translate(values: List[int]) -> List[int]:
        if not values:
            return list(values)
        idx_tensor = torch.tensor(values, dtype=torch.int64)
        sentinel_mask = idx_tensor < 0
        # Clamp negatives to a safe in-range index (0) before gather; the
        # sentinel positions are restored below, so the value gathered for
        # them is discarded.
        safe_idx = torch.where(sentinel_mask, torch.zeros_like(idx_tensor), idx_tensor)
        if lut_len > 0:
            # Out-of-range positive indices fall back to the LUT's final
            # entry (the -1 sentinel), matching the sglang convention.
            oob_mask = safe_idx >= lut_len
            safe_idx = torch.where(
                oob_mask,
                torch.full_like(safe_idx, lut_len - 1),
                safe_idx,
            )
        translated = cpu_lut[safe_idx]
        result = torch.where(sentinel_mask, idx_tensor, translated)
        return [int(x) for x in result.tolist()]

    return dataclasses.replace(
        plan,
        verify_slot_indices=translate(plan.verify_slot_indices),
        verify_prev_slot_indices=translate(plan.verify_prev_slot_indices),
        write_slot_indices=translate(plan.write_slot_indices),
        write_req_seed_slot_indices=translate(plan.write_req_seed_slot_indices),
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
    swa_window_size: Optional[int],
) -> Optional[BatchPlan]:
    accumulator = _PlanAccumulator()

    cursor = 0
    for req_pool_idx, n, k_req in zip(req_pool_indices, seq_lens, prefix_lens):
        next_cursor = cursor + n
        req_pool_idx_int = int(req_pool_idx)
        # Padding row in ReqToTokenPool lives at index 0 (cuda-graph padded
        # batches set padding rows' req_pool_indices to 0). Skipping avoids
        # writing synthetic data into the padding slot's canary buffer.
        if req_pool_idx_int == 0:
            cursor = next_cursor
            continue

        k_req_int = int(k_req)
        if k_req_int > 0:
            _append_verify_entries(
                accumulator=accumulator,
                req_pool_idx=req_pool_idx_int,
                k_req=k_req_int,
                req_to_token_table=req_to_token_table,
                swa_window_size=swa_window_size,
            )

        if n > 0:
            _append_write_entries(
                accumulator=accumulator,
                req_pool_idx=req_pool_idx_int,
                k_req=k_req_int,
                n=n,
                cursor=cursor,
                input_ids_list=input_ids_list,
                out_cache_loc_list=out_cache_loc_list,
                positions_list=positions_list,
                req_to_token_table=req_to_token_table,
            )

        cursor = next_cursor

    return accumulator.into_plan()


def _append_verify_entries(
    *,
    accumulator: "_PlanAccumulator",
    req_pool_idx: int,
    k_req: int,
    req_to_token_table: torch.Tensor,
    swa_window_size: Optional[int],
) -> None:
    slot_indices_for_verify = _pull_verify_slot_indices(
        req_to_token_table=req_to_token_table,
        req_pool_idx=req_pool_idx,
        k_req=k_req,
        swa_window_size=swa_window_size,
    )
    window_start = k_req - len(slot_indices_for_verify)
    for j, slot_idx in enumerate(slot_indices_for_verify):
        pos = window_start + j
        accumulator.verify_positions.append(pos)
        accumulator.verify_slot_indices.append(int(slot_idx))
        if pos == 0:
            accumulator.verify_prev_slot_indices.append(-1)
        elif j > 0:
            accumulator.verify_prev_slot_indices.append(
                int(slot_indices_for_verify[j - 1])
            )
        else:
            # Window starts at pos > 0 (SWA truncation): prev slot lives at
            # column (pos - 1) of the same req in req_to_token. For the
            # full-prefix case (window_start == 0) j > 0 always holds when
            # pos > 0, so this branch is reached only on the SWA path.
            prev_slot = int(req_to_token_table[req_pool_idx, pos - 1])
            accumulator.verify_prev_slot_indices.append(prev_slot)


def _append_write_entries(
    *,
    accumulator: "_PlanAccumulator",
    req_pool_idx: int,
    k_req: int,
    n: int,
    cursor: int,
    input_ids_list: List[int],
    out_cache_loc_list: List[int],
    positions_list: Optional[List[int]],
    req_to_token_table: torch.Tensor,
) -> None:
    seed_slot = -1
    if k_req > 0:
        seed_slot = int(req_to_token_table[req_pool_idx, k_req - 1])
    entry_start = len(accumulator.write_slot_indices)
    accumulator.write_req_seed_slot_indices.append(seed_slot)
    accumulator.write_req_entry_starts.append(entry_start)
    accumulator.write_req_entry_counts.append(n)
    accumulator.write_req_pool_indices.append(req_pool_idx)

    for offset in range(n):
        pos = k_req + offset
        token_id = input_ids_list[cursor + offset]
        slot_idx = out_cache_loc_list[cursor + offset]
        accumulator.write_token_ids.append(int(token_id))
        # ForwardBatch.positions carries the canonical position for each
        # new token. Fall back to the prefix+offset derivation when the
        # tensor is unavailable (e.g. some test paths).
        if positions_list is not None:
            accumulator.write_positions.append(int(positions_list[cursor + offset]))
        else:
            accumulator.write_positions.append(pos)
        accumulator.write_slot_indices.append(int(slot_idx))


class _PlanAccumulator:
    """Mutable per-list buffer that ``_build_plan`` fills row by row."""

    def __init__(self) -> None:
        self.verify_positions: List[int] = []
        self.verify_slot_indices: List[int] = []
        self.verify_prev_slot_indices: List[int] = []

        self.write_token_ids: List[int] = []
        self.write_positions: List[int] = []
        self.write_slot_indices: List[int] = []

        self.write_req_seed_slot_indices: List[int] = []
        self.write_req_entry_starts: List[int] = []
        self.write_req_entry_counts: List[int] = []
        self.write_req_pool_indices: List[int] = []

    def into_plan(self) -> Optional[BatchPlan]:
        num_verify = len(self.verify_positions)
        num_write = len(self.write_token_ids)
        num_write_reqs = len(self.write_req_seed_slot_indices)
        if num_verify == 0 and num_write == 0:
            return None

        return BatchPlan(
            verify_positions=self.verify_positions,
            verify_slot_indices=self.verify_slot_indices,
            verify_prev_slot_indices=self.verify_prev_slot_indices,
            write_token_ids=self.write_token_ids,
            write_positions=self.write_positions,
            write_slot_indices=self.write_slot_indices,
            write_req_seed_slot_indices=self.write_req_seed_slot_indices,
            write_req_entry_starts=self.write_req_entry_starts,
            write_req_entry_counts=self.write_req_entry_counts,
            write_req_pool_indices=self.write_req_pool_indices,
            num_verify=num_verify,
            num_write=num_write,
            num_write_reqs=num_write_reqs,
        )


def _pull_verify_slot_indices(
    *,
    req_to_token_table: torch.Tensor,
    req_pool_idx: int,
    k_req: int,
    swa_window_size: Optional[int],
) -> List[int]:
    """Return slot indices for one req's verify range.

    Non-SWA (``swa_window_size is None``): full ``[0, K_req)`` window —
    every historical position is verified every forward (user
    requirement: a 10k-prefix decode step verifies all 10k positions).

    SWA (``swa_window_size > 0``): clipped to
    ``[max(0, K_req - swa_window_size), K_req)``. The SWA pool's
    ``req_to_token`` map only addresses the most recent
    ``swa_window_size`` slots; older positions point at slots that have
    been evicted / reused by other reqs, so reading them would trip a
    spurious position / hash mismatch.
    """
    if swa_window_size is not None and k_req > swa_window_size:
        window_start = k_req - swa_window_size
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


def allocate_batch_plan_gpu(
    *,
    device: torch.device,
    verify_capacity: int,
    write_capacity: int,
    write_req_capacity: int,
) -> BatchPlanGpu:
    """Allocate the fixed-address per-launch :class:`BatchPlanGpu` tensors.

    Three tile sets:

    1. **Per-verify-entry** (capacity ``verify_capacity``):
       ``verify_slot_indices`` / ``verify_positions`` /
       ``verify_prev_slot_indices`` / ``verify_num_valid``.
    2. **Per-write-entry** (capacity ``write_capacity``):
       ``write_slot_indices`` / ``write_token_ids`` / ``write_positions`` /
       ``expected_write_token_ids`` / ``expected_write_positions``.
    3. **Per-write-req** (capacity ``write_req_capacity``):
       ``write_req_seed_slot_indices`` / ``write_req_entry_starts`` /
       ``write_req_entry_counts`` / ``write_req_num_valid``.

    Per-write-entry rows are pure data driven by the write-req chains; the
    grid is sized by ``verify_capacity + write_req_capacity``.
    """
    for name, cap in [
        ("verify_capacity", verify_capacity),
        ("write_capacity", write_capacity),
        ("write_req_capacity", write_req_capacity),
    ]:
        if cap <= 0:
            raise RuntimeError(
                f"kv-canary: BatchPlanGpu {name} must be positive, got {cap}"
            )

    def zeros_i64(n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=torch.int64, device=device)

    def zeros_i32(n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=torch.int32, device=device)

    def full_i64(n: int, value: int) -> torch.Tensor:
        return torch.full((n,), value, dtype=torch.int64, device=device)

    return BatchPlanGpu(
        verify_slot_indices=zeros_i64(verify_capacity),
        verify_positions=zeros_i64(verify_capacity),
        verify_prev_slot_indices=zeros_i64(verify_capacity),
        verify_num_valid=zeros_i32(1),
        write_slot_indices=zeros_i64(write_capacity),
        write_token_ids=zeros_i64(write_capacity),
        write_positions=zeros_i64(write_capacity),
        expected_write_token_ids=full_i64(write_capacity, _SKIP_SENTINEL),
        expected_write_positions=full_i64(write_capacity, _SKIP_SENTINEL),
        write_req_seed_slot_indices=zeros_i64(write_req_capacity),
        write_req_entry_starts=zeros_i64(write_req_capacity),
        write_req_entry_counts=zeros_i64(write_req_capacity),
        write_req_num_valid=zeros_i32(1),
    )


def fill_batch_plan_gpu_from_plan(
    *,
    launch: BatchPlanGpu,
    plan: BatchPlan,
) -> Tuple[int, int]:
    """Copy a host-side ``BatchPlan`` into the fixed GPU tensors in place.

    Returns ``(num_active_verify, num_active_write_reqs)``. Rows past
    the active count are skipped by the kernel. The write-entry tile is also reset past
    ``plan.num_write`` so a write-req driver that reads
    ``write_*[entry_start + j]`` cannot pick up stale data.

    The verify range now covers the full ``[0, K_req)`` of every req
    (SWA pools clip to ``[K_req - window_size, K_req)`` at plan time),
    and ``verify_capacity`` is sized off ``max_total_num_tokens`` so
    the buffer can hold every slot in the pool simultaneously. An
    overflow here means the plan computed more verify entries than the
    canary's slot pool — a logic bug, not a budget choice — so we
    raise instead of silently truncating.
    """
    verify_capacity = int(launch.verify_slot_indices.shape[0])
    write_capacity = int(launch.write_slot_indices.shape[0])
    write_req_capacity = int(launch.write_req_seed_slot_indices.shape[0])

    if plan.num_verify > verify_capacity:
        raise RuntimeError(
            f"kv-canary: verify entry count {plan.num_verify} exceeds "
            f"verify_capacity {verify_capacity}. This should be "
            "unreachable when verify_capacity == max_total_num_tokens; "
            "indicates a planner bug or undersized capacity."
        )
    if plan.num_write > write_capacity:
        raise RuntimeError(
            f"kv-canary: write entry count {plan.num_write} exceeds "
            f"write_capacity {write_capacity}. Raise canary launch "
            "capacity or disable the canary for this deployment."
        )
    if plan.num_write_reqs > write_req_capacity:
        raise RuntimeError(
            f"kv-canary: write req count {plan.num_write_reqs} exceeds "
            f"write_req_capacity {write_req_capacity}."
        )

    num_active_verify = plan.num_verify
    v = slice(0, num_active_verify)

    device = launch.verify_slot_indices.device

    def to_i64(values: List[int], sl: slice) -> torch.Tensor:
        return torch.tensor(values[sl], dtype=torch.int64, device=device)

    if num_active_verify > 0:
        launch.verify_slot_indices[:num_active_verify].copy_(
            to_i64(plan.verify_slot_indices, v)
        )
        launch.verify_positions[:num_active_verify].copy_(
            to_i64(plan.verify_positions, v)
        )
        launch.verify_prev_slot_indices[:num_active_verify].copy_(
            to_i64(plan.verify_prev_slot_indices, v)
        )
    launch.verify_num_valid.fill_(num_active_verify)

    nw = plan.num_write
    if nw > 0:
        launch.write_slot_indices[:nw].copy_(
            to_i64(plan.write_slot_indices, slice(0, nw))
        )
        launch.write_token_ids[:nw].copy_(to_i64(plan.write_token_ids, slice(0, nw)))
        launch.write_positions[:nw].copy_(to_i64(plan.write_positions, slice(0, nw)))
    if nw < write_capacity:
        launch.write_slot_indices[nw:].zero_()
        launch.write_token_ids[nw:].zero_()
        launch.write_positions[nw:].zero_()

    # Oracle-off callers leave these buffers in their canonical
    # all-sentinel state from allocate_batch_plan_gpu(); pseudo-mode rewrites
    # [:nw] and restores the tail. The kernel skips entries where
    # write_req_num_valid excludes the row, so the tail being stale doesn't
    # produce false violations — but keeping it at the sentinel value
    # keeps reset_batch_plan_gpu_to_skip_sentinel from touching these buffers.
    if plan.expected_write_token_ids is not None:
        if nw > 0:
            launch.expected_write_token_ids[:nw].copy_(
                to_i64(plan.expected_write_token_ids, slice(0, nw))
            )
            launch.expected_write_positions[:nw].copy_(
                to_i64(plan.expected_write_positions, slice(0, nw))
            )
        if nw < write_capacity:
            launch.expected_write_token_ids[nw:].fill_(_SKIP_SENTINEL)
            launch.expected_write_positions[nw:].fill_(_SKIP_SENTINEL)

    nwr = plan.num_write_reqs
    if nwr > 0:
        launch.write_req_seed_slot_indices[:nwr].copy_(
            to_i64(plan.write_req_seed_slot_indices, slice(0, nwr))
        )
        launch.write_req_entry_starts[:nwr].copy_(
            to_i64(plan.write_req_entry_starts, slice(0, nwr))
        )
        launch.write_req_entry_counts[:nwr].copy_(
            to_i64(plan.write_req_entry_counts, slice(0, nwr))
        )
    launch.write_req_num_valid.fill_(nwr)

    return num_active_verify, nwr


def reset_batch_plan_gpu_to_skip_sentinel(launch: BatchPlanGpu) -> None:
    """Reset all active counts so the recorded kernel becomes a no-op.

    Used at capture time and at replay-time when no valid plan exists.
    The expected_write_* buffers do not need touching here: the
    kernel early-exits on a zero write_req_num_valid before reading
    them.
    """
    launch.verify_num_valid.zero_()
    launch.write_req_num_valid.zero_()

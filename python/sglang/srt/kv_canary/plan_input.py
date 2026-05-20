"""Plan-input builders for the two canary caller paths.

Two builders construct or fill a :class:`PlanInput` from sglang state; the runner feeds it into
``canary_plan_step`` alongside the pre-allocated VerifyPlan + WritePlan out buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanInput:
    """Pre-staged input to canary_plan_step. Built by one of two builders (per-forward /
    radix-sweep); fed straight into canary_plan_step alongside the pre-allocated VerifyPlan +
    WritePlan out buffers.

    All tensors live on device.

    Fields:
        fb_req_pool_indices: Per-row ReqToTokenPool row index, shape [bs_capacity], int64.
            0 = padding sentinel. Per-forward path: pre-allocated static buffer that the runner
            writes into each step (see Static-buffer contract below). Radix-sweep path: empty.
        fb_prefix_lens: Per-req prefix length already written before this step, shape
            [bs_capacity], int32. Per-forward extend → extend_prefix_lens; per-forward decode →
            seq_lens - 1; radix-sweep → empty.
        fb_extend_seq_lens: Per-req tokens being written this step, shape [bs_capacity], int32.
            Per-forward → extend or all-ones (decode); radix-sweep → empty.
        extra_verify_slot_indices: Pre-walked flat verify slots, shape [extra_verify_capacity],
            int32. Caller-translated to the target index space (SWA-aware if running on the SWA
            group).
        extra_verify_positions: Same shape, int32. Expected position per extra entry.
        extra_verify_prev_slot_indices: Same shape, int32. -1 for chain-seed extras.
        extra_verify_num_valid: Active extra entry count, shape [1], int32. 0 for the per-forward
            caller.

    extra_verify_capacity is per-runner: per-forward path uses 0 (path doesn't emit extras);
    radix-sweep path uses total-pool-slots (worst case radix-orphan covers entire pool).
    Allocated up front by CanaryRunner. ForwardBatch.positions has dtype-by-backend (cuda=int32,
    hip/npu=int64) — runner casts to int32 before passing to canary_write_step if needed.

    **Static-buffer contract (cuda-graph correctness)**: the per-forward PlanInput's tensors are
    allocated once during CanaryRunner.__init__ (sized for the worst-case per-forward batch).
    The per-forward builder MUTATES those tensors in place each step via ``.copy_()`` /
    ``.fill_()`` / index-assign on the default stream. The captured cuda-graph reads them by
    address; therefore: (a) never reallocate, (b) all writes complete on the default stream
    before the captured region runs (i.e. the writes happen in ``CanaryRunner.before_forward``
    which the caller invokes in ``ModelRunner.forward`` BEFORE ``graph_runner.replay()`` or
    ``model.forward()``).
    """

    fb_req_pool_indices: torch.Tensor
    fb_prefix_lens: torch.Tensor
    fb_extend_seq_lens: torch.Tensor
    extra_verify_slot_indices: torch.Tensor
    extra_verify_positions: torch.Tensor
    extra_verify_prev_slot_indices: torch.Tensor
    extra_verify_num_valid: torch.Tensor


def fill_plan_input_per_forward(
    *,
    forward_batch: "ForwardBatch",
    plan_input_out: PlanInput,
) -> int:
    """Builder for the per-forward (head + tail) caller. MUTATES plan_input_out's static buffers
    in place — does NOT return a new PlanInput (see PlanInput Static-buffer contract).

    - plan_input_out.fb_req_pool_indices[:bs] ← forward_batch.req_pool_indices; rows beyond bs
      are zeroed (padding sentinel).
    - plan_input_out.fb_prefix_lens[:bs] ← extend mode: forward_batch.extend_prefix_lens;
      decode mode: forward_batch.seq_lens - 1. Rows beyond bs are zeroed (padding skipped via
      req_pool_indices sentinel; the lens value there is irrelevant).
    - plan_input_out.fb_extend_seq_lens[:bs] ← extend: forward_batch.extend_seq_lens;
      decode: all-ones.
    - extra_verify_* untouched — they stay all-zero / num_valid = 0 (allocated once with 0
      capacity).

    Returns the current bs (valid prefix length of the per-forward buffers).
    """
    req_pool_indices = forward_batch.req_pool_indices
    bs = int(req_pool_indices.shape[0])
    capacity = int(plan_input_out.fb_req_pool_indices.shape[0])
    if bs > capacity:
        raise RuntimeError(
            f"kv-canary: per-forward batch size {bs} exceeds static capacity {capacity}; "
            "raise the buffer size in CanaryRunner.__init__"
        )

    plan_input_out.fb_req_pool_indices.zero_()
    plan_input_out.fb_req_pool_indices[:bs].copy_(req_pool_indices)
    plan_input_out.fb_prefix_lens.zero_()
    plan_input_out.fb_extend_seq_lens.zero_()

    forward_mode = forward_batch.forward_mode
    if forward_mode is not None and forward_mode.is_extend():
        plan_input_out.fb_prefix_lens[:bs].copy_(
            forward_batch.extend_prefix_lens.to(torch.int32)
        )
        plan_input_out.fb_extend_seq_lens[:bs].copy_(
            forward_batch.extend_seq_lens.to(torch.int32)
        )
    else:
        plan_input_out.fb_prefix_lens[:bs].copy_(
            (forward_batch.seq_lens - 1).to(torch.int32)
        )
        plan_input_out.fb_extend_seq_lens[:bs].fill_(1)

    return bs


def build_plan_input_radix_sweep(
    *,
    radix_cache: "BasePrefixCache",
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> PlanInput:
    """Builder for the radix-sweep caller.

    - fb_req_pool_indices: empty (bs = 0); plan kernel skips the per-req path entirely.
    - fb_prefix_lens / fb_extend_seq_lens: empty.
    - extra_verify_* populated by walk_radix_cache_for_canary covering EVERY slot held by the
      radix tree (no exclusion of running-req-owned slots — the overlap with per-forward
      HEAD/TAIL coverage is harmless redundancy). The builder applies the SWA LUT before
      stuffing into PlanInput (plan kernel does NOT translate extras).
    """
    device = radix_cache.req_to_token_pool.req_to_token.device

    slot_indices, positions, prev_slot_indices = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
    )
    slot_indices = slot_indices.to(device)
    positions = positions.to(device)
    prev_slot_indices = prev_slot_indices.to(device)

    if swa_window_size > 0 and full_to_swa_index_mapping is not None:
        slot_indices = _swa_translate(
            indices=slot_indices,
            lut=full_to_swa_index_mapping,
        )
        prev_slot_indices = _swa_translate(
            indices=prev_slot_indices,
            lut=full_to_swa_index_mapping,
        )

    num_valid = int(slot_indices.shape[0])

    fb_req_pool_indices = torch.empty(0, dtype=torch.int64, device=device)
    fb_prefix_lens = torch.empty(0, dtype=torch.int32, device=device)
    fb_extend_seq_lens = torch.empty(0, dtype=torch.int32, device=device)
    extra_num_valid = torch.tensor([num_valid], dtype=torch.int32, device=device)

    return PlanInput(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        extra_verify_slot_indices=slot_indices,
        extra_verify_positions=positions,
        extra_verify_prev_slot_indices=prev_slot_indices,
        extra_verify_num_valid=extra_num_valid,
    )


def walk_radix_cache_for_canary(
    *,
    radix_cache: "BasePrefixCache",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk the radix tree and emit flat (slot_indices, positions, prev_slot_indices) tensors for
    EVERY slot held by the radix cache (including slots whose tokens are also referenced by a
    currently-running req — that overlap is harmless redundancy with the per-forward HEAD/TAIL
    path).

    For each radix tree node:
    - Slots within the node are chained in order; slot at within-node index j has predecessor at
      j - 1.
    - The first slot of a non-root node's chain has predecessor = the last slot of the parent
      node.
    - The first slot of a root node's first child has predecessor = -1 (chain-seed anchor).
    - Position = depth-from-root of the slot.

    Returns three host int32 tensors (then runner H2D-copies). NOT SWA-translated — caller does
    the LUT lookup before packing into PlanInput.

    Cost: O(total radix slots). Runs on host every sweep_every_n_steps; bounded by pool size.
    If profiling shows this is the sweep hot path, future work can move it to a Triton kernel —
    but for sweep cadences in the 64..1024 range, host walk is fine.
    """
    cache_type = type(radix_cache)
    if cache_type is not RadixCache and cache_type is not SWARadixCache:
        raise NotImplementedError(
            f"walk_radix_cache_for_canary does not support {cache_type.__name__}"
        )

    slot_buf: list[int] = []
    position_buf: list[int] = []
    prev_slot_buf: list[int] = []

    _walk_radix_subtree(
        node=radix_cache.root_node,
        depth=0,
        parent_last_slot=-1,
        slot_buf=slot_buf,
        position_buf=position_buf,
        prev_slot_buf=prev_slot_buf,
        is_root=True,
    )

    slot_tensor = torch.tensor(slot_buf, dtype=torch.int32)
    position_tensor = torch.tensor(position_buf, dtype=torch.int32)
    prev_slot_tensor = torch.tensor(prev_slot_buf, dtype=torch.int32)
    return slot_tensor, position_tensor, prev_slot_tensor


def _walk_radix_subtree(
    *,
    node: "TreeNode",
    depth: int,
    parent_last_slot: int,
    slot_buf: list[int],
    position_buf: list[int],
    prev_slot_buf: list[int],
    is_root: bool,
) -> None:
    if isinstance(node.value, torch.Tensor):
        node_slots = [int(s) for s in node.value.tolist()]
    else:
        node_slots = []

    emit_slots = not is_root and node.lock_ref == 0

    chain_last_slot = parent_last_slot
    for j, slot in enumerate(node_slots):
        prev = parent_last_slot if j == 0 else node_slots[j - 1]
        if emit_slots:
            slot_buf.append(slot)
            position_buf.append(depth + j)
            prev_slot_buf.append(prev)
        chain_last_slot = slot

    child_depth = depth + len(node_slots)
    for child in node.children.values():
        _walk_radix_subtree(
            node=child,
            depth=child_depth,
            parent_last_slot=chain_last_slot,
            slot_buf=slot_buf,
            position_buf=position_buf,
            prev_slot_buf=prev_slot_buf,
            is_root=False,
        )


def _swa_translate(
    *,
    indices: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    if indices.numel() == 0:
        return indices
    lut_dev = lut.to(indices.device).to(torch.int32)
    sentinel_mask = indices < 0
    safe = torch.where(sentinel_mask, torch.zeros_like(indices), indices).to(
        torch.int64
    )
    translated = lut_dev[safe]
    return torch.where(
        sentinel_mask, indices.to(torch.int32), translated.to(torch.int32)
    )

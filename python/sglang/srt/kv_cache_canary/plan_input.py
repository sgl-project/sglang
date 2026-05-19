"""Plan-input builders for the three canary caller paths.

Per upper-level SOT §2.3 / §4.1 / §4.2 / §4.3. Three builders construct a :class:`PlanInput` from sglang
state; the runner feeds it into ``canary_plan_step`` alongside the pre-allocated VerifyPlan + WritePlan
out buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanInput:
    """Pre-staged input to canary_plan_step. Built by one of three builders (per-forward / running-sweep
    / radix-sweep); fed straight into canary_plan_step alongside the pre-allocated VerifyPlan + WritePlan
    out buffers.

    All tensors live on device; builders may produce them via host-side construction + a single H2D copy
    OR via Triton/torch ops directly on device. The runner does NOT inspect this struct's contents — it
    only passes it through to canary_plan_step.

    Fields:
        fb_req_pool_indices: Per-row ReqToTokenPool row index, shape [bs], int64 (matches
            ForwardBatch.req_pool_indices in sglang). 0 = padding sentinel. Per-forward callers pass
            forward_batch.req_pool_indices directly; sweep callers synthesize.
        fb_prefix_lens: Per-req prefix length already written before this step, shape [bs], int32.
            Per-forward extend → extend_prefix_lens; per-forward decode → seq_lens - 1; running-sweep →
            seq_lens; radix-sweep → all-zero (orphans come in via extra_*).
        fb_extend_seq_lens: Per-req tokens being written this step, shape [bs], int32. 0 for sweep
            callers (so the produced WritePlan has write_num_valid_reqs = 0 and is unused downstream).
        extra_verify_slot_indices: Pre-walked flat verify slots, shape [extra_verify_capacity], int32.
            Caller-translated to the target index space (SWA-aware if running on the SWA group).
        extra_verify_positions: Same shape, int32. Expected position per extra entry.
        extra_verify_prev_slot_indices: Same shape, int32. -1 for chain-seed extras.
        extra_verify_num_valid: Active extra entry count, shape [1], int32. 0 for per-forward and
            running-sweep callers.

    extra_verify_capacity is per-runner: per-forward path uses 0 (path doesn't emit extras);
    sweep path uses total-pool-slots (worst case radix-orphan covers entire pool). Allocated up
    front by CanaryRunner. ForwardBatch.positions has dtype-by-backend (cuda=int32, hip/npu=int64)
    — runner casts to int32 before passing to canary_write_step if needed.
    """

    fb_req_pool_indices: torch.Tensor
    fb_prefix_lens: torch.Tensor
    fb_extend_seq_lens: torch.Tensor
    extra_verify_slot_indices: torch.Tensor
    extra_verify_positions: torch.Tensor
    extra_verify_prev_slot_indices: torch.Tensor
    extra_verify_num_valid: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class AliveReqSnapshot:
    """Snapshot of every alive running req's pool index + written prefix length.

    Both tensors live on the same device and have shape [bs], int64 (req_pool_indices) and int32
    (seq_lens). The running-sweep builder consumes this directly; the radix-orphan builder uses the
    pool-index set to dedupe slots against the running set.
    """

    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor


def build_plan_input_per_forward(
    *,
    forward_batch: "ForwardBatch",
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> PlanInput:
    """Builder for the per-forward (head + tail) caller.

    - fb_req_pool_indices = forward_batch.req_pool_indices.
    - fb_prefix_lens: extend mode → forward_batch.extend_prefix_lens; decode mode →
      forward_batch.seq_lens - 1.
    - fb_extend_seq_lens: extend → forward_batch.extend_seq_lens; decode → all-ones.
    - extra_verify_* all zero / empty (num_valid = 0).
    """
    del swa_window_size, full_to_swa_index_mapping

    req_pool_indices = forward_batch.req_pool_indices
    device = req_pool_indices.device
    bs = int(req_pool_indices.shape[0])

    forward_mode = forward_batch.forward_mode
    if forward_mode is not None and forward_mode.is_extend():
        fb_prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32).contiguous()
        fb_extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32).contiguous()
    else:
        fb_prefix_lens = (
            (forward_batch.seq_lens - 1).clamp(min=0).to(torch.int32).contiguous()
        )
        fb_extend_seq_lens = torch.ones(bs, dtype=torch.int32, device=device)

    extra_slot = torch.zeros(1, dtype=torch.int32, device=device)
    extra_position = torch.zeros(1, dtype=torch.int32, device=device)
    extra_prev = torch.zeros(1, dtype=torch.int32, device=device)
    extra_num_valid = torch.zeros(1, dtype=torch.int32, device=device)

    return PlanInput(
        fb_req_pool_indices=req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        extra_verify_slot_indices=extra_slot,
        extra_verify_positions=extra_position,
        extra_verify_prev_slot_indices=extra_prev,
        extra_verify_num_valid=extra_num_valid,
    )


def build_plan_input_running_sweep(
    *,
    req_to_token_pool: "ReqToTokenPool",
    alive_reqs: AliveReqSnapshot,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> PlanInput:
    """Builder for the running-reqs portion of sweep.

    - fb_req_pool_indices: req_pool_indices of every alive running req (including paused, excluding
      queued / freed), gathered on device.
    - fb_prefix_lens: alive_reqs.seq_lens (full written prefix per req).
    - fb_extend_seq_lens: all-zero — sweep never writes, only verifies. Plan kernel still produces a
      WritePlan but with write_num_valid_reqs = 0.
    - extra_verify_* all zero / empty.
    """
    del req_to_token_pool, swa_window_size, full_to_swa_index_mapping

    req_pool_indices = alive_reqs.req_pool_indices
    seq_lens = alive_reqs.seq_lens
    device = req_pool_indices.device
    bs = int(req_pool_indices.shape[0])

    fb_prefix_lens = seq_lens.to(torch.int32).contiguous()
    fb_extend_seq_lens = torch.zeros(bs, dtype=torch.int32, device=device)

    extra_slot = torch.zeros(1, dtype=torch.int32, device=device)
    extra_position = torch.zeros(1, dtype=torch.int32, device=device)
    extra_prev = torch.zeros(1, dtype=torch.int32, device=device)
    extra_num_valid = torch.zeros(1, dtype=torch.int32, device=device)

    return PlanInput(
        fb_req_pool_indices=req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        extra_verify_slot_indices=extra_slot,
        extra_verify_positions=extra_position,
        extra_verify_prev_slot_indices=extra_prev,
        extra_verify_num_valid=extra_num_valid,
    )


def build_plan_input_radix_sweep(
    *,
    radix_cache: "BasePrefixCache",
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> PlanInput:
    """Builder for the radix-orphan portion of sweep.

    - fb_req_pool_indices: empty (bs = 0); plan kernel skips the per-req path entirely.
    - fb_prefix_lens / fb_extend_seq_lens: empty.
    - extra_verify_* populated by walk_radix_cache_for_canary (see §4.2). SWA-translated by THIS builder
      before stuffing into PlanInput (plan kernel does NOT translate extras).
    """
    device = radix_cache.req_to_token_pool.req_to_token.device

    slot_indices, positions, prev_slot_indices = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
        alive_running_req_pool_indices=set(),
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
    alive_running_req_pool_indices: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk the radix tree and emit flat (slot_indices, positions, prev_slot_indices) tensors for every
    slot held by the cache that is NOT owned by any running req (i.e. radix-orphan slots — the alive
    set minus what running_sweep already covers).

    For each radix tree node:
    - Slots within the node are chained in order; slot at within-node index j has predecessor at j - 1.
    - The first slot of a non-root node's chain has predecessor = the last slot of the parent node.
    - The first slot of a root node's first child has predecessor = -1 (chain-seed anchor).
    - Position = depth-from-root of the slot.

    Returns three host int32 tensors (then runner H2D-copies). NOT SWA-translated — caller does the
    LUT lookup before packing into PlanInput.

    Cost: O(total radix slots). Runs on host every sweep_every_n_steps; bounded by pool size. If
    profiling shows this is the sweep hot path, future work can move it to a Triton kernel — but for
    sweep cadences in the 64..1024 range, host walk is fine.
    """
    del alive_running_req_pool_indices

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
) -> None:
    if isinstance(node.value, torch.Tensor):
        node_slots = [int(s) for s in node.value.tolist()]
    else:
        node_slots = []

    chain_last_slot = parent_last_slot
    for j, slot in enumerate(node_slots):
        prev = parent_last_slot if j == 0 else node_slots[j - 1]
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

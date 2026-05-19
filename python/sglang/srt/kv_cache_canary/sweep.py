from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import (
    _CANARY_FIELD_POSITION,
    SKIP_CHAIN_SENTINEL,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref import BatchPlan

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def extract_active_rows(
    forward_batch: Optional["ForwardBatch"],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Pull (req_pool_indices, seq_lens) host lists from a forward batch."""
    if forward_batch is None:
        return None, None
    if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
        return None, None
    indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
    return indices, seq_lens


def compute_alive_owned_slots(
    *,
    req_to_token_pool: "ReqToTokenPool",
    forward_batch: "ForwardBatch",
) -> torch.Tensor:
    """Return slot indices for every token owned by an alive req in ``forward_batch``.

    Slots owned by reqs not in the current batch (free / queued / paused) are
    excluded — sweeping them would alias against the allocator's reuse of those
    bytes and produce false positives. Per allocator invariant, a given slot is
    owned by at most one alive req, so the concatenation has no duplicates.
    """
    device = req_to_token_pool.req_to_token.device
    req_indices_cpu, seq_lens_cpu = extract_active_rows(forward_batch)
    if not req_indices_cpu or not seq_lens_cpu:
        return torch.empty(0, dtype=torch.int64, device=device)

    chunks: List[torch.Tensor] = []
    for r, n in zip(req_indices_cpu, seq_lens_cpu):
        r_int = int(r)
        n_int = int(n)
        # r == 0 is the cuda-graph padding row; n == 0 has no committed tokens.
        if r_int <= 0 or n_int <= 0:
            continue
        chunks.append(req_to_token_pool.req_to_token[r_int, :n_int].to(torch.int64))

    if not chunks:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(chunks, dim=0)


def build_sweep_plan(
    *,
    canary_buf: torch.Tensor,
    alive_slot_indices: torch.Tensor,
) -> BatchPlan:
    """Build a verify-only plan that sweeps every alive slot's real_kv_hash.

    Sets ``verify_prev_slot_indices`` to ``SKIP_CHAIN_SENTINEL`` (sweep cannot
    reconstruct chain ordering across an arbitrary alive set). Feeds back each
    slot's stored ``position`` as ``verify_positions`` (tautology pass — chain
    is skipped and the monotonic check sees stored == stored). The kernel's
    real_kv_hash recompute is the actual detection that runs.
    """
    if alive_slot_indices.numel() == 0:
        return BatchPlan.empty()

    int_view = canary_buf.view(torch.int64).reshape(canary_buf.shape[0], -1)
    verify_positions = (
        int_view[alive_slot_indices, _CANARY_FIELD_POSITION].cpu().tolist()
    )
    verify_slot_indices = alive_slot_indices.detach().cpu().tolist()

    n = len(verify_slot_indices)
    return BatchPlan(
        verify_positions=verify_positions,
        verify_slot_indices=verify_slot_indices,
        verify_prev_slot_indices=[SKIP_CHAIN_SENTINEL] * n,
        write_token_ids=[],
        write_positions=[],
        write_slot_indices=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_pool_indices=[],
        num_verify=n,
        num_write=0,
        num_write_reqs=0,
    )

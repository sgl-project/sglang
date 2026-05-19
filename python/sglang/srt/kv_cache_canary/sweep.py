from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.jit_kernel.kv_cache_canary import (
    _CANARY_FIELD_POSITION,
    SKIP_CHAIN_SENTINEL,
)
from sglang.srt.kv_cache_canary.host_state import BatchPlan

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_alive_owned_slots(
    *,
    req_to_token_pool: "ReqToTokenPool",
    forward_batch: "ForwardBatch",
) -> torch.Tensor:
    """Return slot indices for every token owned by an alive req in ``forward_batch``.

    Alive = req_pool_idx is in ``forward_batch.req_pool_indices`` (the scheduler
    is currently running this req). Owned = the slot is in that req's
    ``[0, seq_len)`` range of ``req_to_token``.

    The per-step canary head kernel ran at the start of this forward, so every
    returned slot has a fresh shadow freeze that the sweep can verify against.
    Slots owned by reqs that are not in the current batch (already free, queued,
    or paused) are excluded — sweeping them would alias against the allocator's
    reuse of those bytes and produce false positives.

    Returns a flat int64 GPU tensor of unique slot indices.
    """
    device = req_to_token_pool.req_to_token.device
    if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
        return torch.empty(0, dtype=torch.int64, device=device)

    req_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens
    if req_indices.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    req_indices_cpu = req_indices.detach().cpu().tolist()
    seq_lens_cpu = seq_lens.detach().cpu().tolist()

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
    return torch.unique(torch.cat(chunks, dim=0))


def build_sweep_plan(
    *,
    canary_buf: torch.Tensor,
    slot_stride_bytes: int,
    alive_slot_indices: torch.Tensor,
) -> BatchPlan:
    """Build a verify-only plan that sweeps every alive slot's real_kv_hash.

    The returned plan:

    - Verifies every alive slot (one verify entry per slot).
    - Sets ``verify_prev_slot_indices`` to ``SKIP_CHAIN_SENTINEL`` so the
      kernel skips the chain hash check (sweep cannot reconstruct chain
      ordering across an arbitrary alive set).
    - Feeds back each slot's stored ``position`` from the canary buffer as
      ``verify_positions`` (tautology pass — chain hash check is skipped and
      the position monotonic check trivially passes).
    - Writes nothing.

    The real_kv_hash recompute + compare branch in the kernel is the actual
    detection that runs.

    ``canary_buf`` must be shape ``[num_slots, CANARY_SLOT_BYTES]`` uint8 and
    live on the same device as ``alive_slot_indices``. ``slot_stride_bytes``
    is kept in the signature for forward compatibility; the position field
    offset is fixed by the shadow layout.
    """
    if alive_slot_indices.numel() == 0:
        return BatchPlan.empty()

    int_view = canary_buf.view(torch.int64).reshape(canary_buf.shape[0], -1)
    rows = int_view[alive_slot_indices].cpu().tolist()
    slot_indices = alive_slot_indices.detach().cpu().tolist()

    n = len(slot_indices)
    verify_slot_indices = [int(s) for s in slot_indices]
    verify_positions = [int(r[_CANARY_FIELD_POSITION]) for r in rows]
    verify_prev_slot_indices = [SKIP_CHAIN_SENTINEL] * n

    _ = slot_stride_bytes
    return BatchPlan(
        verify_positions=verify_positions,
        verify_slot_indices=verify_slot_indices,
        verify_prev_slot_indices=verify_prev_slot_indices,
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

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.plan.entries_kernel import (
    launch_plan_entries_kernel,
)
from sglang.jit_kernel.kv_canary.plan.offsets_kernel import (
    _PLAN_BS_BLOCK_SIZE,
    launch_plan_offsets_kernel,
)
from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan


def canary_plan_step(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    verify_capacity: int,
) -> None:
    """Fill verify_plan_out + write_plan_out from ForwardBatch primitives.

    For each req r with fb_req_pool_indices[r] != 0 (0 = padding sentinel):

    - **Verify entries**: one per pos in [window_start, fb_prefix_lens[r]), where window_start = max(0,
      fb_prefix_lens[r] - swa_window_size) if SWA else 0. slot_idx = req_to_token[fb_req_pool_indices[r], pos]
      (SWA-translated via full_to_swa_index_mapping if non-None); prev_slot_idx =
      req_to_token[fb_req_pool_indices[r], pos-1] for pos > 0, else -1. (SWA windows do NOT reset the chain —
      the writer chains across the entire prefix; sweep verify within an SWA window dereferences the real
      predecessor for chain-link reconstruction.)
    - **Write metadata** (when fb_extend_seq_lens[r] > 0): contribute fb_extend_seq_lens[r] to the per-req
      write count (for write_offsets cumsum). Per-req chain seed = req_to_token[fb_req_pool_indices[r],
      fb_prefix_lens[r]-1] (SWA-translated), or -1 if fb_prefix_lens[r] == 0. Per-token write data
      (fb_input_ids / fb_positions / fb_out_cache_loc) is NOT materialized here — canary_write_step reads it
      directly from ForwardBatch via write_offsets.

    Args:
        verify_plan_out: Pre-allocated VerifyPlan; filled in-place.
        write_plan_out: Pre-allocated WritePlan; filled in-place.
        fb_req_pool_indices: ForwardBatch.req_pool_indices; per-row ReqToTokenPool row index, shape [bs],
            int64. 0 is the padding sentinel.
        fb_prefix_lens: Per-req prefix length already written before this step, shape [bs], int64. Caller
            normalizes: extend → ForwardBatch.extend_prefix_lens, decode → ForwardBatch.seq_lens - 1.
        fb_extend_seq_lens: ForwardBatch.extend_seq_lens; per-req tokens being written this step, shape [bs],
            int64. 1 for pure decode.
        req_to_token: ReqToTokenPool.req_to_token; full-pool slot index table, shape [max_reqs, max_seq_len],
            int32.
        swa_window_size: 0 for the FULL canary group; positive window length for the SWA group.
        full_to_swa_index_mapping: SWA LUT, shape [full_pool_size + 1], int64, or None. Required (non-None) iff
            swa_window_size > 0. Used to translate verify slot indices and chain-seed slot indices at plan time.
            Loaded element-typed via Triton ``tl.load``; intermediate translated slot values are int64 inside the
            kernel and stored in the int64 plan schema.

    Implementation:
        - Two sub-kernels with action-named identifiers, launched in sequence:
          1. ``_plan_offsets_kernel`` (1-D grid ``(1,)``, single program over all ``bs`` reqs):
             reads fb_req_pool_indices[r], fb_prefix_lens[r], fb_extend_seq_lens[r] for each r; computes
             verify_count = (prefix_lens - window_start) and write_count = extend_seq_lens (both 0 if rp == 0
             padding); gathers seed_slot_full = req_to_token[rp, prefix_lens - 1] (or -1 if prefix_lens == 0),
             SWA-translates seed_slot via full_to_swa_index_mapping[seed_slot_full] if non-None; runs
             block-level cumsum (``tl.cumsum``) to produce verify_offsets[bs+1] and
             write_plan_out.write_offsets[bs+1] in-place; scatters write_seed slots; writes scalar totals
             ``verify_plan_out.verify_num_valid`` and ``write_plan_out.write_num_valid_reqs``.
          2. ``_plan_entries_kernel`` (2-D grid ``(bs, max_j_tiles)``, masked by per-req verify_count): for
             each (r, j) with j < verify_count[r], gather slot = req_to_token[fb_req_pool_indices[r],
             window_start[r] + j] (SWA-translated), prev_slot = req_to_token[..., window_start[r] + j - 1]
             when (window_start[r] + j) > 0 (also translated) else -1, position = window_start[r] + j;
             scatter (slot, position, prev_slot) into verify_plan_out at flat index verify_offsets[r] + j.
        - All output tensors are addressed at addresses baked into the cuda-graph capture.

    Calling contract:
        - Pure side-effect; no host work, no D2H.
        - Safe in cuda-graph capture; caller refills all input tensors in-place before replay.
        - Single kernel launch fills both plans end-to-end.
        - Padding rows contribute zero entries.

    Pinned by Python reference
    :func:`sglang.jit_kernel.kv_canary.plan_ref.canary_plan_step_torch_reference`; Triton must match
    byte-for-byte.
    """
    bs = int(fb_req_pool_indices.shape[0])
    if bs > _PLAN_BS_BLOCK_SIZE:
        raise ValueError(
            f"kv-canary: canary_plan_step supports at most bs={_PLAN_BS_BLOCK_SIZE} reqs per launch, "
            f"got bs={bs}. Bump _PLAN_BS_BLOCK_SIZE if real workloads need this."
        )
    if swa_window_size > 0 and full_to_swa_index_mapping is None:
        raise ValueError(
            "kv-canary: canary_plan_step requires full_to_swa_index_mapping when swa_window_size > 0"
        )

    device = verify_plan_out.verify_slot_indices.device
    verify_offsets_scratch = torch.zeros(
        _PLAN_BS_BLOCK_SIZE + 1, dtype=torch.int64, device=device
    )

    plan_verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    if verify_capacity != plan_verify_capacity:
        raise ValueError(
            f"kv-canary: canary_plan_step verify_capacity={verify_capacity} does not match "
            f"verify_plan_out.verify_slot_indices.shape[0]={plan_verify_capacity}"
        )
    # Match the ref's tail-reset semantics: write_offsets positions past index bs are zeroed so a smaller
    # batch never leaks stale prefix-sum entries from a larger previous call. In-place .zero_() is
    # cuda-graph-safe (no allocation) and avoids one Triton launch.
    write_plan_out.write_offsets.zero_()

    # Offsets kernel: per-req count + seed gather + block-level cumsum, single program; the num_valid
    # scalars are written by the same program (it has the totals in registers already).
    launch_plan_offsets_kernel(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        out_verify_offsets_scratch=verify_offsets_scratch,
        out_write_offsets=write_plan_out.write_offsets,
        out_write_seed_slot_indices=write_plan_out.write_seed_slot_indices,
        out_verify_num_valid=verify_plan_out.verify_num_valid,
        out_verify_enable=verify_plan_out.enable,
        out_write_num_valid_reqs=write_plan_out.write_num_valid_reqs,
        swa_window_size=int(swa_window_size),
        verify_capacity=verify_capacity,
    )

    # Entries kernel: per-(req, j-tile) verify entry materialization. The j-axis upper bound is
    # verify_capacity (each req cannot contribute more than verify_capacity entries); we mask per-req actual
    # count read back from verify_offsets_scratch inside the kernel.
    launch_plan_entries_kernel(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_offsets_scratch=verify_offsets_scratch,
        out_verify_slot_indices=verify_plan_out.verify_slot_indices,
        out_verify_positions=verify_plan_out.verify_positions,
        out_verify_prev_slot_indices=verify_plan_out.verify_prev_slot_indices,
        swa_window_size=int(swa_window_size),
    )

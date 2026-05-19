"""Triton host wrapper for the canary plan accumulator.

Defines :func:`canary_plan_step` — the single-launch Triton plan kernel
that fills a :class:`~sglang.jit_kernel.kv_cache_canary_verify.VerifyPlan`
and a :class:`~sglang.jit_kernel.kv_cache_canary_write.WritePlan` from
ForwardBatch primitives plus optional pre-walked flat verify extras.

Byte-equal pinned by
:func:`sglang.jit_kernel.kv_cache_canary_plan_ref.canary_plan_step_torch_reference`.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_cache_canary_verify import VerifyPlan
from sglang.jit_kernel.kv_cache_canary_write import WritePlan


def canary_plan_step(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extra_verify_slot_indices: torch.Tensor,
    extra_verify_positions: torch.Tensor,
    extra_verify_prev_slot_indices: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> None:
    """Fill verify_plan_out + write_plan_out from ForwardBatch primitives + optional pre-walked flat verify
    entries. Single Triton launch.

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

    Extra flat verify entries (extra_verify_*[: extra_verify_num_valid[0]]) are appended to verify_plan_out
    **after** the per-req-derived entries. Used by radix-cache-orphan sweep; caller is responsible for
    SWA-translating these entries before passing in (plan kernel does NOT translate the extras).

    Sweep callers pass fb_extend_seq_lens = all-zero → write_plan_out is filled with write_num_valid_reqs = 0;
    downstream skips canary_write_step.

    Args:
        verify_plan_out: Pre-allocated VerifyPlan; filled in-place.
        write_plan_out: Pre-allocated WritePlan; filled in-place. write_num_valid_reqs = 0 for sweep callers.
        fb_req_pool_indices: ForwardBatch.req_pool_indices; per-row ReqToTokenPool row index, shape [bs],
            int32. 0 is the padding sentinel.
        fb_prefix_lens: Per-req prefix length already written before this step, shape [bs], int32. Caller
            normalizes: extend → ForwardBatch.extend_prefix_lens, decode → ForwardBatch.seq_lens - 1, sweep
            over running → seq_lens.
        fb_extend_seq_lens: ForwardBatch.extend_seq_lens; per-req tokens being written this step, shape [bs],
            int32. 1 for pure decode; 0 for sweep.
        req_to_token: ReqToTokenPool.req_to_token; full-pool slot index table, shape [max_reqs, max_seq_len],
            int32.
        extra_verify_slot_indices: Pre-walked extra verify slots, shape [extra_verify_capacity], int32.
            Caller-translated to the target index space.
        extra_verify_positions: Same shape, int32. Expected position per extra entry.
        extra_verify_prev_slot_indices: Same shape, int32. -1 for chain-seed extras.
        extra_verify_num_valid: Active extra entry count, shape [1], int32. 0 for per-forward and running-sweep
            callers.
        swa_window_size: 0 for the FULL canary group; positive window length for the SWA group.
        full_to_swa_index_mapping: SWA LUT, shape [full_pool_size + 1], int32, or None. Required (non-None) iff
            swa_window_size > 0. Used to translate verify slot indices and chain-seed slot indices at plan time.

    Implementation:
        - Single Triton @triton.jit launch (`canary_plan_kernel`) split into three logical phases inside one
          program:
          1. **Per-req count + seed gather** (1-D grid `(bs,)`, each program = one req): each program reads
             fb_req_pool_indices[r], fb_prefix_lens[r], fb_extend_seq_lens[r], computes verify_count =
             (prefix_lens - window_start) and write_count = extend_seq_lens (both 0 if rp == 0 padding), and
             gathers seed_slot_full = req_to_token[rp, prefix_lens - 1] (or -1 if prefix_lens == 0).
             SWA-translates seed_slot via full_to_swa_index_mapping[seed_slot_full] if non-None.
          2. **Block-level cumsum** (`tl.cumsum(verify_counts, axis=0)` and `tl.cumsum(write_counts, axis=0)`)
             produces verify_offsets[bs+1] and write_plan_out.write_offsets[bs+1] in-place. write_seed slots
             from phase 1 are scattered to write_plan_out.write_seed_slot_indices.
          3. **Per-entry materialization** (2-D logical grid `(bs, max_verify_per_req)`, masked by per-req
             verify_count): for each (r, j) with j < verify_count[r], gather slot =
             req_to_token[fb_req_pool_indices[r], window_start[r] + j] (SWA-translated), prev_slot =
             req_to_token[..., window_start[r] + j - 1] when (window_start[r] + j) > 0 (also translated) else
             -1, position = window_start[r] + j; scatter (slot, position, prev_slot) into verify_plan_out at
             flat index verify_offsets[r] + j.
        - **Extra entries append** (1-D grid `(extra_verify_num_valid,)`): copy extra_verify_*[: num_valid]
          into verify_plan_out.verify_* starting at flat index verify_offsets[bs]. Extras are
          caller-pre-translated, no LUT pass.
        - Scalar writes: verify_plan_out.verify_num_valid = verify_offsets[bs] + extra_verify_num_valid;
          write_plan_out.write_num_valid_reqs = bs (or smaller if padding rows trail). Done by a single program
          at the end.
        - All output tensors are addressed at addresses baked into the cuda-graph capture; phases are launched
          as one kernel (via tl.where / tl.arange branching) so capture sees a single launch per call.

    Calling contract:
        - Pure side-effect; no host work, no D2H.
        - Safe in cuda-graph capture; caller refills all input tensors in-place before replay.
        - Single kernel launch fills both plans end-to-end.
        - Padding rows contribute zero entries.

    Pinned by Python reference
    :func:`sglang.jit_kernel.kv_cache_canary_plan_ref.canary_plan_step_torch_reference`; Triton must match
    byte-for-byte.
    """
    raise NotImplementedError(
        "Use canary_plan_step_torch_reference until Triton kernel lands"
    )

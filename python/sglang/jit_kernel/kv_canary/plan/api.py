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


def launch_canary_plan_kernels(
    *,
    verify_plan_out: VerifyPlan,
    write_plan_out: WritePlan,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    verify_capacity: int,
    req_to_verify_expected_tokens: Optional[torch.Tensor],
    req_to_verify_expected_tokens_valid_lens: Optional[torch.Tensor],
    kv_token_id_vs_position_offset: int,
) -> None:
    """Fill verify_plan_out + write_plan_out from normalized canary plan inputs.

    For each req r with req_pool_indices[r] != 0 (0 = padding sentinel):

    - **Verify entries**: one per pos in [window_start, prefix_lens[r]), where window_start = max(0,
      prefix_lens[r] - swa_window_size) if SWA else 0. slot_idx = req_to_token[req_pool_indices[r], pos]
      (SWA-translated via full_to_swa_index_mapping if non-None); prev_slot_idx =
      req_to_token[req_pool_indices[r], pos-1] for pos > 0, else -1. (SWA windows do NOT reset the chain —
      the writer chains across the entire prefix; sweep verify within an SWA window dereferences the real
      predecessor for chain-link reconstruction.) Expected-token gather: when
      ``req_to_verify_expected_tokens`` is supplied, ``expected_input_id =
      req_to_verify_expected_tokens[rp, pos + kv_token_id_vs_position_offset]`` when ``0 <= pos +
      kv_token_id_vs_position_offset < req_to_verify_expected_tokens_valid_lens[r]``, else the ``-1``
      sentinel (which the verify kernel treats as "skip token-id check").
    - **Write metadata** (when extend_seq_lens[r] > 0): contribute extend_seq_lens[r] to the per-req
      write count (for write_offsets cumsum). Per-req chain seed = req_to_token[req_pool_indices[r],
      prefix_lens[r]-1] (SWA-translated), or -1 if prefix_lens[r] == 0. Per-token write data
      (input_ids / positions / out_cache_loc) is NOT materialized here — launch_canary_write_kernel
      reads it directly from ForwardBatch via write_offsets.

    Args:
        verify_plan_out: Pre-allocated VerifyPlan; filled in-place.
        write_plan_out: Pre-allocated WritePlan; filled in-place.
        req_pool_indices: Per-row ReqToTokenPool row index, shape [bs], int64. 0 is the padding sentinel.
        prefix_lens: Per-req prefix length already written before this step, shape [bs], int64.
        extend_seq_lens: Per-req tokens being written this step, shape [bs], int64.
        req_to_token: ReqToTokenPool.req_to_token; full-pool slot index table, shape [max_reqs, max_seq_len],
            int32.
        swa_window_size: 0 for the FULL canary group; positive window length for the SWA group.
        full_to_swa_index_mapping: SWA LUT, shape [full_pool_size + 1], int64, or None. Required (non-None) iff
            swa_window_size > 0. Used to translate verify slot indices and chain-seed slot indices at plan time.
            Loaded element-typed via Triton ``tl.load``; intermediate translated slot values are int64 inside the
            kernel and stored in the int64 plan schema.
        verify_capacity: Length of verify_plan_out.verify_*; on overflow the offsets kernel clears
            verify_enable and plan_entries skips the scatter.
        req_to_verify_expected_tokens: Optional source-of-truth token pool, shape [max_reqs, max_context_len],
            int32. When supplied, the plan kernel gathers expected_input_id for each verify entry from
            ``[rp, pos + kv_token_id_vs_position_offset]``; when None, every entry gets the ``-1`` sentinel.
        req_to_verify_expected_tokens_valid_lens: Per-req snapshot length on ``req_to_verify_expected_tokens``,
            shape [bs], int64. Required iff ``req_to_verify_expected_tokens`` is set. Reads past
            ``valid_lens[r]`` skip the gather (emit ``-1``) — this is what makes the plan kernel correct in the
            presence of EAGLE draft / verify positions written past the committed history, and across pool
            rows recycled from a longer previous owner whose stale tail still lives at high indices.
        kv_token_id_vs_position_offset: Per-buffer-group logical-position offset applied to ``pos`` before
            indexing ``req_to_verify_expected_tokens``. 0 for target pools; +1 for EAGLE draft.

    Implementation:
        - Two sub-kernels launched in sequence:
          1. Triton ``_plan_offsets_kernel`` (1-D grid ``(1,)``, single program over all ``bs`` reqs):
             reads req_pool_indices[r], prefix_lens[r], extend_seq_lens[r] for each r; computes
             verify_count = (prefix_lens - window_start) and write_count = extend_seq_lens (both 0 if rp == 0
             padding); gathers seed_slot_full = req_to_token[rp, prefix_lens - 1] (or -1 if prefix_lens == 0),
             SWA-translates seed_slot via full_to_swa_index_mapping[seed_slot_full] if non-None; runs
             block-level cumsum (``tl.cumsum``) to produce verify_offsets[_PLAN_BS_BLOCK_SIZE + 1] and
             write_plan_out.write_offsets[write_req_capacity + 1] in-place; scatters write_seed slots; writes
             scalar totals ``verify_plan_out.verify_num_valid`` and ``write_plan_out.write_num_valid_reqs``.
          2. CUDA ``plan_entries_persistent_kernel`` (1-D persistent grid sized to ``num_sms *
             kBlocksPerSm`` blocks of ``kBlockSize`` threads), wrapped by Python
             ``launch_plan_entries_kernel``: each thread grid-strides over ``tid ∈ [0, total_verify)``,
             locates its owning req via ``find_req_id`` (binary search on verify_offsets), computes
             out_position = window_start[req_id] + (tid - verify_offsets[req_id]), gathers slot =
             req_to_token[rp, out_position] (SWA-translated when ``HAS_SWA_LUT``), prev_slot =
             req_to_token[rp, out_position - 1] when out_position > 0 (also translated) else -1, and
             scatters (slot, position, prev_slot) into verify_plan_out at flat index tid.
        - All output tensors are addressed at addresses baked into the cuda-graph capture.

    Calling contract:
        - Pure side-effect; no host work, no D2H.
        - Safe in cuda-graph capture; caller refills all input tensors in-place before replay.
        - The wrapper launches the plan sub-kernels needed to fill both plans end-to-end.
        - Padding rows contribute zero entries.

    Pinned by Python reference
    :func:`sglang.jit_kernel.kv_canary.plan_ref.launch_canary_plan_kernels_torch_reference`; both the Triton
    offsets kernel and the CUDA JIT entries kernel must match byte-for-byte.
    """
    bs = int(req_pool_indices.shape[0])
    if bs > _PLAN_BS_BLOCK_SIZE:
        raise ValueError(
            f"kv-canary: launch_canary_plan_kernels supports at most bs={_PLAN_BS_BLOCK_SIZE} reqs per launch, "
            f"got bs={bs}. Bump _PLAN_BS_BLOCK_SIZE if real workloads need this."
        )
    if swa_window_size > 0 and full_to_swa_index_mapping is None:
        raise ValueError(
            "kv-canary: launch_canary_plan_kernels requires full_to_swa_index_mapping when swa_window_size > 0"
        )

    device = verify_plan_out.verify_slot_indices.device
    verify_offsets_scratch = torch.empty(
        _PLAN_BS_BLOCK_SIZE + 1, dtype=torch.int64, device=device
    )

    plan_verify_capacity = int(verify_plan_out.verify_slot_indices.shape[0])
    if verify_capacity != plan_verify_capacity:
        raise ValueError(
            f"kv-canary: launch_canary_plan_kernels verify_capacity={verify_capacity} does not match "
            f"verify_plan_out.verify_slot_indices.shape[0]={plan_verify_capacity}"
        )

    write_plan_out.write_offsets.zero_()

    launch_plan_offsets_kernel(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
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

    launch_plan_entries_kernel(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_offsets_scratch=verify_offsets_scratch,
        verify_enable=verify_plan_out.enable,
        req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        req_to_verify_expected_tokens_valid_lens=req_to_verify_expected_tokens_valid_lens,
        out_verify_slot_indices=verify_plan_out.verify_slot_indices,
        out_verify_expected_tokens=verify_plan_out.verify_expected_tokens,
        out_verify_expected_positions=verify_plan_out.verify_expected_positions,
        out_verify_prev_slot_indices=verify_plan_out.verify_prev_slot_indices,
        kv_token_id_vs_position_offset=int(kv_token_id_vs_position_offset),
        swa_window_size=int(swa_window_size),
    )

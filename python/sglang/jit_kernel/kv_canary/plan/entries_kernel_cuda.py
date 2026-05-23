"""CUDA persistent-kernel launcher for the kv_canary plan-entries step.

Replaces the per-tile Triton kernel in ``entries_kernel.py`` with a 1-D persistent CUDA kernel: each
thread services one verify entry (binary-searching the offsets prefix-sum to find its req_id), and the
grid is fixed at ``num_sms * blocks_per_sm`` blocks so the launch fits inside a cuda graph without any
host-side feedback. See ``csrc/kv_canary/plan_entries.cuh`` for the kernel body and the plan note at
``lab/docs/pkgs/sglang/notes/2026-05-23-cuda-persistent-plan-entries-kernel.md`` for the algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_plan_entries_module() -> "Module":
    """Compile and cache the JIT kv_canary plan-entries CUDA module.

    No template specialization: the persistent kernel already branches on ``HAS_SWA_LUT`` at compile
    time inside the wrapper (two kernel instantiations baked into the host launcher), so the Python
    side only needs one module.
    """
    return load_jit(
        "kv_canary_plan_entries",
        cuda_files=["kv_canary/plan_entries.cuh"],
        cuda_wrappers=[
            (
                "plan_entries",
                "canary_plan_entries::PlanEntriesKernel::run",
            ),
        ],
    )


def launch_plan_entries_cuda(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    verify_offsets_scratch: torch.Tensor,
    out_verify_slot_indices: torch.Tensor,
    out_verify_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    swa_window_size: int,
) -> None:
    """Launch the CUDA persistent plan-entries kernel.

    All tensors and dtypes must match what ``launch_plan_entries_kernel`` (Triton) accepts; the input /
    output ABI is identical so the Triton and CUDA paths are drop-in interchangeable.

    Args:
        req_pool_indices: ``[bs_padded]`` int64. Rows past the actual ``bs`` carry REQ_POOL_IDX_PADDING.
        prefix_lens: ``[bs_padded]`` int64. Padding rows carry 0.
        req_to_token: ``[max_reqs, max_seq_len]`` int32. Row 0 is the padding row.
        full_to_swa_index_mapping: ``[lut_len]`` int64 or None. Required iff ``swa_window_size > 0``.
        verify_offsets_scratch: ``[bs_padded + 1]`` int64. Exclusive prefix-sum of per-req verify counts;
            ``verify_offsets_scratch[bs_padded]`` is the total number of verify entries this launch.
        out_verify_slot_indices: ``[verify_capacity]`` int64. Output.
        out_verify_positions: ``[verify_capacity]`` int64. Output.
        out_verify_prev_slot_indices: ``[verify_capacity]`` int64. Output.
        swa_window_size: 0 for the FULL pool; positive for the SWA pool.
    """
    bs_padded = int(req_pool_indices.shape[0])
    if bs_padded == 0:
        return

    req_to_token_stride0 = int(req_to_token.stride(0))

    module = _jit_plan_entries_module()
    module.plan_entries(
        req_pool_indices,
        prefix_lens,
        req_to_token,
        full_to_swa_index_mapping,
        verify_offsets_scratch,
        out_verify_slot_indices,
        out_verify_positions,
        out_verify_prev_slot_indices,
        req_to_token_stride0,
        bs_padded,
        int(swa_window_size),
    )

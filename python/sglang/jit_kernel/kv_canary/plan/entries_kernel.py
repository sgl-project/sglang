from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_plan_entries_module() -> "Module":
    """Compile and cache the JIT kv_canary plan-entries CUDA module."""
    return load_jit(
        "kv_canary_plan_entries",
        cuda_files=["kv_canary/plan_entries.cuh"],
        cuda_wrappers=[
            ("plan_entries", "PlanEntriesKernel::run"),
        ],
    )


def launch_plan_entries_kernel(
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
        int(req_to_token.stride(0)),
        int(req_pool_indices.shape[0]),
        int(swa_window_size),
    )

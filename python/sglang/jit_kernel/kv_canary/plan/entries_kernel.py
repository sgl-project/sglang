from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_plan_entries_module() -> "Module":
    return load_jit(
        "kv_canary_plan_entries",
        cuda_files=["kv_canary/canary_plan_entries.cuh"],
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
    verify_enable: torch.Tensor,
    expected_token_pool: Optional[torch.Tensor],
    expected_token_valid_lens: Optional[torch.Tensor],
    out_verify_slot_indices: torch.Tensor,
    out_verify_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    out_verify_expected_tokens: torch.Tensor,
    swa_window_size: int,
    slot_token_offset: int,
) -> None:
    """Scatter per-entry verify plan rows.

    Both ``expected_token_pool`` and ``expected_token_valid_lens`` must be either both set or both
    ``None``; when ``None`` the kernel writes the ``-1`` "skip token check" sentinel into
    ``out_verify_expected_tokens`` for every active entry.
    """
    if (expected_token_pool is None) != (expected_token_valid_lens is None):
        raise ValueError(
            "kv-canary: expected_token_pool and expected_token_valid_lens must be both None or both set"
        )
    module = _jit_plan_entries_module()
    module.plan_entries(
        req_pool_indices,
        prefix_lens,
        req_to_token,
        full_to_swa_index_mapping,
        verify_offsets_scratch,
        verify_enable,
        expected_token_pool,
        expected_token_valid_lens,
        out_verify_slot_indices,
        out_verify_positions,
        out_verify_prev_slot_indices,
        out_verify_expected_tokens,
        int(swa_window_size),
        int(slot_token_offset),
    )

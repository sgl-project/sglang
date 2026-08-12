from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_plan_entries_module(
    has_swa_lut: bool, has_verify_expected_token_pool: bool
) -> Module:
    args = make_cpp_args(has_swa_lut, has_verify_expected_token_pool)
    return load_jit(
        "kv_canary_plan_entries",
        *args,
        cuda_files=["kv_canary/canary_plan_entries.cuh"],
        cuda_wrappers=[
            ("plan_entries", f"PlanEntriesKernel<{args}>::run"),
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
    req_to_verify_expected_tokens: Optional[torch.Tensor],
    req_to_verify_expected_tokens_valid_lens: Optional[torch.Tensor],
    out_verify_slot_indices: torch.Tensor,
    out_verify_expected_tokens: torch.Tensor,
    out_verify_expected_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    kv_token_id_vs_position_offset: int,
    swa_window_size: int,
) -> None:
    has_swa_lut = full_to_swa_index_mapping is not None
    has_verify_expected_token_pool = req_to_verify_expected_tokens is not None
    if (
        has_verify_expected_token_pool
        and req_to_verify_expected_tokens_valid_lens is None
    ):
        raise ValueError(
            "kv-canary: launch_plan_entries_kernel requires "
            "req_to_verify_expected_tokens_valid_lens when req_to_verify_expected_tokens is set"
        )
    module = _jit_plan_entries_module(has_swa_lut, has_verify_expected_token_pool)
    module.plan_entries(
        req_pool_indices,
        prefix_lens,
        req_to_token,
        full_to_swa_index_mapping,
        verify_offsets_scratch,
        verify_enable,
        req_to_verify_expected_tokens,
        req_to_verify_expected_tokens_valid_lens,
        out_verify_slot_indices,
        out_verify_expected_tokens,
        out_verify_expected_positions,
        out_verify_prev_slot_indices,
        int(kv_token_id_vs_position_offset),
        int(swa_window_size),
    )

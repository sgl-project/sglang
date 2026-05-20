from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvHashMode,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode as CanaryInputCheckMode
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.srt.kv_canary.violation_state import ViolationLog

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def invoke_plan(
    *,
    plan_input: PlanInput,
    verify_plan: VerifyPlan,
    write_plan: WritePlan,
    group: CanaryBufferGroup,
    req_to_token: torch.Tensor,
    swa_window_size: int,
) -> None:
    """Invoke canary_plan_step for one (plan_input x buffer group) pair."""
    window = swa_window_size if group.kind is PoolKind.SWA else 0
    canary_plan_step(
        verify_plan_out=verify_plan,
        write_plan_out=write_plan,
        fb_req_pool_indices=plan_input.fb_req_pool_indices,
        fb_prefix_lens=plan_input.fb_prefix_lens,
        fb_extend_seq_lens=plan_input.fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=plan_input.extra_verify_slot_indices,
        extra_verify_positions=plan_input.extra_verify_positions,
        extra_verify_prev_slot_indices=plan_input.extra_verify_prev_slot_indices,
        extra_verify_num_valid=plan_input.extra_verify_num_valid,
        swa_window_size=window,
        full_to_swa_index_mapping=group.swa_index_lut,
    )


def launch_endpoints_per_forward(
    *,
    endpoints: tuple[CanaryEndpoint, ...],
    group: CanaryBufferGroup,
    tag_filter: Callable[[CanaryLaunchTag], bool],
    verify_plan: VerifyPlan,
    write_plan: WritePlan,
    forward_batch: "ForwardBatch",
    expected_input_tokens: torch.Tensor,
    expected_input_positions: torch.Tensor,
    violation_log: ViolationLog,
    real_kv_hash_mode: RealKvHashMode,
    input_check_mode: CanaryInputCheckMode,
) -> None:
    """Per-forward HEAD/TAIL endpoint launches. Iterates endpoints, applies tag_filter, drops
    SWEEP_* tags.
    """
    positions = forward_batch.positions
    if positions.dtype != torch.int32:
        positions = positions.to(torch.int32)
    out_cache_loc = forward_batch.out_cache_loc
    if out_cache_loc is not None and out_cache_loc.dtype != torch.int32:
        out_cache_loc = out_cache_loc.to(torch.int32)
    input_ids = forward_batch.input_ids
    if input_ids is not None and input_ids.dtype != torch.int32:
        input_ids = input_ids.to(torch.int32)

    num_tokens = int(positions.shape[0])
    expected_tokens_slice = expected_input_tokens[:num_tokens]
    expected_positions_slice = expected_input_positions[:num_tokens]

    for endpoint in endpoints:
        if not _endpoint_belongs_to_group(endpoint, group):
            continue
        if not tag_filter(endpoint.kernel_kind):
            continue
        if _is_sweep_tag(endpoint.kernel_kind):
            continue
        endpoint.launch_per_forward(
            verify_plan=verify_plan,
            write_plan=write_plan,
            fb_input_ids=input_ids,
            fb_positions=positions,
            fb_out_cache_loc=out_cache_loc,
            input_check_mode=input_check_mode,
            expected_input_tokens=expected_tokens_slice,
            expected_input_positions=expected_positions_slice,
            violation_log=violation_log,
            real_kv_hash_mode=real_kv_hash_mode,
        )


def launch_endpoints_sweep(
    *,
    endpoints: tuple[CanaryEndpoint, ...],
    group: CanaryBufferGroup,
    verify_plan: VerifyPlan,
    violation_log: ViolationLog,
    real_kv_hash_mode: RealKvHashMode,
) -> None:
    """Sweep endpoint launches. Iterates endpoints, keeps only SWEEP_* tags for the given group."""
    for endpoint in endpoints:
        if not _endpoint_belongs_to_group(endpoint, group):
            continue
        if not _is_sweep_tag(endpoint.kernel_kind):
            continue
        endpoint.launch_sweep(
            verify_plan=verify_plan,
            violation_log=violation_log,
            real_kv_hash_mode=real_kv_hash_mode,
        )


def _is_sweep_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.SWEEP_K_FULL,
        CanaryLaunchTag.SWEEP_V_FULL,
        CanaryLaunchTag.SWEEP_K_SWA,
        CanaryLaunchTag.SWEEP_V_SWA,
    )


def _endpoint_belongs_to_group(
    endpoint: CanaryEndpoint, group: CanaryBufferGroup
) -> bool:
    suffix = endpoint.kernel_kind.name.rsplit("_", 1)[1]
    return suffix == group.kind.name

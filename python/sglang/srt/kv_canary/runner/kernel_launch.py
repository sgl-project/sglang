from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.jit_kernel.kv_canary.consts import (
    RealKvHashMode,
)
from sglang.jit_kernel.kv_canary.plan import launch_canary_plan_kernels
from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.srt.kv_canary.state import ViolationLog

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_INPUT_IDS = "forward_batch.input_ids"
_OUT_LOC = "forward_batch.out_cache_loc"
_POSITIONS = "forward_batch.positions"


def _assert_boundary_int64(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Assert (not convert) the boundary tensor is int64 + contiguous.

    The previous helper performed ``tensor.to(torch.int64).contiguous()``,
    which silently allocates fresh storage inside the cuda-graph capture
    region when the source dtype/layout differs. SFM-era phase 2/3 must be
    capture-safe, so we now require upstream callers (phase 1 outside the
    graph) to deliver int64 contiguous tensors and only assert here.
    """
    assert tensor.dtype == torch.int64, (
        f"kv-canary: {name} must have dtype torch.int64, got {tensor.dtype}; "
        "the upstream phase-1 hook must canonicalize before entering the "
        "cuda-graph region"
    )
    assert tensor.is_contiguous(), (
        f"kv-canary: {name} must be contiguous, got non-contiguous; "
        "the upstream phase-1 hook must canonicalize before entering the "
        "cuda-graph region"
    )
    return tensor


def invoke_plan(
    *,
    plan_input: PlanInput,
    verify_plan: VerifyPlan,
    write_plan: WritePlan,
    group: CanaryBufferGroup,
    req_to_token: torch.Tensor,
    swa_window_size: int,
) -> None:
    window = swa_window_size if group.kind is PoolKind.SWA else 0
    launch_canary_plan_kernels(
        verify_plan_out=verify_plan,
        write_plan_out=write_plan,
        req_pool_indices=plan_input.req_pool_indices,
        prefix_lens=plan_input.prefix_lens,
        extend_seq_lens=plan_input.extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=window,
        full_to_swa_index_mapping=group.swa_index_lut,
        verify_capacity=int(verify_plan.verify_slot_indices.shape[0]),
    )


def launch_endpoints_per_forward(
    *,
    endpoints: tuple[CanaryEndpoint, ...],
    group: CanaryBufferGroup,
    tag_filter: Callable[[CanaryLaunchTag], bool],
    verify_plan: VerifyPlan,
    write_plan: WritePlan,
    forward_batch: "ForwardBatch",
    expected_inputs: ExpectedInputs,
    violation_log: ViolationLog,
    real_kv_hash_mode: RealKvHashMode,
    input_check_mode: bool,
) -> None:
    positions = _assert_boundary_int64(forward_batch.positions, _POSITIONS)
    out_cache_loc = _assert_boundary_int64(forward_batch.out_cache_loc, _OUT_LOC)
    input_ids = _assert_boundary_int64(forward_batch.input_ids, _INPUT_IDS)

    num_tokens = int(positions.shape[0])
    if expected_inputs.tokens.shape[0] != num_tokens:
        raise RuntimeError(
            f"kv-canary: expected_inputs.tokens shape {expected_inputs.tokens.shape[0]} "
            f"!= num_tokens {num_tokens}; caller must slice before invoking"
        )
    if expected_inputs.positions.shape[0] != num_tokens:
        raise RuntimeError(
            f"kv-canary: expected_inputs.positions shape {expected_inputs.positions.shape[0]} "
            f"!= num_tokens {num_tokens}; caller must slice before invoking"
        )

    active_endpoints = [
        endpoint
        for endpoint in endpoints
        if _endpoint_belongs_to_group(endpoint, group)
        and tag_filter(endpoint.kernel_kind)
        and not _is_sweep_tag(endpoint.kernel_kind)
    ]
    assert len(active_endpoints) > 0

    for endpoint in active_endpoints:
        endpoint.launch_per_forward(
            verify_plan=verify_plan,
            write_plan=write_plan,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            input_check_mode=input_check_mode,
            expected_inputs=expected_inputs,
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
    active_endpoints = [
        endpoint
        for endpoint in endpoints
        if _endpoint_belongs_to_group(endpoint, group)
        and _is_sweep_tag(endpoint.kernel_kind)
    ]
    assert len(active_endpoints) > 0

    for endpoint in active_endpoints:
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

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.kv_canary.utils import (
    RING_CAPACITY,
    SWA_WINDOW,
    BenchCase,
    build_fast_matrix_cases,
    build_full_matrix_cases,
    cases_to_x_vals,
    make_real_kv_sources,
    naive_slot_copy_fn,
)
from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    CanaryLaunchTag,
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
    launch_canary_verify_kernel,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=900, suite="nightly-kernel-1-gpu", nightly=True)
# AMD mirrors the CUDA nightly registration (nightly-only, no per-PR suite).
# Note: amd_ci_exec.sh sets SGLANG_IS_IN_CI, so this runs the CI-reduced range
# (build_fast_matrix_cases via get_benchmark_range), same as CUDA nightly.
register_amd_ci(est_time=900, suite="nightly-amd-kernel-1-gpu", nightly=True)


_X_NAMES = [
    "scenario",
    "bs",
    "prefix_len",
    "mode",
    "extend_len",
    "pool_kind",
    "real_kv_kind",
    "hash_mode",
]
_X_VALS = cases_to_x_vals(
    get_benchmark_range(
        full_range=build_full_matrix_cases(),
        ci_range=build_fast_matrix_cases(),
    )
)

_KERNEL_KIND_X_NAMES = ["kernel_kind_name"]
_KERNEL_KIND_X_VALS = [(tag.name,) for tag in CanaryLaunchTag]


def _verify_entry_count(case: BenchCase) -> int:
    if case.pool_kind == "swa_window_128":
        per_req = min(case.prefix_len, SWA_WINDOW)
    else:
        per_req = case.prefix_len
    return case.bs * per_req


def _verify_num_slots(case: BenchCase) -> int:
    if case.pool_kind == "swa_window_128":
        per_req_slots = SWA_WINDOW
    else:
        per_req_slots = max(1, case.prefix_len)
    return max(2, case.bs * per_req_slots + 1)


def _build_verify_inputs(case: BenchCase, *, device: torch.device) -> Tuple[
    torch.Tensor,
    VerifyPlan,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[RealKvSource, ...],
]:
    total_entries = _verify_entry_count(case)
    capacity = max(1, total_entries)
    num_slots = _verify_num_slots(case)

    canary_buf = torch.zeros(
        num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
    )

    slot_indices = torch.empty(capacity, dtype=torch.int64, device=device)
    positions = torch.empty(capacity, dtype=torch.int64, device=device)
    prev_slots = torch.empty(capacity, dtype=torch.int64, device=device)
    if total_entries > 0:
        flat_idx = torch.arange(total_entries, device=device, dtype=torch.int64)
        per_req = total_entries // case.bs if case.bs > 0 else 0
        slot_indices[:total_entries] = (flat_idx % max(num_slots - 1, 1)).to(
            torch.int64
        )
        positions[:total_entries] = (flat_idx % max(per_req, 1)).to(torch.int64)
        is_head = (flat_idx % max(per_req, 1)) == 0
        prev_seq = (flat_idx - 1) % max(num_slots - 1, 1)
        prev_slots[:total_entries] = torch.where(
            is_head, torch.full_like(flat_idx, -1), prev_seq
        ).to(torch.int64)
    if capacity > total_entries:
        slot_indices[total_entries:] = 0
        positions[total_entries:] = 0
        prev_slots[total_entries:] = -1

    num_valid = torch.tensor([total_entries], dtype=torch.int32, device=device)
    enable = torch.ones(1, dtype=torch.int32, device=device)
    expected_input_ids = torch.full((capacity,), -1, dtype=torch.int64, device=device)
    plan = VerifyPlan(
        verify_slot_indices=slot_indices,
        verify_expected_tokens=expected_input_ids,
        verify_expected_positions=positions,
        verify_prev_slot_indices=prev_slots,
        verify_num_valid=num_valid,
        enable=enable,
    )

    violation_ring = torch.zeros(
        RING_CAPACITY, consts.VIOLATION_FIELDS, dtype=torch.int64, device=device
    )
    violation_write_index = torch.zeros(1, dtype=torch.int32, device=device)
    slot_run_counter = torch.zeros(1, dtype=torch.int64, device=device)
    kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=device)
    enable_chain_position_assert = torch.ones(1, dtype=torch.int32, device=device)

    real_kv_sources = make_real_kv_sources(
        kind=case.real_kv_kind, num_slots=num_slots, device=device
    )

    return (
        canary_buf,
        plan,
        violation_ring,
        violation_write_index,
        slot_run_counter,
        kernel_run_counter,
        enable_chain_position_assert,
        real_kv_sources,
    )


def _build_context(
    *,
    canary_buf: torch.Tensor,
    violation_ring: torch.Tensor,
    violation_write_index: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    enable_chain_position_assert: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    kernel_kind: CanaryLaunchTag,
    hash_mode: consts.RealKvHashMode,
) -> VerifyOrWriteContext:
    return VerifyOrWriteContext(
        canary_buf=canary_buf,
        kernel_kind=kernel_kind,
        violation_ring=violation_ring,
        violation_write_index=violation_write_index,
        slot_run_counter=slot_run_counter,
        kernel_run_counter=kernel_run_counter,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=hash_mode,
        enable_chain_position_assert=enable_chain_position_assert,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_X_NAMES,
        x_vals=_X_VALS,
        line_arg="provider",
        line_vals=["canary", "naive"],
        line_names=["canary_verify_step", "naive index_copy_"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="kv-canary-verify-perf",
        args={},
    )
)
def benchmark(
    scenario: str,
    bs: int,
    prefix_len: int,
    mode: str,
    extend_len: int,
    pool_kind: str,
    real_kv_kind: str,
    hash_mode: str,
    provider: str,
) -> Tuple[float, float, float]:
    case = BenchCase(
        scenario=scenario,
        bs=bs,
        prefix_len=prefix_len,
        mode=mode,
        extend_len=extend_len,
        pool_kind=pool_kind,
        real_kv_kind=real_kv_kind,
        hash_mode=hash_mode,
    )
    device = torch.device(DEFAULT_DEVICE)

    if provider == "canary":
        (
            canary_buf,
            plan,
            violation_ring,
            violation_write_index,
            slot_run_counter,
            kernel_run_counter,
            enable_chain_position_assert,
            real_kv_sources,
        ) = _build_verify_inputs(case, device=device)
        hash_mode_enum = consts.RealKvHashMode[case.hash_mode.upper()]
        context = _build_context(
            canary_buf=canary_buf,
            violation_ring=violation_ring,
            violation_write_index=violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            enable_chain_position_assert=enable_chain_position_assert,
            real_kv_sources=real_kv_sources,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            hash_mode=hash_mode_enum,
        )

        def fn() -> None:
            violation_write_index.zero_()
            launch_canary_verify_kernel(
                context=context,
                plan=plan,
                check_verify_expected_token=True,
            )

    else:
        fn = naive_slot_copy_fn(total=_verify_entry_count(case), device=device)

    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_KERNEL_KIND_X_NAMES,
        x_vals=_KERNEL_KIND_X_VALS,
        line_arg="provider",
        line_vals=["canary"],
        line_names=["canary_verify_step"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="kv-canary-verify-kernel-kind-perf",
        args={},
    )
)
def benchmark_kernel_kind(
    kernel_kind_name: str,
    provider: str,
) -> Tuple[float, float, float]:
    case = BenchCase(
        scenario="kernel_kind",
        bs=32,
        prefix_len=4096,
        mode="extend",
        extend_len=128,
        pool_kind="full",
        real_kv_kind="none",
        hash_mode="none",
    )
    device = torch.device(DEFAULT_DEVICE)

    (
        canary_buf,
        plan,
        violation_ring,
        violation_write_index,
        slot_run_counter,
        kernel_run_counter,
        enable_chain_position_assert,
        real_kv_sources,
    ) = _build_verify_inputs(case, device=device)
    kernel_kind = CanaryLaunchTag[kernel_kind_name]
    hash_mode_enum = consts.RealKvHashMode[case.hash_mode.upper()]
    context = _build_context(
        canary_buf=canary_buf,
        violation_ring=violation_ring,
        violation_write_index=violation_write_index,
        slot_run_counter=slot_run_counter,
        kernel_run_counter=kernel_run_counter,
        enable_chain_position_assert=enable_chain_position_assert,
        real_kv_sources=real_kv_sources,
        kernel_kind=kernel_kind,
        hash_mode=hash_mode_enum,
    )

    def fn() -> None:
        violation_write_index.zero_()
        launch_canary_verify_kernel(
            context=context,
            plan=plan,
            check_verify_expected_token=True,
        )

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
    benchmark_kernel_kind.run(print_data=True)

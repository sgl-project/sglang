"""Sweep-matrix benchmark for canary_verify_step.

Cartesian product over (bs, prefix_len, mode, pool_kind). Fast subset runs by default; full
cartesian product is gated behind ``--runslow`` / ``--bench-full`` via this directory's
``conftest.py``.

Per case the bench reports: name, microseconds per call, nanoseconds per processed verify slot,
and the ratio against a naive ``kv_buf[slot] = payload`` baseline of the same total slot count.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.jit_kernel.benchmark.kv_canary.bench_helpers import (
    RING_CAPACITY,
    SWA_WINDOW,
    BenchCase,
    baseline_us_slot_copy,
    do_bench,
    select_matrix_cases,
)
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    VIOLATION_FIELDS,
    CanaryLaunchTag,
    RealKvHashMode,
    VerifyPlan,
    canary_verify_step,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")
register_cuda_ci(est_time=900, suite="nightly-kernel-1-gpu", nightly=True)


def _verify_entry_count(case: BenchCase) -> int:
    if case.pool_kind == "swa_window_128":
        per_req = min(case.prefix_len, SWA_WINDOW)
    else:
        per_req = case.prefix_len
    return case.bs * per_req


def _build_verify_inputs(
    case: BenchCase, *, device: torch.device
) -> tuple[
    torch.Tensor, VerifyPlan, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    total_entries = _verify_entry_count(case)
    capacity = max(1, total_entries)

    if case.pool_kind == "swa_window_128":
        per_req_slots = SWA_WINDOW
    else:
        per_req_slots = max(1, case.prefix_len)
    num_slots = max(2, case.bs * per_req_slots + 1)

    canary_buf = torch.zeros(
        num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
    )

    slot_indices = torch.empty(capacity, dtype=torch.int32, device=device)
    positions = torch.empty(capacity, dtype=torch.int32, device=device)
    prev_slots = torch.empty(capacity, dtype=torch.int32, device=device)
    if total_entries > 0:
        flat_idx = torch.arange(total_entries, device=device, dtype=torch.int64)
        per_req = total_entries // case.bs if case.bs > 0 else 0
        slot_indices[:total_entries] = (flat_idx % max(num_slots - 1, 1)).to(
            torch.int32
        )
        positions[:total_entries] = (flat_idx % max(per_req, 1)).to(torch.int32)
        is_head = (flat_idx % max(per_req, 1)) == 0
        prev_seq = (flat_idx - 1) % max(num_slots - 1, 1)
        prev_slots[:total_entries] = torch.where(
            is_head, torch.full_like(flat_idx, -1), prev_seq
        ).to(torch.int32)
    if capacity > total_entries:
        slot_indices[total_entries:] = 0
        positions[total_entries:] = 0
        prev_slots[total_entries:] = -1

    num_valid = torch.tensor([total_entries], dtype=torch.int32, device=device)
    plan = VerifyPlan(
        verify_slot_indices=slot_indices,
        verify_positions=positions,
        verify_prev_slot_indices=prev_slots,
        verify_num_valid=num_valid,
    )

    violation_ring = torch.zeros(
        RING_CAPACITY, VIOLATION_FIELDS, dtype=torch.int64, device=device
    )
    violation_write_index = torch.zeros(1, dtype=torch.int32, device=device)
    slot_run_counter = torch.zeros(1, dtype=torch.int64, device=device)
    kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=device)

    return (
        canary_buf,
        plan,
        violation_ring,
        violation_write_index,
        slot_run_counter,
        kernel_run_counter,
    )


def _run_one_case(case: BenchCase) -> dict:
    device = torch.device(DEFAULT_DEVICE)
    (
        canary_buf,
        plan,
        violation_ring,
        violation_write_index,
        slot_run_counter,
        kernel_run_counter,
    ) = _build_verify_inputs(case, device=device)

    def fn() -> None:
        canary_verify_step(
            canary_buf=canary_buf,
            plan=plan,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=violation_ring,
            violation_write_index=violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )

    fn()
    torch.cuda.synchronize()

    canary_us = do_bench(fn)
    baseline_us = baseline_us_slot_copy(total=_verify_entry_count(case), device=device)
    total_entries = _verify_entry_count(case)
    per_slot_ns = (canary_us * 1000.0 / total_entries) if total_entries > 0 else 0.0
    ratio = canary_us / baseline_us if baseline_us > 0 else float("inf")

    return {
        "name": case.case_id,
        "us_per_call": canary_us,
        "per_slot_ns": per_slot_ns,
        "ratio": ratio,
    }


@pytest.fixture(scope="module", autouse=True)
def _print_header():
    print(
        "\n[bench_kv_canary_verify] name | us/call | per_slot_ns | ratio_vs_naive_write",
        flush=True,
    )
    yield


@pytest.mark.parametrize("case", select_matrix_cases())
def test_bench_verify(case: BenchCase) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    result = _run_one_case(case)
    print(
        f"  {result['name']} | {result['us_per_call']:.3f} | "
        f"{result['per_slot_ns']:.3f} | {result['ratio']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

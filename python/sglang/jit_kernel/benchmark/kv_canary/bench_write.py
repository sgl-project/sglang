"""Sweep-matrix benchmark for canary_write_step.

Cartesian product over (bs, prefix_len, mode, pool_kind). Fast subset runs by default; full slow
subset gated behind ``--runslow`` / ``--bench-full`` via this directory's ``conftest.py``.

Per case the bench reports: name, microseconds per call, nanoseconds per processed write slot, and
the ratio against a naive ``kv_buf[slot] = payload`` baseline of the same total slot count.
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
)
from sglang.jit_kernel.kv_canary.write import (
    CanaryPseudoMode,
    WritePlan,
    canary_write_step,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")
register_cuda_ci(est_time=900, suite="nightly-kernel-1-gpu", nightly=True)


def _write_entry_count(case: BenchCase) -> int:
    return case.bs * case.extend_len


def _build_write_inputs(case: BenchCase, *, device: torch.device) -> dict:
    total_entries = _write_entry_count(case)
    num_tokens_padded = max(1, total_entries)

    per_req_slots = max(
        SWA_WINDOW if case.pool_kind == "swa_window_128" else 1,
        case.prefix_len + case.extend_len,
    )
    num_slots = max(2, case.bs * per_req_slots + 1)

    canary_buf = torch.zeros(
        num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
    )

    write_offsets = torch.zeros(case.bs + 1, dtype=torch.int32, device=device)
    if case.bs > 0:
        offsets_host = torch.arange(0, case.bs + 1, dtype=torch.int32) * case.extend_len
        write_offsets.copy_(offsets_host.to(device))

    write_seed_slots = torch.empty(case.bs, dtype=torch.int32, device=device)
    if case.bs > 0:
        if case.prefix_len == 0:
            write_seed_slots.fill_(-1)
        else:
            per_req_stride = per_req_slots
            seeds = (
                torch.arange(case.bs, dtype=torch.int32, device=device) * per_req_stride
                + case.prefix_len
                - 1
            )
            write_seed_slots.copy_(seeds)

    write_num_valid_reqs = torch.tensor([case.bs], dtype=torch.int32, device=device)

    plan = WritePlan(
        write_offsets=write_offsets,
        write_seed_slot_indices=write_seed_slots,
        write_num_valid_reqs=write_num_valid_reqs,
    )

    fb_input_ids = torch.zeros(num_tokens_padded, dtype=torch.int32, device=device)
    fb_positions = torch.zeros(num_tokens_padded, dtype=torch.int32, device=device)
    fb_out_cache_loc = torch.zeros(num_tokens_padded, dtype=torch.int32, device=device)
    if total_entries > 0:
        flat_idx = torch.arange(total_entries, device=device, dtype=torch.int64)
        per_req_idx = flat_idx % max(case.extend_len, 1)
        req_idx = flat_idx // max(case.extend_len, 1)
        per_req_stride = per_req_slots
        slots = (req_idx * per_req_stride + case.prefix_len + per_req_idx) % max(
            num_slots, 1
        )
        fb_input_ids[:total_entries] = (flat_idx % 32768).to(torch.int32)
        fb_positions[:total_entries] = (case.prefix_len + per_req_idx).to(torch.int32)
        fb_out_cache_loc[:total_entries] = slots.to(torch.int32)

    # For SWA cases, pre-translate fb_out_cache_loc the same way an endpoint would (identity LUT here
    # so the cost is realistic but slot values are unchanged); FULL cases pass it through.
    if case.pool_kind == "swa_window_128":
        full_to_swa = torch.arange(num_slots + 1, dtype=torch.int32, device=device)
        full_to_swa[-1] = -1
        fb_out_cache_loc = full_to_swa[fb_out_cache_loc.to(torch.int64)].to(torch.int32)

    pseudo_expected_tokens = torch.zeros(
        num_tokens_padded, dtype=torch.int32, device=device
    )
    pseudo_expected_positions = torch.zeros(
        num_tokens_padded, dtype=torch.int32, device=device
    )

    violation_ring = torch.zeros(
        RING_CAPACITY, VIOLATION_FIELDS, dtype=torch.int64, device=device
    )
    violation_write_index = torch.zeros(1, dtype=torch.int32, device=device)
    slot_run_counter = torch.zeros(1, dtype=torch.int64, device=device)
    kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=device)

    return dict(
        canary_buf=canary_buf,
        plan=plan,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        violation_ring=violation_ring,
        violation_write_index=violation_write_index,
        slot_run_counter=slot_run_counter,
        kernel_run_counter=kernel_run_counter,
    )


def _run_one_case(case: BenchCase) -> dict:
    device = torch.device(DEFAULT_DEVICE)
    inputs = _build_write_inputs(case, device=device)

    def fn() -> None:
        canary_write_step(
            canary_buf=inputs["canary_buf"],
            plan=inputs["plan"],
            fb_input_ids=inputs["fb_input_ids"],
            fb_positions=inputs["fb_positions"],
            fb_out_cache_loc=inputs["fb_out_cache_loc"],
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            pseudo_mode=CanaryPseudoMode.OFF,
            pseudo_expected_tokens=inputs["pseudo_expected_tokens"],
            pseudo_expected_positions=inputs["pseudo_expected_positions"],
            violation_ring=inputs["violation_ring"],
            violation_write_index=inputs["violation_write_index"],
            slot_run_counter=inputs["slot_run_counter"],
            kernel_run_counter=inputs["kernel_run_counter"],
            real_kv_sources=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )

    fn()
    torch.cuda.synchronize()

    canary_us = do_bench(fn)
    baseline_us = baseline_us_slot_copy(total=_write_entry_count(case), device=device)
    total_entries = _write_entry_count(case)
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
        "\n[bench_kv_canary_write] name | us/call | per_slot_ns | ratio_vs_naive_write",
        flush=True,
    )
    yield


@pytest.mark.parametrize("case", select_matrix_cases())
def test_bench_write(case: BenchCase) -> None:
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

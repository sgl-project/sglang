"""Sweep-matrix benchmark for canary_write_step (triton.testing.perf_report style)."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.kv_canary.bench_helpers import (
    RING_CAPACITY,
    SWA_WINDOW,
    BenchCase,
    build_fast_matrix_cases,
    build_full_matrix_cases,
    cases_to_x_vals,
    naive_slot_copy_fn,
)
from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
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


_X_NAMES = ["bs", "prefix_len", "mode", "extend_len", "pool_kind"]
_X_VALS = cases_to_x_vals(
    get_benchmark_range(
        full_range=build_full_matrix_cases(),
        ci_range=build_fast_matrix_cases(),
    )
)


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

    # SWA endpoints would gather the LUT here; identity LUT keeps the bench self-consistent while still
    # exercising the host gather cost.
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_X_NAMES,
        x_vals=_X_VALS,
        line_arg="provider",
        line_vals=["canary", "naive"],
        line_names=["canary_write_step", "naive index_copy_"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="kv-canary-write-perf",
        args={},
    )
)
def benchmark(
    bs: int,
    prefix_len: int,
    mode: str,
    extend_len: int,
    pool_kind: str,
    provider: str,
) -> Tuple[float, float, float]:
    case = BenchCase(
        bs=bs,
        prefix_len=prefix_len,
        mode=mode,
        extend_len=extend_len,
        pool_kind=pool_kind,
    )
    device = torch.device(DEFAULT_DEVICE)

    if provider == "canary":
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

    else:
        fn = naive_slot_copy_fn(total=_write_entry_count(case), device=device)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)

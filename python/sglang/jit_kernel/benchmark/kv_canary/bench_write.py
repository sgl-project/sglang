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
    VerifyOrWriteContext,
)
from sglang.jit_kernel.kv_canary.write import WritePlan, launch_canary_write_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, suite="nightly-kernel-1-gpu", nightly=True)


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

_KERNEL_KIND_X_NAMES = ["kernel_kind_name", "enable_write_verify_inputs_name"]
_KERNEL_KIND_X_VALS = [
    (tag.name, str(enable)) for tag in CanaryLaunchTag for enable in (False, True)
]


def _write_entry_count(case: BenchCase) -> int:
    return case.bs * case.extend_len


def _write_num_slots(case: BenchCase) -> int:
    per_req_slots = max(
        SWA_WINDOW if case.pool_kind == "swa_window_128" else 1,
        case.prefix_len + case.extend_len,
    )
    return max(2, case.bs * per_req_slots + 1)


def _build_write_inputs(
    case: BenchCase, *, device: torch.device, mirror_expected_inputs: bool = False
) -> dict:
    total_entries = _write_entry_count(case)
    num_tokens_padded = max(1, total_entries)

    per_req_slots = max(
        SWA_WINDOW if case.pool_kind == "swa_window_128" else 1,
        case.prefix_len + case.extend_len,
    )
    num_slots = _write_num_slots(case)

    canary_buf = torch.zeros(
        num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
    )

    write_offsets = torch.zeros(case.bs + 1, dtype=torch.int64, device=device)
    if case.bs > 0:
        offsets_host = torch.arange(0, case.bs + 1, dtype=torch.int64) * case.extend_len
        write_offsets.copy_(offsets_host.to(device))

    write_seed_slots = torch.empty(case.bs, dtype=torch.int64, device=device)
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
            write_seed_slots.copy_(seeds.to(torch.int64))

    write_num_valid_reqs = torch.tensor([case.bs], dtype=torch.int32, device=device)

    plan = WritePlan(
        write_offsets=write_offsets,
        write_seed_slot_indices=write_seed_slots,
        write_num_valid_reqs=write_num_valid_reqs,
    )

    input_ids = torch.zeros(num_tokens_padded, dtype=torch.int64, device=device)
    positions = torch.zeros(num_tokens_padded, dtype=torch.int64, device=device)
    out_cache_loc = torch.zeros(num_tokens_padded, dtype=torch.int64, device=device)
    if total_entries > 0:
        flat_idx = torch.arange(total_entries, device=device, dtype=torch.int64)
        per_req_idx = flat_idx % max(case.extend_len, 1)
        req_idx = flat_idx // max(case.extend_len, 1)
        per_req_stride = per_req_slots
        slots = (req_idx * per_req_stride + case.prefix_len + per_req_idx) % max(
            num_slots, 1
        )
        input_ids[:total_entries] = (flat_idx % 32768).to(torch.int64)
        positions[:total_entries] = (case.prefix_len + per_req_idx).to(torch.int64)
        out_cache_loc[:total_entries] = slots.to(torch.int64)

    if case.pool_kind == "swa_window_128":
        full_to_swa = torch.arange(num_slots + 1, dtype=torch.int64, device=device)
        full_to_swa[-1] = -1
        out_cache_loc = full_to_swa[out_cache_loc]

    if mirror_expected_inputs:
        expected_input_tokens = input_ids.clone()
        expected_input_positions = positions.clone()
    else:
        expected_input_tokens = None
        expected_input_positions = None

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

    return dict(
        canary_buf=canary_buf,
        plan=plan,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
        violation_ring=violation_ring,
        violation_write_index=violation_write_index,
        slot_run_counter=slot_run_counter,
        kernel_run_counter=kernel_run_counter,
        enable_chain_position_assert=enable_chain_position_assert,
        real_kv_sources=real_kv_sources,
    )


def _build_context(
    *,
    inputs: dict,
    kernel_kind: CanaryLaunchTag,
    hash_mode: consts.RealKvHashMode,
) -> VerifyOrWriteContext:
    return VerifyOrWriteContext(
        canary_buf=inputs["canary_buf"],
        kernel_kind=kernel_kind,
        violation_ring=inputs["violation_ring"],
        violation_write_index=inputs["violation_write_index"],
        slot_run_counter=inputs["slot_run_counter"],
        kernel_run_counter=inputs["kernel_run_counter"],
        real_kv_sources=inputs["real_kv_sources"],
        real_kv_hash_mode=hash_mode,
        enable_chain_position_assert=inputs["enable_chain_position_assert"],
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
        inputs = _build_write_inputs(case, device=device)
        hash_mode_enum = consts.RealKvHashMode[case.hash_mode.upper()]
        context = _build_context(
            inputs=inputs,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            hash_mode=hash_mode_enum,
        )

        def fn() -> None:
            launch_canary_write_kernel(
                context=context,
                plan=inputs["plan"],
                input_ids=inputs["input_ids"],
                positions=inputs["positions"],
                out_cache_loc=inputs["out_cache_loc"],
                enable_write_input_assert=False,
                expected_input_tokens=inputs["expected_input_tokens"],
                expected_input_positions=inputs["expected_input_positions"],
            )

    else:
        fn = naive_slot_copy_fn(total=_write_entry_count(case), device=device)

    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_KERNEL_KIND_X_NAMES,
        x_vals=_KERNEL_KIND_X_VALS,
        line_arg="provider",
        line_vals=["canary"],
        line_names=["canary_write_step"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="kv-canary-write-kernel-kind-perf",
        args={},
    )
)
def benchmark_kernel_kind(
    kernel_kind_name: str,
    enable_write_verify_inputs_name: str,
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

    enable_write_verify_inputs = enable_write_verify_inputs_name == "True"
    inputs = _build_write_inputs(
        case, device=device, mirror_expected_inputs=enable_write_verify_inputs
    )
    kernel_kind = CanaryLaunchTag[kernel_kind_name]
    hash_mode_enum = consts.RealKvHashMode[case.hash_mode.upper()]
    context = _build_context(
        inputs=inputs,
        kernel_kind=kernel_kind,
        hash_mode=hash_mode_enum,
    )

    def fn() -> None:
        launch_canary_write_kernel(
            context=context,
            plan=inputs["plan"],
            input_ids=inputs["input_ids"],
            positions=inputs["positions"],
            out_cache_loc=inputs["out_cache_loc"],
            enable_write_input_assert=enable_write_verify_inputs,
            expected_input_tokens=inputs["expected_input_tokens"],
            expected_input_positions=inputs["expected_input_positions"],
        )

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
    benchmark_kernel_kind.run(print_data=True)

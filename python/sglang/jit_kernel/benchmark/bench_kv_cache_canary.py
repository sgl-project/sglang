import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.kv_cache_canary import (
    KERNEL_KIND_HEAD,
    VIOLATION_FIELDS,
    canary_step,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="base-b-kernel-benchmark-1-gpu-large")

NUM_SLOTS_LIST = get_benchmark_range(
    full_range=[2**n for n in range(4, 14)],
    ci_range=[64, 1024],
)

configs = list(itertools.product(NUM_SLOTS_LIST))

_SEED = 0xC0FFEE1234567890


def _build_state(num_slots: int, ring_capacity: int = 256) -> dict:
    return dict(
        violation_ring=torch.zeros(
            ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=DEFAULT_DEVICE
        ),
        violation_ring_valid=torch.zeros(
            ring_capacity, dtype=torch.int32, device=DEFAULT_DEVICE
        ),
        violation_write_index=torch.zeros(1, dtype=torch.int32, device=DEFAULT_DEVICE),
        first_violation=torch.zeros(
            VIOLATION_FIELDS, dtype=torch.int64, device=DEFAULT_DEVICE
        ),
        first_violation_set=torch.zeros(1, dtype=torch.int32, device=DEFAULT_DEVICE),
        is_errored=torch.zeros(1, dtype=torch.int32, device=DEFAULT_DEVICE),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device=DEFAULT_DEVICE),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=DEFAULT_DEVICE),
    )


def _canary_step(num_slots: int, slot_stride_bytes: int, mode: str) -> None:
    """One canary step launch under the given mode (all-write / all-verify / mixed)."""
    src = torch.zeros(
        num_slots * 2, slot_stride_bytes, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    dst = torch.zeros(
        num_slots * 2, slot_stride_bytes, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    state = _build_state(num_slots)

    if mode == "write":
        num_verify = 0
        num_write = num_slots
        num_write_reqs = 1
    elif mode == "verify":
        num_verify = num_slots
        num_write = 0
        num_write_reqs = 0
    else:  # mixed: half verify, half write
        num_verify = num_slots // 2
        num_write = num_slots - num_verify
        num_write_reqs = 1

    verify_slot_indices = torch.arange(
        max(num_verify, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    verify_positions = torch.arange(
        max(num_verify, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    verify_req_ids = torch.zeros(
        max(num_verify, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    verify_prev_slot_indices = torch.full(
        (max(num_verify, 1),), -1, dtype=torch.int64, device=DEFAULT_DEVICE
    )
    verify_active_mask = torch.zeros(
        max(num_verify, 1), dtype=torch.int32, device=DEFAULT_DEVICE
    )
    verify_active_mask[:num_verify] = 1

    write_slot_indices = torch.arange(
        max(num_write, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    write_token_ids = torch.arange(
        max(num_write, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    write_positions = torch.arange(
        max(num_write, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    write_req_ids = torch.zeros(
        max(num_write, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )

    write_req_seed_slot_indices = torch.full(
        (max(num_write_reqs, 1),), -1, dtype=torch.int64, device=DEFAULT_DEVICE
    )
    write_req_entry_starts = torch.zeros(
        max(num_write_reqs, 1), dtype=torch.int64, device=DEFAULT_DEVICE
    )
    write_req_entry_counts = torch.full(
        (max(num_write_reqs, 1),), num_write, dtype=torch.int64, device=DEFAULT_DEVICE
    )
    write_req_active_mask = torch.zeros(
        max(num_write_reqs, 1), dtype=torch.int32, device=DEFAULT_DEVICE
    )
    write_req_active_mask[:num_write_reqs] = 1

    canary_step(
        src_buf=src.flatten(),
        dst_buf=dst.flatten(),
        slot_stride_bytes=slot_stride_bytes,
        verify_slot_indices=verify_slot_indices,
        verify_positions=verify_positions,
        verify_req_ids=verify_req_ids,
        verify_prev_slot_indices=verify_prev_slot_indices,
        verify_active_mask=verify_active_mask,
        write_slot_indices=write_slot_indices,
        write_token_ids=write_token_ids,
        write_positions=write_positions,
        write_req_ids=write_req_ids,
        write_req_seed_slot_indices=write_req_seed_slot_indices,
        write_req_entry_starts=write_req_entry_starts,
        write_req_entry_counts=write_req_entry_counts,
        write_req_active_mask=write_req_active_mask,
        seed=_SEED,
        violation_ring=state["violation_ring"],
        violation_ring_valid=state["violation_ring_valid"],
        violation_write_index=state["violation_write_index"],
        first_violation=state["first_violation"],
        first_violation_set=state["first_violation_set"],
        is_errored=state["is_errored"],
        slot_run_counter=state["slot_run_counter"],
        kernel_run_counter=state["kernel_run_counter"],
        kernel_kind=KERNEL_KIND_HEAD,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_slots"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["write_only", "verify_only", "mixed"],
        line_names=["Write only", "Verify only", "Mixed 50/50"],
        styles=[("blue", "-"), ("green", "-"), ("red", "--")],
        ylabel="us",
        plot_name="kv-cache-canary-performance",
        args={},
    )
)
def benchmark(num_slots: int, provider: str):
    slot_stride_bytes = 256

    if provider == "write_only":
        fn = lambda: _canary_step(num_slots, slot_stride_bytes, "write")
    elif provider == "verify_only":
        fn = lambda: _canary_step(num_slots, slot_stride_bytes, "verify")
    else:
        fn = lambda: _canary_step(num_slots, slot_stride_bytes, "mixed")

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)

"""KV cache canary kernel benchmarks.

Three axes are covered:

1. ``context_len`` — single-req decode that verifies the FULL prefix
   every forward (a 10k-token prefix verifies all 10k positions);
2. ``extend_chunk`` — single-req extend writes a chunk of new tokens
   per forward;
3. ``decode_bs`` — many concurrent decode reqs, each contributing a
   short history + 1 new write per forward (bs up to 1024).

All scenarios use ``triton.testing.do_bench_cudagraph`` so we measure
the same code path that ships in production (kernel inside a captured
CUDA graph).
"""

from __future__ import annotations

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
    CANARY_EXPECTED_SKIP_SENTINEL,
    CANARY_SLOT_BYTES,
    KERNEL_KIND_HEAD,
    VIOLATION_FIELDS,
    canary_step,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref_legacy import BatchPlanGpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-benchmark-1-gpu-large")

CONTEXT_LEN_LIST = get_benchmark_range(
    full_range=[128, 512, 1024, 4096, 10240],
    ci_range=[128, 1024],
)
EXTEND_CHUNK_LIST = get_benchmark_range(
    full_range=[256, 1024, 2048, 4096],
    ci_range=[256, 1024],
)
DECODE_BS_LIST = get_benchmark_range(
    full_range=[16, 64, 256, 1024],
    ci_range=[16, 64],
)

# Per-req decode history depth in the decode_bs sweep. Short enough to
# keep the GPU-resident verify entry count manageable (bs * history)
# while still exercising the verify path.
_DECODE_HISTORY = 64


def _build_state(*, num_slots: int, ring_capacity: int = 256) -> dict:
    return dict(
        violation_ring=torch.zeros(
            ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=DEFAULT_DEVICE
        ),
        violation_write_index=torch.zeros(1, dtype=torch.int32, device=DEFAULT_DEVICE),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device=DEFAULT_DEVICE),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=DEFAULT_DEVICE),
    )


def _i64(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=DEFAULT_DEVICE)


def _i32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=DEFAULT_DEVICE)


def _launch(
    *,
    buf: torch.Tensor,
    verify_slot_indices: list[int],
    verify_positions: list[int],
    verify_prev_slot_indices: list[int],
    write_slot_indices: list[int],
    write_token_ids: list[int],
    write_positions: list[int],
    write_req_seed_slot_indices: list[int],
    write_req_entry_starts: list[int],
    write_req_entry_counts: list[int],
    state: dict,
) -> None:
    n_verify = len(verify_slot_indices)
    n_write = len(write_slot_indices)
    n_write_reqs = len(write_req_seed_slot_indices)
    plan = BatchPlanGpu(
        verify_slot_indices=_i64(verify_slot_indices or [0]),
        verify_positions=_i64(verify_positions or [0]),
        verify_prev_slot_indices=_i64(verify_prev_slot_indices or [-1]),
        verify_num_valid=_i32([n_verify]),
        write_slot_indices=_i64(write_slot_indices or [0]),
        write_token_ids=_i64(write_token_ids or [0]),
        write_positions=_i64(write_positions or [0]),
        write_req_seed_slot_indices=_i64(write_req_seed_slot_indices or [-1]),
        write_req_entry_starts=_i64(write_req_entry_starts or [0]),
        write_req_entry_counts=_i64(write_req_entry_counts or [0]),
        write_req_num_valid=_i32([n_write_reqs]),
        expected_write_token_ids=_i64(
            [CANARY_EXPECTED_SKIP_SENTINEL] * max(n_write, 1)
        ),
        expected_write_positions=_i64(
            [CANARY_EXPECTED_SKIP_SENTINEL] * max(n_write, 1)
        ),
    )
    canary_step(
        buf=buf,
        plan=plan,
        violation_ring=state["violation_ring"],
        violation_write_index=state["violation_write_index"],
        slot_run_counter=state["slot_run_counter"],
        kernel_run_counter=state["kernel_run_counter"],
        kernel_kind=KERNEL_KIND_HEAD,
        real_kv_buf=torch.zeros(1, 1, dtype=torch.uint8, device=DEFAULT_DEVICE),
        real_kv_read_bytes=0,
        real_kv_hash_mode=0,
    )


def _context_len_step(context_len: int) -> None:
    """Single decode step on a req with ``context_len`` already written.

    The verify path covers ALL prefix positions in the same forward,
    plus the canary writes the new decode token at the tail.
    """
    num_slots = context_len + 2
    buf = torch.zeros(
        num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    state = _build_state(num_slots=num_slots)

    verify_slot_indices = list(range(context_len))
    verify_positions = list(range(context_len))
    verify_prev_slot_indices = [-1] + list(range(context_len - 1))
    _launch(
        buf=buf,
        verify_slot_indices=verify_slot_indices,
        verify_positions=verify_positions,
        verify_prev_slot_indices=verify_prev_slot_indices,
        write_slot_indices=[context_len],
        write_token_ids=[1234],
        write_positions=[context_len],
        write_req_seed_slot_indices=[context_len - 1] if context_len > 0 else [-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        state=state,
    )


def _extend_chunk_step(chunk_size: int) -> None:
    """Single req writes ``chunk_size`` new tokens (chunked prefill)."""
    num_slots = chunk_size + 2
    buf = torch.zeros(
        num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    state = _build_state(num_slots=num_slots)

    _launch(
        buf=buf,
        verify_slot_indices=[],
        verify_positions=[],
        verify_prev_slot_indices=[],
        write_slot_indices=list(range(chunk_size)),
        write_token_ids=[(i * 17) & 0xFFFF for i in range(chunk_size)],
        write_positions=list(range(chunk_size)),
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[chunk_size],
        state=state,
    )


def _decode_bs_step(batch_size: int) -> None:
    """``batch_size`` concurrent decode reqs, each verifies a short history + writes 1 token."""
    history = _DECODE_HISTORY
    total_slots_per_req = history + 1
    num_slots = batch_size * total_slots_per_req
    buf = torch.zeros(
        num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=DEFAULT_DEVICE
    )
    state = _build_state(num_slots=num_slots)

    verify_slot_indices: list[int] = []
    verify_positions: list[int] = []
    verify_prev_slot_indices: list[int] = []
    write_slot_indices: list[int] = []
    write_token_ids: list[int] = []
    write_positions: list[int] = []
    write_req_seed_slot_indices: list[int] = []
    write_req_entry_starts: list[int] = []
    write_req_entry_counts: list[int] = []
    for r in range(batch_size):
        base = r * total_slots_per_req
        for j in range(history):
            slot = base + j
            verify_slot_indices.append(slot)
            verify_positions.append(j)
            verify_prev_slot_indices.append(-1 if j == 0 else base + j - 1)
        write_req_seed_slot_indices.append(base + history - 1)
        write_req_entry_starts.append(len(write_slot_indices))
        write_req_entry_counts.append(1)
        write_slot_indices.append(base + history)
        write_token_ids.append(((r + 1) * 5) & 0xFFFF)
        write_positions.append(history)

    _launch(
        buf=buf,
        verify_slot_indices=verify_slot_indices,
        verify_positions=verify_positions,
        verify_prev_slot_indices=verify_prev_slot_indices,
        write_slot_indices=write_slot_indices,
        write_token_ids=write_token_ids,
        write_positions=write_positions,
        write_req_seed_slot_indices=write_req_seed_slot_indices,
        write_req_entry_starts=write_req_entry_starts,
        write_req_entry_counts=write_req_entry_counts,
        state=state,
    )


_context_len_configs = list(itertools.product(CONTEXT_LEN_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["context_len"],
        x_vals=_context_len_configs,
        line_arg="provider",
        line_vals=["full_prefix_verify"],
        line_names=["Full-prefix verify"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="kv-cache-canary-context-len",
        args={},
    )
)
def benchmark_context_len(context_len: int, provider: str):
    return run_benchmark(lambda: _context_len_step(context_len))


_extend_chunk_configs = list(itertools.product(EXTEND_CHUNK_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["chunk_size"],
        x_vals=_extend_chunk_configs,
        line_arg="provider",
        line_vals=["chunk_write"],
        line_names=["Chunk write"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="kv-cache-canary-extend-chunk",
        args={},
    )
)
def benchmark_extend_chunk(chunk_size: int, provider: str):
    return run_benchmark(lambda: _extend_chunk_step(chunk_size))


_decode_bs_configs = list(itertools.product(DECODE_BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=_decode_bs_configs,
        line_arg="provider",
        line_vals=["decode_step"],
        line_names=["Decode step"],
        styles=[("red", "-")],
        ylabel="us",
        plot_name="kv-cache-canary-decode-bs",
        args={},
    )
)
def benchmark_decode_bs(batch_size: int, provider: str):
    return run_benchmark(lambda: _decode_bs_step(batch_size))


if __name__ == "__main__":
    benchmark_context_len.run(print_data=True)
    benchmark_extend_chunk.run(print_data=True)
    benchmark_decode_bs.run(print_data=True)

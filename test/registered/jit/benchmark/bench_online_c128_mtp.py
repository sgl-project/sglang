"""Benchmark online c128 speculative write-prefix kernel."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
    run_benchmark_no_cudagraph,
)
from sglang.jit_kernel.dsv4.online_c128_mtp import _jit_online_c128_mtp_module
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=10, stage="jit-kernel-benchmark", runner_config="amd")

HEAD_DIM = 512
STATE_DIM = HEAD_DIM * 3
SWA_PAGE_SIZE = 128

BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    ci_range=[8, 64],
)
NUM_VERIFY_TOKENS_RANGE = get_benchmark_range(
    full_range=[1, 4, 8],
    ci_range=[8],
)
LAYOUT_RANGE = ["uniform", "dspark-ragged"]
BENCHMARK_CONFIGS = list(
    itertools.product(BATCH_SIZE_RANGE, NUM_VERIFY_TOKENS_RANGE, LAYOUT_RANGE)
)


@dataclass
class BenchmarkCase:
    kv_score_input: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    verify_lens: torch.Tensor
    extend_start_loc: torch.Tensor
    req_to_token: torch.Tensor
    ape: torch.Tensor
    state: torch.Tensor
    layer_bs: int
    num_verify_tokens: int
    state_slot_stride: int


def round_up_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def make_seq_lens(batch_size: int, num_verify_tokens: int) -> torch.Tensor:
    # Cover chunk positions around the interesting boundaries. This exercises
    # both the has-partial path and the final_seq % 128 == 0 skip-write path.
    offsets = torch.tensor([0, 1, 2, 63, 120, 126, 127], dtype=torch.int64)
    seq_offsets = offsets[torch.arange(batch_size, dtype=torch.int64) % offsets.numel()]
    base = 8 * SWA_PAGE_SIZE
    seq_lens = base + seq_offsets
    assert int(seq_lens.max()) + num_verify_tokens < base + 2 * SWA_PAGE_SIZE
    return seq_lens.to(device=DEFAULT_DEVICE)


def make_req_to_token(
    batch_size: int, max_seq_len: int, num_chunks: int
) -> torch.Tensor:
    chunk_ids = torch.arange(max_seq_len, dtype=torch.int32) // SWA_PAGE_SIZE
    req_offsets = torch.arange(batch_size, dtype=torch.int32).unsqueeze(1) * num_chunks
    req_to_token = req_offsets + chunk_ids.unsqueeze(0)
    return req_to_token.contiguous().to(device=DEFAULT_DEVICE)


def make_case(batch_size: int, num_verify_tokens: int, layout: str) -> BenchmarkCase:
    seq_lens = make_seq_lens(batch_size, num_verify_tokens)
    req_pool_indices = torch.arange(
        batch_size, dtype=torch.int64, device=DEFAULT_DEVICE
    )

    max_seq_len = int(seq_lens.max().item()) + num_verify_tokens + SWA_PAGE_SIZE
    num_chunks = round_up_div(max_seq_len, SWA_PAGE_SIZE)
    req_to_token = make_req_to_token(batch_size, max_seq_len, num_chunks)

    num_full_locs = batch_size * num_chunks

    state_slot_stride = num_full_locs
    state = torch.empty(
        (state_slot_stride * (1 + num_verify_tokens), STATE_DIM),
        dtype=torch.float32,
        device=DEFAULT_DEVICE,
    )
    state.normal_(mean=0.0, std=0.01)

    if layout == "dspark-ragged":
        verify_lens = (
            torch.arange(batch_size, dtype=torch.int32, device=DEFAULT_DEVICE)
            % num_verify_tokens
            + 1
        )
        extend_start_loc = torch.nn.functional.pad(
            torch.cumsum(verify_lens, dim=0)[:-1], (1, 0)
        )
        num_input_tokens = int(verify_lens.sum().item())
    else:
        assert layout == "uniform"
        verify_lens = torch.empty((0,), dtype=torch.int32, device=DEFAULT_DEVICE)
        extend_start_loc = torch.empty((0,), dtype=torch.int32, device=DEFAULT_DEVICE)
        num_input_tokens = batch_size * num_verify_tokens

    kv_score_input = torch.randn(
        num_input_tokens,
        HEAD_DIM * 2,
        dtype=torch.float32,
        device=DEFAULT_DEVICE,
    )
    ape = torch.randn(128, HEAD_DIM, dtype=torch.float32, device=DEFAULT_DEVICE)

    return BenchmarkCase(
        kv_score_input=kv_score_input,
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        verify_lens=verify_lens,
        extend_start_loc=extend_start_loc,
        req_to_token=req_to_token,
        ape=ape,
        state=state,
        layer_bs=batch_size,
        num_verify_tokens=num_verify_tokens,
        state_slot_stride=state_slot_stride,
    )


def call_write_prefix(module, case: BenchmarkCase) -> None:
    module.write_prefix_states(
        case.kv_score_input,
        case.seq_lens,
        case.req_pool_indices,
        case.verify_lens,
        case.extend_start_loc,
        case.req_to_token,
        case.ape,
        case.state,
        case.layer_bs,
        case.num_verify_tokens,
        case.state_slot_stride,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "num_verify_tokens", "layout"],
        x_vals=BENCHMARK_CONFIGS,
        line_arg="launch_mode",
        line_vals=["cuda_graph", "eager"],
        line_names=["CUDA graph", "Eager launch"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="online-c128-spec-write-prefix-performance",
        args={},
    )
)
def benchmark(
    batch_size: int, num_verify_tokens: int, layout: str, launch_mode: str
) -> tuple[float, float, float]:
    case = make_case(batch_size, num_verify_tokens, layout)
    module = _jit_online_c128_mtp_module(
        HEAD_DIM, case.seq_lens.dtype, case.req_pool_indices.dtype, case.state.dtype
    )

    def fn():
        call_write_prefix(module, case)

    if launch_mode == "cuda_graph":
        return run_benchmark(fn)
    if launch_mode == "eager":
        return run_benchmark_no_cudagraph(fn)
    raise ValueError(f"Unknown launch_mode: {launch_mode}")


if __name__ == "__main__":
    benchmark.run(print_data=True)

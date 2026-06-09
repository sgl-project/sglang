"""Benchmark online c128 MTP write-prefix kernel.

This microbenchmark targets the two latency-sensitive changes in
``online_c128_mtp.cuh``:

1. one thread handles one head-dim element for head_dim=512;
2. per-step memory loads are issued before the online-softmax loop.

Run the same command on the before/after commits and compare the reported
latency.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List

import torch

from sglang.jit_kernel.dsv4.online_c128_mtp import _jit_online_c128_mtp_module


HEAD_DIM = 512
STATE_DIM = HEAD_DIM * 3
SWA_PAGE_SIZE = 128


@dataclass
class BenchmarkCase:
    kv_score_input: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    req_to_token: torch.Tensor
    full_to_swa: torch.Tensor
    ape: torch.Tensor
    state: torch.Tensor
    layer_bs: int
    num_verify_tokens: int
    state_slot_stride: int


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def round_up_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def make_seq_lens(batch_size: int, num_verify_tokens: int, device: str) -> torch.Tensor:
    # Cover chunk positions around the interesting boundaries. This exercises
    # both the has-partial path and the final_seq % 128 == 0 skip-write path.
    offsets = torch.tensor([0, 1, 2, 63, 120, 126, 127], dtype=torch.int64)
    seq_offsets = offsets[torch.arange(batch_size, dtype=torch.int64) % offsets.numel()]
    base = 8 * SWA_PAGE_SIZE
    seq_lens = base + seq_offsets
    assert int(seq_lens.max()) + num_verify_tokens < base + 2 * SWA_PAGE_SIZE
    return seq_lens.to(device=device)


def make_req_to_token(batch_size: int, max_seq_len: int, num_chunks: int, device: str) -> torch.Tensor:
    chunk_ids = torch.arange(max_seq_len, dtype=torch.int32) // SWA_PAGE_SIZE
    req_offsets = torch.arange(batch_size, dtype=torch.int32).unsqueeze(1) * num_chunks
    req_to_token = req_offsets + chunk_ids.unsqueeze(0)
    return req_to_token.contiguous().to(device=device)


def make_case(
    batch_size: int,
    num_verify_tokens: int,
    device: str,
    seed: int,
) -> BenchmarkCase:
    torch.manual_seed(seed + batch_size * 17 + num_verify_tokens)

    seq_lens = make_seq_lens(batch_size, num_verify_tokens, device)
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)

    max_seq_len = int(seq_lens.max().item()) + num_verify_tokens + SWA_PAGE_SIZE
    num_chunks = round_up_div(max_seq_len, SWA_PAGE_SIZE)
    req_to_token = make_req_to_token(batch_size, max_seq_len, num_chunks, device)

    num_full_locs = batch_size * num_chunks
    full_to_swa = (
        torch.arange(num_full_locs, dtype=torch.int64, device=device) * SWA_PAGE_SIZE
    )

    state_slot_stride = num_full_locs
    state = torch.empty(
        (state_slot_stride * (1 + num_verify_tokens), STATE_DIM),
        dtype=torch.float32,
        device=device,
    )
    state.normal_(mean=0.0, std=0.01)

    kv_score_input = torch.randn(
        batch_size * num_verify_tokens,
        HEAD_DIM * 2,
        dtype=torch.float32,
        device=device,
    )
    ape = torch.randn(128, HEAD_DIM, dtype=torch.float32, device=device)

    return BenchmarkCase(
        kv_score_input=kv_score_input,
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        full_to_swa=full_to_swa,
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
        case.req_to_token,
        case.full_to_swa,
        case.ape,
        case.state,
        case.layer_bs,
        SWA_PAGE_SIZE,
        case.num_verify_tokens,
        case.state_slot_stride,
    )


def measure_with_events(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def measure_with_cuda_graph(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run_benchmark(
    batch_sizes: Iterable[int],
    num_verify_tokens: int,
    warmup: int,
    iters: int,
    device: str,
    seed: int,
    cuda_graph: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if num_verify_tokens <= 0 or num_verify_tokens > 8:
        raise ValueError("num_verify_tokens must be in [1, 8]")

    torch_device = torch.device(device)
    if torch_device.index is not None:
        torch.cuda.set_device(torch_device)
    module = _jit_online_c128_mtp_module(HEAD_DIM)

    print(
        "kernel=online_c128_mtp.write_prefix_states "
        f"head_dim={HEAD_DIM} num_verify_tokens={num_verify_tokens} "
        f"warmup={warmup} iters={iters} cuda_graph={cuda_graph}"
    )
    print("batch_size,latency_us,state_mib")

    for batch_size in batch_sizes:
        case = make_case(batch_size, num_verify_tokens, device, seed)
        fn = lambda: call_write_prefix(module, case)

        if cuda_graph:
            latency_us = measure_with_cuda_graph(fn, warmup, iters)
        else:
            latency_us = measure_with_events(fn, warmup, iters)

        state_mib = case.state.numel() * case.state.element_size() / math.pow(2, 20)
        print(f"{batch_size},{latency_us:.3f},{state_mib:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--num-verify-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Use CUDA graph replay to reduce Python launch overhead noise.",
    )
    args = parser.parse_args()

    run_benchmark(
        batch_sizes=parse_int_list(args.batch_sizes),
        num_verify_tokens=args.num_verify_tokens,
        warmup=args.warmup,
        iters=args.iters,
        device=args.device,
        seed=args.seed,
        cuda_graph=args.cuda_graph,
    )


if __name__ == "__main__":
    main()

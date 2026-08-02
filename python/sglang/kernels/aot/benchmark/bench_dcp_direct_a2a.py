"""Benchmark direct DCP A2A against the existing A2A + Triton path.

Run on a single node with at least four peer-accessible NVIDIA GPUs::

    python bench_dcp_direct_a2a.py
    python bench_dcp_direct_a2a.py --num-gpu 2,4
"""

from __future__ import annotations

import atexit
import functools
import os

import torch
import torch.distributed as dist
import triton.testing

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.srt.layers.dcp import DirectSymmA2AWorkspace, dcp_a2a_lse_reduce
from sglang.utils import is_in_ci


IS_CI = is_in_ci()
PROVIDERS = ("direct-eager", "direct-graph", "a2a-eager", "a2a-graph")
CONFIGS = (
    [(1, 2, 64, "bf16", False), (16, 2, 128, "bf16", False)]
    if IS_CI
    else [
        (tokens, heads_per_rank, head_dim, dtype, base_e)
        for tokens in (1, 8, 32, 128)
        for heads_per_rank in (2, 8)
        for head_dim in (64, 128)
        for dtype in ("fp16", "bf16")
        for base_e in (True, False)
    ]
)
WORLD_SIZES = (2, 4)


@functools.lru_cache(maxsize=1)
def _init_group():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    cp_group = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    ps._WORLD = cp_group
    atexit.register(ps.destroy_distributed_environment)
    return cp_group


def _capture(fn, cp_group):
    torch.cuda.synchronize()
    dist.barrier(group=cp_group.cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = fn()
    torch.cuda.synchronize()
    dist.barrier(group=cp_group.cpu_group)
    return graph, captured_result


def _make_a2a_buffers(
    *,
    world_size: int,
    tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    # FP32 LSE occupies two FP16/BF16 columns when transported as bytes.
    lse_pack_dim = (
        torch.empty((), dtype=torch.float32).element_size()
        // torch.empty((), dtype=dtype).element_size()
    )
    return {
        "send_combined": torch.empty(
            world_size,
            tokens,
            heads_per_rank,
            head_dim + lse_pack_dim,
            dtype=dtype,
            device=device,
        ),
        "recv_combined": torch.empty(
            world_size,
            tokens,
            heads_per_rank,
            head_dim + lse_pack_dim,
            dtype=dtype,
            device=device,
        ),
        "send_lse": torch.empty(
            world_size,
            tokens,
            heads_per_rank,
            dtype=torch.float32,
            device=device,
        ),
        "recv_lse": torch.empty(
            world_size,
            tokens,
            heads_per_rank,
            dtype=torch.float32,
            device=device,
        ),
    }


def _time_fixed_repetitions(fn, cp_group, *, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=cp_group.cpu_group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0 / repetitions

    # Report the slowest rank, which determines collective latency.
    latency = torch.tensor([elapsed_us], dtype=torch.float64, device="cpu")
    dist.all_reduce(latency, op=dist.ReduceOp.MAX, group=cp_group.cpu_group)
    dist.barrier(group=cp_group.cpu_group)
    return latency.item()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["tokens", "heads_per_rank", "head_dim", "dtype_name", "base_e"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=list(PROVIDERS),
        line_names=[
            "Direct eager",
            "Direct CUDA graph",
            "A2A + Triton eager",
            "A2A + Triton CUDA graph",
        ],
        styles=[
            ("green", "-"),
            ("blue", "-"),
            ("red", "--"),
            ("orange", "--"),
        ],
        ylabel="Latency (us, slowest rank)",
        plot_name="dcp-direct-a2a",
        args={},
    )
)
def benchmark(tokens, heads_per_rank, head_dim, dtype_name, base_e, provider):
    cp_group = _init_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    total_heads = cp_group.world_size * heads_per_rank
    output = torch.randn(tokens, total_heads, head_dim, dtype=dtype, device=device)
    lse = torch.randn(tokens, total_heads, dtype=torch.float32, device=device)

    workspace = None
    graph = None
    captured_result = None
    if provider.startswith("direct"):
        workspace = DirectSymmA2AWorkspace(
            cp_group=cp_group,
            device=device,
            max_num_tokens=tokens,
            heads_per_rank=heads_per_rank,
            head_dim=head_dim,
            dtype=dtype,
            num_ubatches=1,
        )
        combined = torch.empty(
            tokens, heads_per_rank, head_dim, dtype=dtype, device=device
        )

        def eager_call(workspace=workspace):
            return workspace.lse_reduce(
                output,
                lse,
                is_lse_base_on_e=base_e,
                output=combined,
            )

        if provider.endswith("-graph"):
            graph, captured_result = _capture(eager_call, cp_group)
            fn = graph.replay
        else:
            fn = eager_call
    else:
        cuda_graph_buffers = _make_a2a_buffers(
            world_size=cp_group.world_size,
            tokens=tokens,
            heads_per_rank=heads_per_rank,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        def baseline_call():
            return dcp_a2a_lse_reduce(
                output,
                lse,
                cp_group,
                is_lse_base_on_e=base_e,
                cuda_graph_buffers=cuda_graph_buffers,
                comm_backend="a2a",
            )

        if provider.endswith("-graph"):
            graph, captured_result = _capture(baseline_call, cp_group)
            fn = graph.replay
        else:
            fn = baseline_call

    try:
        latency = _time_fixed_repetitions(
            fn,
            cp_group,
            warmup=2 if IS_CI else 10,
            repetitions=10 if IS_CI else 100,
        )
    finally:
        # Keep captured tensors/workspace alive through the last synchronized
        # iteration, then release them in the same order on every rank.
        del fn
        del captured_result
        del graph
        del workspace
        torch.cuda.synchronize()
        dist.barrier(group=cp_group.cpu_group)
    return latency


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=WORLD_SIZES,
        main_fn=lambda: benchmark.run(print_data=True),
        timeout=1200,
    )

#!/usr/bin/env python3
"""Token sweep: fused AR+RMSNorm vs split AR then RMSNorm, on the K3 front.

Launch: torchrun --nproc_per_node=8 \\
  test/manual/kernels/bench_fused_ar_rmsnorm.py

The combined [latent | shared] buffer is 3 rows of NORM_DIM per token; only
the first num_tokens rows are normalized. Prints the crossover for
SGLANG_ROCM_K3_FUSED_AR_RMSNORM_MAX_TOKENS.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

NORM_DIM = 3584
ROWS_PER_TOKEN = 3
EPS = 1e-6
WARMUP = 20
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "1"))
TOKENS = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096]
if os.environ.get("BENCH_TOKENS"):
    TOKENS = [int(t) for t in os.environ["BENCH_TOKENS"].split(",")]


def _init():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, local_rank


def _time_graph(graph: torch.cuda.CUDAGraph) -> float:
    stream = torch.cuda.current_stream()
    for _ in range(WARMUP):
        graph.replay()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    stream.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS


def _capture(fn):
    from sglang.srt.distributed.parallel_state import graph_capture

    with graph_capture() as gc:
        s = gc.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            fn()
    return graph


def main() -> int:
    rank, local_rank = _init()
    device = torch.device(f"cuda:{local_rank}")

    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_reduce,
    )
    from sglang.srt.layers.communication.k3_fused_ar_rmsnorm import try_fused_ar_rmsnorm
    from sglang.srt.layers.layernorm import RMSNorm

    weight = torch.ones(NORM_DIM, dtype=torch.bfloat16, device=device)
    # Same kernel the model runs: routed_expert_norm is an RMSNorm module.
    norm = RMSNorm(NORM_DIM, eps=EPS).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    rows = []

    for tokens in TOKENS:
        n_rows = tokens * ROWS_PER_TOKEN
        g = torch.Generator(device=device)
        g.manual_seed(rank * 1000 + tokens)
        base = torch.randn(
            n_rows, NORM_DIM, dtype=torch.bfloat16, device=device, generator=g
        )

        probe = try_fused_ar_rmsnorm(base.clone(), weight, EPS, num_norm_rows=tokens)
        if probe is None:
            if rank == 0:
                rows.append((tokens, None, None))
                print(f"tokens={tokens}: fused path declined (1-stage regime)")
            continue

        fused_buf = base.clone()

        def _fused(buf=fused_buf, n=tokens):
            try_fused_ar_rmsnorm(buf, weight, EPS, num_norm_rows=n)

        split_buf = base.clone()

        def _split(buf=split_buf, n=tokens):
            reduced = tensor_model_parallel_all_reduce(buf)
            norm(reduced[:n])

        g_fused = _capture(_fused)
        g_split = _capture(_split)
        # Median of interleaved repeats; deltas here are noise-dominated.
        fused_runs = []
        split_runs = []
        for _ in range(REPEATS):
            fused_runs.append(_time_graph(g_fused))
            split_runs.append(_time_graph(g_split))
        fused_runs.sort()
        split_runs.sort()
        us_fused = fused_runs[len(fused_runs) // 2]
        us_split = split_runs[len(split_runs) // 2]

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (us_fused, us_split))
        if rank == 0:
            mean_fused = sum(x[0] for x in gathered) / len(gathered)
            mean_split = sum(x[1] for x in gathered) / len(gathered)
            rows.append((tokens, mean_fused, mean_split))
            delta = mean_split - mean_fused
            verdict = "fused" if delta > 0 else "split"
            print(
                f"tokens={tokens:<4} rows={n_rows:<4} "
                f"fused={mean_fused:8.2f} us  split={mean_split:8.2f} us  "
                f"delta={delta:+7.2f} us ({100 * delta / mean_split:+5.1f}%)  "
                f"-> {verdict}"
            )

    if rank == 0:
        measured = [r for r in rows if r[1] is not None]
        wins = [t for t, f, s in measured if s > f]
        losses = [t for t, f, s in measured if s <= f]
        print("\nfused wins at tokens:", wins or "none")
        print("fused loses at tokens:", losses or "none")
        if wins and losses and min(losses) > min(wins):
            print(f"suggested SGLANG_ROCM_K3_FUSED_AR_RMSNORM_MAX_TOKENS={max(wins)}")
        elif not losses:
            print("no upper crossover in range; cap 0 (off) is correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())

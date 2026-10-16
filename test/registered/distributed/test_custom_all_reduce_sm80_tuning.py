"""Standalone SM80 TP8 diagnostic; NOT a model/runtime dispatch change.

Run in the existing CUDA environment (no model load required)::

    python -m torch.distributed.run --standalone --nproc_per_node=8 \
      test/registered/distributed/test_custom_all_reduce_sm80_tuning.py \
      --blocks 4,8,16,32,108 --algos push,pull --include-v1

Defaults: BF16 10240/40960 bytes, 83 AR calls per captured graph, 10 timing
samples of 256 graph replays. Use --replays 32 --repeats 3 for a first pass.
--workloads pure,interleaved additionally measures rank-dependent GEMM work.
The interleaved time includes GEMMs; it is NOT isolated communication latency.
Every rank must receive identical arguments. Keep the model API idle while
measuring; concurrent activity invalidates comparative latency conclusions.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import statistics
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", default="4,8,16,32,108")
    parser.add_argument("--algos", default="push,pull")
    parser.add_argument("--sizes", default="10240,40960", help="BF16 message BYTES")
    parser.add_argument("--loops", type=int, default=83)
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--workloads", default="pure", help="pure,interleaved")
    parser.add_argument("--include-v1", action="store_true")
    parser.add_argument("--matmul-dim", type=int, default=512)
    args = parser.parse_args()
    args.blocks = [int(x) for x in args.blocks.split(",")]
    args.algos = args.algos.split(",")
    args.sizes = [int(x) for x in args.sizes.split(",")]
    args.workloads = args.workloads.split(",")
    if not all(x > 0 for x in args.blocks + args.sizes):
        parser.error("blocks and byte sizes must be positive")
    if not all(x % 16 == 0 for x in args.sizes):
        parser.error("message byte sizes must be multiples of 16")
    if not set(args.algos) <= {"push", "pull"}:
        parser.error("algos must be push,pull")
    if not set(args.workloads) <= {"pure", "interleaved"}:
        parser.error("workloads must be pure,interleaved")
    if min(args.loops, args.replays, args.repeats, args.matmul_dim) < 1:
        parser.error("loops/replays/repeats/matmul-dim must be positive")
    return args


def main():
    import torch
    import torch.distributed as dist

    import sglang.srt.distributed.parallel_state as ps
    from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.runtime_context import get_parallel

    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8:
        raise RuntimeError("This diagnostic requires exactly 8 torchrun ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise RuntimeError("This diagnostic is scoped to CUDA SM80")
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=240))
    # Identical topology bootstrap to the existing registered v2 kernel test.
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    group = coord.cpu_group
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    capacity = max(1024 * 1024, max(args.sizes))
    if rank == 0:
        print(json.dumps({"kind": "setup", "world": world, "gpu": torch.cuda.get_device_name(device),
                          "torch": torch.__version__, "args": vars(args)}), flush=True)

    def summarize_samples(samples):
        med = statistics.median(samples)
        return {"median_graph_ms": med, "min_graph_ms": min(samples),
                "max_graph_ms": max(samples), "median_us_per_ar": med * 1000 / args.loops,
                "samples_graph_ms": samples}

    @torch.inference_mode()
    def bench_case(comm, label, nbytes, workload):
        numel = nbytes // 2
        pattern = (torch.arange(numel, device=device, dtype=torch.int32) % 4).to(torch.bfloat16)
        inputs = (pattern[None, :].expand(args.loops, -1) + rank + 1).contiguous()
        expected = pattern.float() * world + world * (world + 1) / 2
        dim = args.matmul_dim
        # Identical operand distributions but rank-dependent numbers of tiny
        # GEMMs create device-ready skew without host sleeps or extra collectives.
        a = torch.ones((1, dim), device=device, dtype=torch.bfloat16)
        w = torch.eye(dim, device=device, dtype=torch.bfloat16)
        mm_out = torch.empty_like(a)
        graph = torch.cuda.CUDAGraph()
        # cuBLAS handles/workspaces are stream-associated and lazy initialized.
        # Warm on the SAME explicit stream used for capture: torch.cuda.graph's
        # implicit capture stream may otherwise first create a handle in capture.
        capture_stream = torch.cuda.current_stream(device)
        outputs = []
        # Warm eager launches also compile the exact dtype/world kernel before
        # graph capture. Legacy comm.capture warmup has placeholder semantics;
        # numeric assertions are performed only after actual graph replay.
        dist.barrier(group=group)
        for _ in range(2):
            if workload == "interleaved":
                torch.mm(a, w, out=mm_out)
            comm.custom_all_reduce(inputs[0])
        torch.cuda.synchronize()
        dist.barrier(group=group)
        with comm.capture():
            with torch.cuda.graph(graph, stream=capture_stream):
                for i in range(args.loops):
                    if workload == "interleaved":
                        count = 1 + ((rank + i) % 3)
                        for _ in range(count):
                            torch.mm(a, w, out=mm_out)
                    outputs.append(comm.custom_all_reduce(inputs[i]))
        torch.cuda.synchronize()
        dist.barrier(group=group)

        def validate(offset):
            graph.replay()
            torch.cuda.synchronize()
            stacked = torch.stack(outputs).float()
            reference = (expected + world * offset).expand_as(stacked)
            torch.testing.assert_close(stacked, reference, atol=0, rtol=0)

        validate(0)
        # Confirm pointer registration reads current inputs on replay rather
        # than stale captured values, and checks every collective output.
        inputs.add_(1)
        validate(1)
        inputs.sub_(1)
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        local_times = []
        local_wall = []
        for _ in range(args.repeats):
            dist.barrier(group=group)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            local_times.append(start.elapsed_time(end) / args.replays)
            local_wall.append((time.perf_counter() - wall_start) * 1000 / args.replays)
        gathered = [None] * world
        dist.all_gather_object(gathered, {"gpu_ms": local_times, "wall_ms": local_wall}, group=group)
        if rank == 0:
            # Max across ranks per sample, not the best/fastest rank. Each event
            # span includes graph queueing and device-side waits between nodes.
            worst = [max(row["gpu_ms"][i] for row in gathered) for i in range(args.repeats)]
            row = {"kind": "result", "backend": label, "bytes": nbytes,
                   "dtype": "bfloat16", "workload": workload, "ar_per_graph": args.loops,
                   "replays_per_sample": args.replays, "numeric": "exact PASS",
                   **summarize_samples(worst),
                   "per_rank_median_graph_ms": [statistics.median(x["gpu_ms"]) for x in gathered],
                   "max_rank_median_wall_ms": max(statistics.median(x["wall_ms"]) for x in gathered)}
            print(json.dumps(row), flush=True)
        del graph, outputs, inputs, pattern, a, w, mm_out
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    variants = [("v2", b, algo) for b in args.blocks for algo in args.algos]
    if args.include_v1:
        variants.append(("v1", None, None))
    for impl, blocks, algo in variants:
        if impl == "v2":
            comm = CustomAllReduceV2(group, device, max_size=capacity,
                                    max_pull_size=capacity, max_push_size=capacity,
                                    max_push_blocks=blocks, max_pull_blocks=blocks)
            comm.override_algo = {"push": AllReduceAlgo.ONE_SHOT_PUSH,
                                  "pull": AllReduceAlgo.ONE_SHOT_PULL}[algo]
            label = f"v2-{algo}-blocks{blocks}"
        else:
            comm = CustomAllreduce(group, device, max_size=capacity)
            label = "v1-legacy"
        if comm.disabled:
            raise RuntimeError(f"{label} communicator disabled on rank {rank}")
        for workload in args.workloads:
            for size in args.sizes:
                bench_case(comm, label, size, workload)
        dist.barrier(group=group)
        comm.close()
        # v2 close does not null obj; disable to avoid destructor double teardown.
        comm.disabled = True
        del comm
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier(group=group)
    if rank == 0:
        print(json.dumps({"kind": "complete", "status": "PASS"}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Communication benchmark for GLM-5.2 PP transport patterns.

Measures send/recv, all_reduce, all_gather, reduce_scatter latency and
bandwidth across two GPUs using torch.distributed.

Usage:
    torchrun --standalone --nproc_per_node=2 \
        scripts/perf/bench_glm52_pp_transport.py \
        --backend nccl --warmup 20 --iterations 100 \
        --output-json /tmp/glm52_comm_metrics.json
"""
import argparse, json, os, statistics, time
import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="PP transport benchmark")
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--sizes-bytes", default="4096,16384,65536,262144,1048576,4194304,16777216,67108864,268435456")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--output-json", default=None)
    p.add_argument("--hidden-size", type=int, default=5120)
    p.add_argument("--capture-layers", type=int, default=3)
    p.add_argument("--token-rows", type=int, default=4)
    return p.parse_args()


DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def measure_send_recv(rank, size, device, dtype, num_elems, warmup, iters, bidirectional):
    """Measure one-way (or bidirectional) send/recv latency."""
    tensor = torch.randn(num_elems, dtype=dtype, device=device)
    recv = torch.empty(num_elems, dtype=dtype, device=device)
    latencies = []

    for i in range(warmup + iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        if rank == 0:
            dist.send(tensor, dst=1)
            if bidirectional:
                dist.recv(recv, src=1)
        else:
            dist.recv(recv, src=0)
            if bidirectional:
                dist.send(tensor, dst=0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            latencies.append(t1 - t0)

    return latencies


def measure_collective(rank, size, device, dtype, num_elems, warmup, iters, op):
    """Measure all_reduce / all_gather / reduce_scatter latency."""
    tensor = torch.randn(num_elems, dtype=dtype, device=device)
    latencies = []

    for i in range(warmup + iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        if op == "all_reduce":
            dist.all_reduce(tensor)
        elif op == "all_gather":
            out = [torch.empty_like(tensor) for _ in range(size)]
            dist.all_gather(out, tensor)
        elif op == "reduce_scatter":
            chunk = num_elems // size
            inp = [torch.randn(chunk, dtype=dtype, device=device) for _ in range(size)]
            out = torch.empty(chunk, dtype=dtype, device=device)
            dist.reduce_scatter(out, inp)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup:
            latencies.append(t1 - t0)

    return latencies


def compute_stats(latencies, bytes_transferred, bidirectional=False):
    if not latencies:
        return None
    s = sorted(latencies)
    n = len(s)
    median = s[n // 2]
    p95 = s[int(n * 0.95)] if n > 1 else s[0]
    mean = sum(s) / n
    min_lat = s[0]
    effective_gbs = bytes_transferred / median / 1e9 if median > 0 else 0
    return {
        "min_latency_us": min_lat * 1e6,
        "median_latency_us": median * 1e6,
        "p95_latency_us": p95 * 1e6,
        "mean_latency_us": mean * 1e6,
        "effective_gbs": effective_gbs,
        "iterations": n,
    }


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    if args.backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    size = dist.get_world_size()

    sizes = [int(x) for x in args.sizes_bytes.split(",")]
    dtype_size = torch.finfo(dtype).bits // 8

    all_results = []

    # Payload calculations
    hs = args.hidden_size
    cl = args.capture_layers
    tr = args.token_rows
    payload = {
        "hidden_states_bytes": tr * hs * dtype_size,
        "residual_bytes": tr * hs * dtype_size,
        "aux_bytes": tr * cl * hs * dtype_size,
        "combined_pp_payload_bytes": tr * hs * dtype_size * (2 + cl),
    }
    if rank == 0:
        print(f"GLM-5.2 PP Payload Estimate (hidden_size={hs}, capture_layers={cl}, token_rows={tr}):")
        for k, v in payload.items():
            print(f"  {k}: {v} bytes ({v/1024:.1f} KiB)")

    # Send/recv benchmark
    for sz in sizes:
        num_elems = sz // dtype_size
        if num_elems < 1:
            continue
        lats = measure_send_recv(rank, size, device, dtype, num_elems,
                                  args.warmup, args.iterations, args.bidirectional)
        stats = compute_stats(lats, sz, args.bidirectional)
        if stats:
            stats["operation"] = "send_recv" + ("_bidir" if args.bidirectional else "")
            stats["bytes"] = sz
            stats["dtype"] = args.dtype
            all_results.append(stats)
            if rank == 0:
                print(f"send_recv {sz:>12d}B  median={stats['median_latency_us']:.2f}us  "
                      f"bw={stats['effective_gbs']:.2f} GB/s")

    # Collective benchmarks
    for op in ["all_reduce", "all_gather", "reduce_scatter"]:
        for sz in sizes:
            num_elems = sz // dtype_size
            if num_elems < 1:
                continue
            if op == "reduce_scatter" and num_elems < size:
                continue
            lats = measure_collective(rank, size, device, dtype, num_elems,
                                       args.warmup, args.iterations, op)
            stats = compute_stats(lats, sz)
            if stats:
                stats["operation"] = op
                stats["bytes"] = sz
                stats["dtype"] = args.dtype
                all_results.append(stats)
                if rank == 0:
                    print(f"{op:20s} {sz:>12d}B  median={stats['median_latency_us']:.2f}us  "
                          f"bw={stats['effective_gbs']:.2f} GB/s")

    if args.output_json and rank == 0:
        out = {
            "rank": rank,
            "world_size": size,
            "backend": args.backend,
            "device": str(device),
            "dtype": args.dtype,
            "payload_estimate": payload,
            "results": all_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

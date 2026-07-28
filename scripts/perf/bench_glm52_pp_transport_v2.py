#!/usr/bin/env python3
"""Corrected communication benchmark for GLM-5.2 PP transport.

Implements proper timing protocols:
  - one_way_with_ack: send payload + receive 1-byte ack
  - ping_pong: send + receive back, report RTT and RTT/2 estimate
  - all_reduce, all_gather, reduce_scatter with CUDA event timing

Uses CUDA events for NCCL operations to measure completion, not enqueue.

Usage:
    torchrun --standalone --nproc_per_node=2 \
        scripts/perf/bench_glm52_pp_transport_v2.py \
        --backend nccl --warmup 20 --iterations 100 \
        --output-json /tmp/nccl_raw.json
"""
import argparse, json, os, math
import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="PP transport benchmark v2")
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--sizes-bytes",
                   default="4096,16384,65536,204800,262144,1048576,4194304,16777216,67108864,268435456")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output-json", default=None)
    p.add_argument("--contamination", default="LIGHTLY_CONTENDED")
    # GLM-derived payload sizes
    p.add_argument("--hidden-size", type=int, default=5120)
    p.add_argument("--capture-layers", type=int, default=3)
    p.add_argument("--token-rows", default="1,4,16,64,256,1024")
    return p.parse_args()


DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def compute_stats(latencies_us, payload_bytes, protocol="one_way"):
    if not latencies_us:
        return None
    s = sorted(latencies_us)
    n = len(s)
    median = s[n // 2]
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    p95 = s[p95_idx] if n > 1 else s[0]
    p99 = s[p99_idx] if n > 1 else s[0]
    mean = sum(s) / n
    if n > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in s) / (n - 1))
        cv = std / mean if mean > 0 else 0
    else:
        std = 0
        cv = 0
    effective_gbs = payload_bytes / median * 1e-3 if median > 0 else 0  # bytes/us = GB/s

    warnings = []
    if median <= 0:
        warnings.append("SUSPICIOUS_RESULT: zero or negative latency")
    if n > 1 and p95 < median:
        warnings.append("SUSPICIOUS_RESULT: p95 below median")
    if n > 1 and p99 < p95:
        warnings.append("SUSPICIOUS_RESULT: p99 below p95")
    # PCIe Gen5 x16 theoretical: ~63 GB/s one direction
    # But H100 PCIe is Gen5 x16: ~128 GB/s bidirectional, ~64 GB/s one direction
    # With SHM fallback, expect much less
    if effective_gbs > 100 and protocol != "ping_pong":
        warnings.append(f"SUSPICIOUS_RESULT: effective BW={effective_gbs:.1f} GB/s exceeds PCIe Gen5 x16 ceiling")

    return {
        "median_latency_us": median,
        "p95_latency_us": p95,
        "p99_latency_us": p99,
        "mean_latency_us": mean,
        "std_us": std,
        "cv": cv,
        "effective_gbs": effective_gbs,
        "iterations": n,
        "warnings": warnings,
    }


def measure_one_way_with_ack(rank, device, dtype, num_elems, warmup, iters):
    """One-way send/recv with acknowledgement.

    Rank 0: start_event -> send payload -> recv 1-byte ack -> end_event
    Rank 1: recv payload -> validate -> send 1-byte ack

    Measures upper-bound end-to-end one-way completion time.
    """
    payload = torch.randn(num_elems, dtype=dtype, device=device)
    recv_buf = torch.empty(num_elems, dtype=dtype, device=device)
    ack = torch.tensor([1], dtype=torch.int8, device=device)
    ack_recv = torch.tensor([0], dtype=torch.int8, device=device)

    latencies = []

    for i in range(warmup + iters):
        if rank == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            dist.send(payload, dst=1)
            dist.recv(ack_recv, src=1)
            end.record()
            end.synchronize()
            if i >= warmup:
                latencies.append(start.elapsed_time(end) * 1000)  # ms -> us
        else:
            dist.barrier()
            dist.recv(recv_buf, src=0)
            # Validate
            assert recv_buf.shape == payload.shape
            dist.send(ack, dst=0)

    return latencies


def measure_ping_pong(rank, device, dtype, num_elems, warmup, iters):
    """Ping-pong: Rank 0 sends payload, Rank 1 returns it.

    Measures RTT. RTT/2 is an estimate of one-way latency.
    """
    payload = torch.randn(num_elems, dtype=dtype, device=device)
    recv_buf = torch.empty(num_elems, dtype=dtype, device=device)

    latencies = []

    for i in range(warmup + iters):
        if rank == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            dist.send(payload, dst=1)
            dist.recv(recv_buf, src=1)
            end.record()
            end.synchronize()
            if i >= warmup:
                latencies.append(start.elapsed_time(end) * 1000)  # ms -> us
        else:
            dist.barrier()
            dist.recv(recv_buf, src=0)
            dist.send(recv_buf, dst=0)

    return latencies


def measure_collective(rank, size, device, dtype, num_elems, warmup, iters, op):
    """Measure all_reduce / all_gather / reduce_scatter with CUDA event timing."""
    tensor = torch.randn(num_elems, dtype=dtype, device=device)
    latencies = []

    for i in range(warmup + iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()

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

        end.record()
        end.synchronize()
        if i >= warmup:
            latencies.append(start.elapsed_time(end) * 1000)

    return latencies


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

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    sizes = [int(x) for x in args.sizes_bytes.split(",")]
    dtype_size = torch.finfo(dtype).bits // 8

    # GLM-derived payload sizes
    token_rows_list = [int(x) for x in args.token_rows.split(",")]
    glm_payloads = {}
    for tr in token_rows_list:
        hs_bytes = tr * args.hidden_size * dtype_size
        res_bytes = tr * args.hidden_size * dtype_size
        aux_bytes = tr * args.capture_layers * args.hidden_size * dtype_size
        combined = hs_bytes + res_bytes + aux_bytes
        glm_payloads[f"hidden_only_M{tr}"] = hs_bytes
        glm_payloads[f"hidden+residual_M{tr}"] = hs_bytes + res_bytes
        glm_payloads[f"hidden+residual+aux_M{tr}"] = combined

    all_sizes = list(set(sizes + list(glm_payloads.values())))
    all_sizes.sort()

    all_results = []

    if rank == 0:
        print(f"GPU: {gpu_name}, backend={args.backend}, dtype={args.dtype}")
        print(f"Contamination: {args.contamination}")
        print(f"GLM-derived payloads: {glm_payloads}")

    # P2P status
    p2p_status = "unavailable (SHM/host-mediated fallback)"
    if device.type == "cuda":
        # Check P2P
        can_p2p = torch.cuda.can_device_access_peer(0, 1) if size > 1 else False
        p2p_status = "available" if can_p2p else "unavailable (SHM/host-mediated fallback)"

    # One-way with ack
    for sz in all_sizes:
        num_elems = sz // dtype_size
        if num_elems < 1:
            continue
        lats = measure_one_way_with_ack(rank, device, dtype, num_elems, args.warmup, args.iterations)
        if rank == 0:
            stats = compute_stats(lats, sz, "one_way_with_ack")
            if stats:
                stats["operation"] = "send_recv_one_way_with_ack"
                stats["protocol"] = "one_way_with_ack"
                stats["bytes"] = sz
                stats["dtype"] = args.dtype
                stats["direction"] = "rank0_to_rank1"
                stats["rank"] = rank
                stats["nccl_transport"] = p2p_status
                stats["p2p_status"] = p2p_status
                stats["contamination"] = args.contamination
                stats["gpu"] = gpu_name
                all_results.append(stats)
                print(f"one_way_ack  {sz:>12d}B  median={stats['median_latency_us']:.2f}us  "
                      f"bw={stats['effective_gbs']:.2f} GB/s")

    # Ping-pong
    for sz in all_sizes:
        num_elems = sz // dtype_size
        if num_elems < 1:
            continue
        lats = measure_ping_pong(rank, device, dtype, num_elems, args.warmup, args.iterations)
        if rank == 0:
            stats = compute_stats(lats, sz, "ping_pong")
            if stats:
                # RTT/2 estimate
                rtt_half = stats["median_latency_us"] / 2
                stats["rtt_us"] = stats["median_latency_us"]
                stats["rtt_half_estimate_us"] = rtt_half
                stats["effective_gbs"] = sz / rtt_half * 1e-3 if rtt_half > 0 else 0
                stats["operation"] = "ping_pong"
                stats["protocol"] = "ping_pong"
                stats["bytes"] = sz
                stats["dtype"] = args.dtype
                stats["direction"] = "bidirectional"
                stats["rank"] = rank
                stats["nccl_transport"] = p2p_status
                stats["p2p_status"] = p2p_status
                stats["contamination"] = args.contamination
                stats["gpu"] = gpu_name
                stats["note"] = "RTT/2 is an estimate, not a direct measurement"
                all_results.append(stats)
                print(f"ping_pong    {sz:>12d}B  RTT={stats['rtt_us']:.2f}us  "
                      f"RTT/2={rtt_half:.2f}us  est_BW={stats['effective_gbs']:.2f} GB/s")

    # Collectives
    for op in ["all_reduce", "all_gather", "reduce_scatter"]:
        for sz in all_sizes:
            num_elems = sz // dtype_size
            if num_elems < 1:
                continue
            if op == "reduce_scatter" and num_elems < size:
                continue
            lats = measure_collective(rank, size, device, dtype, num_elems, args.warmup, args.iterations, op)
            if rank == 0:
                stats = compute_stats(lats, sz, op)
                if stats:
                    stats["operation"] = op
                    stats["protocol"] = "collective"
                    stats["bytes"] = sz
                    stats["dtype"] = args.dtype
                    stats["direction"] = "all_to_all"
                    stats["rank"] = rank
                    stats["nccl_transport"] = p2p_status
                    stats["p2p_status"] = p2p_status
                    stats["contamination"] = args.contamination
                    stats["gpu"] = gpu_name
                    all_results.append(stats)
                    print(f"{op:20s} {sz:>12d}B  median={stats['median_latency_us']:.2f}us  "
                          f"bw={stats['effective_gbs']:.2f} GB/s")

    if args.output_json and rank == 0:
        out = {
            "gpu_name": gpu_name,
            "rank": rank,
            "world_size": size,
            "backend": args.backend,
            "device": str(device),
            "dtype": args.dtype,
            "p2p_status": p2p_status,
            "contamination": args.contamination,
            "glm_payloads": glm_payloads,
            "results": all_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare old vs packed PP transport on real SGLang GroupCoordinator.

Modes:
  A. old: existing send_tensor_dict / recv_tensor_dict
  B. packed: single contiguous send (no schema cache, no static buffers)
  C. packed_cached: packed with schema cache
  D. packed_static: packed with schema cache + static buffers

Uses production GroupCoordinator.send_tensor_dict / recv_tensor_dict for old path.
Uses direct dist.send/recv for packed path with pack/unpack helpers.

Usage:
    torchrun --standalone --nproc_per_node=2 \
        scripts/perf/bench_pp_transport_comparison.py \
        --backend nccl --warmup 20 --iterations 100 --trials 3 \
        --output-json /tmp/pp_comparison.json
"""
import argparse, json, os, math, time, struct
import torch
import torch.distributed as dist

def parse_args():
    p = argparse.ArgumentParser(description="PP transport comparison")
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--output-json", default=None)
    p.add_argument("--contamination", default="LIGHTLY_CONTENDED")
    p.add_argument("--hidden-size", type=int, default=5120)
    p.add_argument("--capture-layers", type=int, default=3)
    p.add_argument("--topk-size", type=int, default=64)
    p.add_argument("--token-rows", default="1,4,16,64,256,1024")
    return p.parse_args()

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}

def compute_stats(latencies_us):
    if not latencies_us:
        return None
    s = sorted(latencies_us)
    n = len(s)
    median = s[n // 2]
    p95 = s[min(int(n * 0.95), n - 1)] if n > 1 else s[0]
    p99 = s[min(int(n * 0.99), n - 1)] if n > 1 else s[0]
    mean = sum(s) / n
    if n > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in s) / (n - 1))
        cv = std / mean if mean > 0 else 0
    else:
        std = 0; cv = 0
    return {
        "median_latency_us": median, "p95_latency_us": p95, "p99_latency_us": p99,
        "mean_latency_us": mean, "std_us": std, "cv": cv, "iterations": n,
    }

def make_full_dict(rows, hs, cl, tk, dtype, device):
    from sglang.srt.speculative.glm52_eagle3_pp import (
        GLM52_EAGLE3_AUX_PP_KEY, allocate_packed_aux_buffer,
    )
    return {
        "hidden_states": torch.randn(rows, hs, dtype=dtype, device=device),
        "residual": torch.randn(rows, hs, dtype=dtype, device=device),
        "topk_indices": torch.randint(0, tk, (rows, tk), dtype=torch.int32, device=device),
        GLM52_EAGLE3_AUX_PP_KEY: allocate_packed_aux_buffer(rows, cl, hs, dtype, device),
    }

def benchmark_old(rank, send_dict, pp_group, warmup, iters, device):
    latencies = []
    for i in range(warmup + iters):
        if rank == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            pp_group.send_tensor_dict(tensor_dict=send_dict, async_send=False)
            end.record()
            end.synchronize()
            if i >= warmup:
                latencies.append(start.elapsed_time(end) * 1000)
        else:
            dist.barrier()
            _ = pp_group.recv_tensor_dict()
    return latencies

def benchmark_packed(rank, send_dict, warmup, iters, device, hidden_size,
                     capture_layers, topk_size, dtype, use_cache=False, use_static=False):
    from sglang.srt.distributed.pp_packed_transport import (
        calculate_pp_buffer_layout, pack_pp_proxy_tensors, unpack_pp_proxy_tensors,
        PPSchemaEntry, PPSchemaCache, PPStaticBufferRegistry,
        get_send_schema_cache, get_recv_schema_cache,
    )
    bucket = max(send_dict["hidden_states"].shape[0], 16)
    # Round up to power of 2
    bucket = 1 << (bucket - 1).bit_length()
    bucket = max(bucket, 16)

    key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
        send_dict, hidden_size, capture_layers, topk_size, bucket
    )

    send_cache = get_send_schema_cache() if use_cache else PPSchemaCache()
    recv_cache = get_recv_schema_cache() if use_cache else PPSchemaCache()

    static_reg = PPStaticBufferRegistry(device) if use_static else None

    latencies = []
    for i in range(warmup + iters):
        rows = send_dict["hidden_states"].shape[0]
        if rank == 0:
            # Get or allocate buffers
            if use_static:
                data_buf, ctrl_buf = static_reg.get_or_allocate(bucket, data_nelem, ctrl_nelem, dtype)
            else:
                data_buf = torch.zeros(max(data_nelem, 1), dtype=dtype, device=device)
                ctrl_buf = torch.zeros(max(ctrl_nelem, 1), dtype=torch.int32, device=device)

            # Pack
            pack_pp_proxy_tensors(send_dict, data_buf, ctrl_buf, d_off, c_off, rows)

            # Schema handling
            entry = send_cache.lookup(key)
            if entry is None:
                entry = send_cache.register(key, data_nelem, ctrl_nelem, d_off, c_off)

            # Send: header (CPU) + data + control
            header = [entry.schema_id, rows, key.presence_mask]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # Send header as a simple GPU tensor
            hdr_tensor = torch.tensor(header, dtype=torch.long, device=device)
            dist.send(hdr_tensor, dst=1)

            # Send data buffer
            dist.send(data_buf[:data_nelem], dst=1)
            # Send control buffer
            if ctrl_nelem > 0:
                dist.send(ctrl_buf[:ctrl_nelem], dst=1)

            end.record()
            end.synchronize()
            if i >= warmup:
                latencies.append(start.elapsed_time(end) * 1000)
        else:
            # Receive header
            hdr_tensor = torch.empty(3, dtype=torch.long, device=device)
            dist.recv(hdr_tensor, src=0)
            header_recv = hdr_tensor.tolist()

            schema_id, active_rows, presence = header_recv

            # Lookup or wait for schema
            entry = recv_cache.get_by_id(schema_id)
            if entry is None:
                # First time: register from known key
                entry = recv_cache.register(key, data_nelem, ctrl_nelem, d_off, c_off)

            if use_static:
                data_buf, ctrl_buf = static_reg.get_or_allocate(bucket, data_nelem, ctrl_nelem, dtype)
            else:
                data_buf = torch.empty(max(data_nelem, 1), dtype=dtype, device=device)
                ctrl_buf = torch.empty(max(ctrl_nelem, 1), dtype=torch.int32, device=device)


            dist.recv(data_buf[:data_nelem], src=0)
            if ctrl_nelem > 0:
                dist.recv(ctrl_buf[:ctrl_nelem], src=0)

            # Unpack
            result = unpack_pp_proxy_tensors(
                data_buf, ctrl_buf, entry, active_rows, device,
                dtype, hidden_size, capture_layers, topk_size
            )
            # Validate
            assert "hidden_states" in result
            assert result["hidden_states"].shape[0] == active_rows

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

    from sglang.srt.distributed.parallel_state import GroupCoordinator
    ranks = list(range(size))
    pp_group = GroupCoordinator(
        group_ranks=[ranks], local_rank=local_rank if device.type == "cuda" else 0,
        torch_distributed_backend=args.backend,
        use_pynccl=False, use_pymscclpp=False, use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
        use_xpu_communicator=False, use_npu_communicator=False,
    )

    token_rows_list = [int(x) for x in args.token_rows.split(",")]
    all_results = []

    if rank == 0:
        print(f"GPU: {gpu_name}, backend={args.backend}, dtype={args.dtype}")
        print(f"Hidden size: {args.hidden_size} (ASSUMED for GLM-5.2)")
        print(f"Contamination: {args.contamination}")

    modes = [
        ("old", lambda r, d, w, i: benchmark_old(r, d, pp_group, w, i, device)),
        ("packed", lambda r, d, w, i: benchmark_packed(r, d, w, i, device, args.hidden_size, args.capture_layers, args.topk_size, dtype, use_cache=False, use_static=False)),
        ("packed_cached", lambda r, d, w, i: benchmark_packed(r, d, w, i, device, args.hidden_size, args.capture_layers, args.topk_size, dtype, use_cache=True, use_static=False)),
        ("packed_static", lambda r, d, w, i: benchmark_packed(r, d, w, i, device, args.hidden_size, args.capture_layers, args.topk_size, dtype, use_cache=True, use_static=True)),
    ]

    for mode_name, mode_fn in modes:
        for tr in token_rows_list:
            send_dict = make_full_dict(tr, args.hidden_size, args.capture_layers, args.topk_size, dtype, device)
            payload_bytes = sum(t.nelement() * t.element_size() for t in send_dict.values())

            trial_medians = []
            for trial in range(args.trials):
                send_dict = make_full_dict(tr, args.hidden_size, args.capture_layers, args.topk_size, dtype, device)
                lats = mode_fn(rank, send_dict, args.warmup, args.iterations)
                if rank == 0:
                    stats = compute_stats(lats)
                    if stats:
                        trial_medians.append(stats["median_latency_us"])

            if rank == 0 and trial_medians:
                trial_medians.sort()
                med = trial_medians[len(trial_medians) // 2]
                eff_gbs = payload_bytes / med * 1e-3 if med > 0 else 0
                result = {
                    "mode": mode_name, "token_rows": tr, "payload_bytes": payload_bytes,
                    "median_latency_us": med,
                    "p95_latency_us": trial_medians[min(int(len(trial_medians) * 0.95), len(trial_medians) - 1)],
                    "effective_gbs": eff_gbs,
                    "trials": args.trials, "iterations_per_trial": args.iterations,
                    "gpu": gpu_name, "contamination": args.contamination,
                }
                all_results.append(result)
                print(f"  {mode_name:20s} M={tr:>5d}  bytes={payload_bytes:>10d}  "
                      f"median={med:.2f}us  bw={eff_gbs:.2f} GB/s")

    if args.output_json and rank == 0:
        with open(args.output_json, "w") as f:
            json.dump({"results": all_results, "gpu": gpu_name, "backend": args.backend,
                       "contamination": args.contamination}, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark real SGLang PP transport functions.

Imports and executes the actual production SGLang transport:
  - pp_group.send_tensor_dict()
  - pp_group.recv_tensor_dict()

These are the real functions used by scheduler_pp_mixin.py.

Tests dictionaries containing:
  - hidden_states only
  - hidden_states + residual
  - hidden_states + residual + EAGLE3 aux
  - hidden_states + residual + EAGLE3 aux + topk_indices
  - Result relay: next_token_ids, accept_lens, new_seq_lens, bonus tokens

Usage:
    torchrun --standalone --nproc_per_node=2 \
        scripts/perf/bench_sglang_pp_transport_real.py \
        --backend nccl --warmup 20 --iterations 100 \
        --output-json /tmp/sglang_pp_real.json
"""
import argparse, json, os, math, time
import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="Real SGLang PP transport benchmark")
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output-json", default=None)
    p.add_argument("--contamination", default="LIGHTLY_CONTENDED")
    p.add_argument("--hidden-size", type=int, default=5120, help="GLM-5.2 hidden_size (ASSUMED)")
    p.add_argument("--capture-layers", type=int, default=3)
    p.add_argument("--topk-size", type=int, default=64, help="Number of experts for topk_indices")
    p.add_argument("--token-rows", default="1,4,16,64,256,1024")
    p.add_argument("--batch-transitions", action="store_true", help="Test dynamic batch transitions")
    p.add_argument("--test-packed", action="store_true", help="Test packed payload alternative")
    return p.parse_args()


DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def compute_stats(latencies_us):
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
    return {
        "median_latency_us": median,
        "p95_latency_us": p95,
        "p99_latency_us": p99,
        "mean_latency_us": mean,
        "std_us": std,
        "cv": cv,
        "iterations": n,
    }


def make_proxy_dict_hidden_only(token_rows, hidden_size, dtype, device):
    return {
        "hidden_states": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
    }


def make_proxy_dict_hidden_residual(token_rows, hidden_size, dtype, device):
    return {
        "hidden_states": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
        "residual": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
    }


def make_proxy_dict_full(token_rows, hidden_size, capture_layers, topk_size, dtype, device):
    from sglang.srt.speculative.glm52_eagle3_pp import (
        GLM52_EAGLE3_AUX_PP_KEY,
        allocate_packed_aux_buffer,
    )
    return {
        "hidden_states": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
        "residual": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
        "topk_indices": torch.randint(0, topk_size, (token_rows, topk_size), dtype=torch.int32, device=device),
        GLM52_EAGLE3_AUX_PP_KEY: allocate_packed_aux_buffer(
            token_rows, capture_layers, hidden_size, dtype, device
        ),
    }


def make_proxy_dict_result_relay(token_rows, hidden_size, dtype, device):
    """Simulate PP result relay: next_token_ids, accept_lens, bonus tokens."""
    return {
        "hidden_states": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
        "next_token_ids": torch.randint(0, 151552, (token_rows,), dtype=torch.int64, device=device),
        "accept_lens": torch.randint(1, 8, (token_rows,), dtype=torch.int32, device=device),
        "new_seq_lens": torch.randint(1, 200000, (token_rows,), dtype=torch.int32, device=device),
        "bonus_tokens": torch.randint(0, 151552, (token_rows,), dtype=torch.int64, device=device),
    }


def make_proxy_dict_verify_chain(token_rows, hidden_size, capture_layers, topk_size, dtype, device):
    """Simulate next verify chain: hidden + residual + aux + topk + next_token_ids."""
    from sglang.srt.speculative.glm52_eagle3_pp import (
        GLM52_EAGLE3_AUX_PP_KEY,
        allocate_packed_aux_buffer,
    )
    return {
        "hidden_states": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
        "residual": torch.randn(token_rows, hidden_size, dtype=dtype, device=device),
        "topk_indices": torch.randint(0, topk_size, (token_rows, topk_size), dtype=torch.int32, device=device),
        GLM52_EAGLE3_AUX_PP_KEY: allocate_packed_aux_buffer(
            token_rows, capture_layers, hidden_size, dtype, device
        ),
        "next_token_ids": torch.randint(0, 151552, (token_rows,), dtype=torch.int64, device=device),
        "accept_lens": torch.randint(1, 8, (token_rows,), dtype=torch.int32, device=device),
    }


def benchmark_send_recv_dict(rank, send_dict, pp_group, warmup, iters, device):
    """Benchmark real send_tensor_dict / recv_tensor_dict."""
    latencies = []

    for i in range(warmup + iters):
        if rank == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            p2p_works = pp_group.send_tensor_dict(
                tensor_dict=send_dict,
                async_send=False,
            )
            end.record()
            end.synchronize()
            if i >= warmup:
                latencies.append(start.elapsed_time(end) * 1000)
        else:
            dist.barrier()
            recv_dict = pp_group.recv_tensor_dict()
            # Validate
            for key in send_dict:
                assert key in recv_dict, f"Missing key: {key}"
                assert recv_dict[key].shape == send_dict[key].shape, \
                    f"Shape mismatch for {key}: {recv_dict[key].shape} vs {send_dict[key].shape}"

    return latencies


def benchmark_packed_send_recv(rank, token_rows, hidden_size, capture_layers, topk_size,
                                dtype, device, pp_group, warmup, iters):
    """Benchmark packed payload: pack all tensors into one contiguous buffer."""
    from sglang.srt.speculative.glm52_eagle3_pp import (
        GLM52_EAGLE3_AUX_PP_KEY,
        allocate_packed_aux_buffer,
    )

    # Calculate total payload
    dtype_size = torch.finfo(dtype).bits // 8
    hs_bytes = token_rows * hidden_size * dtype_size
    res_bytes = token_rows * hidden_size * dtype_size
    aux_bytes = token_rows * capture_layers * hidden_size * dtype_size
    topk_bytes = token_rows * topk_size * 4  # int32
    total_bytes = hs_bytes + res_bytes + aux_bytes + topk_bytes

    # Create packed buffer
    total_elements = total_bytes // dtype_size
    packed = torch.randn(total_elements, dtype=dtype, device=device)

    latencies = []

    for i in range(warmup + iters):
        if rank == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            # Send as single tensor
            dist.send(packed, dst=1)
            end.record()
            end.synchronize()
            if i >= warmup:
                latencies.append(start.elapsed_time(end) * 1000)
        else:
            dist.barrier()
            recv_buf = torch.empty_like(packed)
            dist.recv(recv_buf, src=0)

    return latencies, total_bytes


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

    # Get the PP group from SGLang's parallel state
    # We need to set up a PP group
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    # Create a simple PP group with 2 ranks
    ranks = list(range(size))
    pp_group = GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank if device.type == "cuda" else 0,
        torch_distributed_backend=args.backend,
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
    )

    token_rows_list = [int(x) for x in args.token_rows.split(",")]

    all_results = []

    if rank == 0:
        print(f"GPU: {gpu_name}, backend={args.backend}, dtype={args.dtype}")
        print(f"Hidden size: {args.hidden_size} (ASSUMED for GLM-5.2)")
        print(f"Contamination: {args.contamination}")
        print(f"PP group: send_tensor_dict / recv_tensor_dict")

    # P2P status
    p2p_status = "unavailable (SHM/host-mediated fallback)"
    if device.type == "cuda":
        can_p2p = torch.cuda.can_device_access_peer(0, 1) if size > 1 else False
        p2p_status = "available" if can_p2p else "unavailable (SHM/host-mediated fallback)"

    # Test each payload type
    payload_types = [
        ("hidden_only", lambda tr, hs, cl, tk, dt, dev: make_proxy_dict_hidden_only(tr, hs, dt, dev)),
        ("hidden+residual", lambda tr, hs, cl, tk, dt, dev: make_proxy_dict_hidden_residual(tr, hs, dt, dev)),
        ("hidden+residual+aux+topk", lambda tr, hs, cl, tk, dt, dev: make_proxy_dict_full(tr, hs, cl, tk, dt, dev)),
        ("result_relay", lambda tr, hs, cl, tk, dt, dev: make_proxy_dict_result_relay(tr, hs, dt, dev)),
        ("verify_chain", lambda tr, hs, cl, tk, dt, dev: make_proxy_dict_verify_chain(tr, hs, cl, tk, dt, dev)),
    ]

    for payload_name, make_fn in payload_types:
        for tr in token_rows_list:
            send_dict = make_fn(tr, args.hidden_size, args.capture_layers, args.topk_size, dtype, device)

            # Calculate payload bytes
            payload_bytes = sum(t.nelement() * t.element_size() for t in send_dict.values())

            trial_medians = []
            for trial in range(args.trials):
                # Recreate dict each trial
                send_dict = make_fn(tr, args.hidden_size, args.capture_layers, args.topk_size, dtype, device)
                lats = benchmark_send_recv_dict(rank, send_dict, pp_group, args.warmup, args.iterations, device)
                if rank == 0:
                    stats = compute_stats(lats)
                    if stats:
                        trial_medians.append(stats["median_latency_us"])

            if rank == 0 and trial_medians:
                trial_medians.sort()
                median_of_medians = trial_medians[len(trial_medians) // 2]
                effective_gbs = payload_bytes / median_of_medians * 1e-3 if median_of_medians > 0 else 0

                result = {
                    "payload_type": payload_name,
                    "token_rows": tr,
                    "hidden_size": args.hidden_size,
                    "capture_layers": args.capture_layers,
                    "topk_size": args.topk_size,
                    "dtype": args.dtype,
                    "payload_bytes": payload_bytes,
                    "median_latency_us": median_of_medians,
                    "p95_latency_us": trial_medians[min(int(len(trial_medians) * 0.95), len(trial_medians) - 1)],
                    "effective_gbs": effective_gbs,
                    "trials": args.trials,
                    "iterations_per_trial": args.iterations,
                    "production_functions": "pp_group.send_tensor_dict / pp_group.recv_tensor_dict",
                    "num_tensor_sends": len(send_dict),
                    "p2p_status": p2p_status,
                    "gpu": gpu_name,
                    "contamination": args.contamination,
                }
                all_results.append(result)
                print(f"  {payload_name:30s} M={tr:>5d}  bytes={payload_bytes:>10d}  "
                      f"median={median_of_medians:.2f}us  bw={effective_gbs:.2f} GB/s")

    # Packed payload test
    if args.test_packed:
        if rank == 0:
            print("\n=== Packed Payload (benchmark-only) ===")
        for tr in token_rows_list:
            trial_medians = []
            for trial in range(args.trials):
                lats, total_bytes = benchmark_packed_send_recv(
                    rank, tr, args.hidden_size, args.capture_layers, args.topk_size,
                    dtype, device, pp_group, args.warmup, args.iterations
                )
                if rank == 0:
                    stats = compute_stats(lats)
                    if stats:
                        trial_medians.append(stats["median_latency_us"])

            if rank == 0 and trial_medians:
                trial_medians.sort()
                median_of_medians = trial_medians[len(trial_medians) // 2]
                effective_gbs = total_bytes / median_of_medians * 1e-3 if median_of_medians > 0 else 0
                result = {
                    "payload_type": "packed_benchmark_only",
                    "token_rows": tr,
                    "hidden_size": args.hidden_size,
                    "dtype": args.dtype,
                    "payload_bytes": total_bytes,
                    "median_latency_us": median_of_medians,
                    "effective_gbs": effective_gbs,
                    "trials": args.trials,
                    "production_functions": "dist.send/recv single packed tensor (benchmark-only)",
                    "p2p_status": p2p_status,
                    "gpu": gpu_name,
                    "contamination": args.contamination,
                }
                all_results.append(result)
                print(f"  packed_benchmark_only            M={tr:>5d}  bytes={total_bytes:>10d}  "
                      f"median={median_of_medians:.2f}us  bw={effective_gbs:.2f} GB/s")

    # Batch transitions
    if args.batch_transitions and rank == 0:
        print("\n=== Dynamic Batch Transitions ===")
    if args.batch_transitions:
        transitions = [(16, 1), (16, 4), (8, 2), (32, 1)]
        for from_rows, to_rows in transitions:
            for trial in range(args.trials):
                send_dict_1 = make_proxy_dict_full(from_rows, args.hidden_size, args.capture_layers,
                                                    args.topk_size, dtype, device)
                send_dict_2 = make_proxy_dict_full(to_rows, args.hidden_size, args.capture_layers,
                                                    args.topk_size, dtype, device)

                if rank == 0:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    dist.barrier()
                    start.record()
                    pp_group.send_tensor_dict(tensor_dict=send_dict_1, async_send=False)
                    pp_group.send_tensor_dict(tensor_dict=send_dict_2, async_send=False)
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end) * 1000
                    if trial == 0 and rank == 0:
                        print(f"  {from_rows}->{to_rows}: {elapsed:.2f}us")
                else:
                    dist.barrier()
                    _ = pp_group.recv_tensor_dict()
                    _ = pp_group.recv_tensor_dict()

    if args.output_json and rank == 0:
        out = {
            "gpu_name": gpu_name,
            "backend": args.backend,
            "dtype": args.dtype,
            "hidden_size": args.hidden_size,
            "hidden_size_verified": False,
            "capture_layers": args.capture_layers,
            "p2p_status": p2p_status,
            "contamination": args.contamination,
            "production_functions": {
                "send": "sglang.srt.distributed.parallel_state.GroupCoordinator.send_tensor_dict",
                "recv": "sglang.srt.distributed.parallel_state.GroupCoordinator.recv_tensor_dict",
                "metadata_path": "send_object (CPU/Gloo) -> _split_tensor_dict metadata_list",
                "tensor_path": "isend/irecv per tensor in tensor_list",
                "sync_points": "metadata send (blocking) -> per-tensor irecv.wait()",
            },
            "results": all_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Benchmark custom RS+AG vs naive RS+AG vs custom AR.

This script compares:
1. Custom RS+AG (optimized, deterministic)
2. Naive RS+AG (torch.distributed, deterministic but slower)
3. Custom AR (faster but non-deterministic)

Usage:
    # Default TP=8
    python benchmark/kernels/all_reduce/benchmark_rs_ag.py
    
    # Custom TP size
    python benchmark/kernels/all_reduce/benchmark_rs_ag.py --tp 2
"""

import multiprocessing as mp
import socket
import time
import torch
import torch.distributed as dist
import statistics
import os

# Enable deterministic RS+AG for benchmarking (automatically enabled in script)
os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
os.environ["SGLANG_PREFER_CUSTOM_ALLREDUCE_FOR_DETERMINISM"] = "1"


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def naive_rs_ag(tensor, world_size, rank, group):
    """Naive RS+AG using torch.distributed."""
    tensor_flat = tensor.flatten()
    total_size = tensor_flat.numel()
    chunk_size = total_size // world_size
    
    # Reduce-scatter
    output_chunk = torch.empty(chunk_size, dtype=tensor.dtype, device=tensor.device)
    input_chunks = [tensor_flat[i*chunk_size:(i+1)*chunk_size].clone() for i in range(world_size)]
    torch.distributed.reduce_scatter(output_chunk, input_chunks, group=group)
    
    # All-gather
    output_flat = torch.empty(total_size, dtype=tensor.dtype, device=tensor.device)
    output_chunks = [output_flat[i*chunk_size:(i+1)*chunk_size] for i in range(world_size)]
    torch.distributed.all_gather(output_chunks, output_chunk, group=group)
    
    return output_flat.reshape(tensor.shape)


def worker(world_size, rank, port, results_queue):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from sglang.srt.distributed.parallel_state import initialize_model_parallel, get_tp_group
        from torch.distributed import new_group
        
        # Initialize model parallel (needed for GroupCoordinator)
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )
        
        # Get TP group coordinator (has proper group name for custom ops)
        tp_group = get_tp_group()
        
        # Create gloo group for custom AR
        dist.barrier()
        ar_group = new_group(backend="gloo")
        dist.barrier()
        
        custom_ar = CustomAllreduce(group=ar_group, device=device)
        
        if custom_ar is None or custom_ar.disabled:
            if rank == 0:
                print("ERROR: Custom AR not available or disabled")
            dist.destroy_process_group()
            return

        batch_sizes = [1, 4, 8, 16, 32, 50]
        hidden_dim = 16384
        num_trials = 20
        num_warmup = 3

        torch.manual_seed(42 + rank)
        results = {}

        for bs in batch_sizes:
            inp = torch.randn(bs, hidden_dim, dtype=torch.bfloat16, device=device)
            inp_flat = inp.flatten()
            total_size = inp_flat.numel()
            
            # Ensure divisible by world_size
            if total_size % world_size != 0:
                pad_size = world_size - (total_size % world_size)
                inp_flat = torch.cat([inp_flat, torch.zeros(pad_size, dtype=inp.dtype, device=device)])
                total_size = inp_flat.numel()
            
            chunk_size = total_size // world_size

            # Warmup
            for _ in range(num_warmup):
                _ = naive_rs_ag(inp, world_size, rank, ar_group)
            torch.cuda.synchronize()

            # Benchmark naive RS+AG
            times_naive = []
            for _ in range(num_trials):
                inp_copy = inp.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = naive_rs_ag(inp_copy, world_size, rank, ar_group)
                torch.cuda.synchronize()
                times_naive.append(time.perf_counter() - start)

            # Benchmark custom RS+AG (using parallel_state all_reduce with env var set)
            times_custom_rs_ag = []
            for _ in range(num_warmup):
                inp_copy = inp.clone()
                torch.cuda.synchronize()
                _ = tp_group.all_reduce(inp_copy)
                torch.cuda.synchronize()
            torch.cuda.synchronize()

            for _ in range(num_trials):
                inp_copy = inp.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = tp_group.all_reduce(inp_copy)
                torch.cuda.synchronize()
                times_custom_rs_ag.append(time.perf_counter() - start)

            # Benchmark custom AR (non-deterministic but faster)
            # Temporarily disable env var to test custom AR
            old_env = os.environ.get("SGLANG_PREFER_CUSTOM_ALLREDUCE_FOR_DETERMINISM")
            os.environ.pop("SGLANG_PREFER_CUSTOM_ALLREDUCE_FOR_DETERMINISM", None)
            
            times_custom_ar = []
            if custom_ar.should_custom_ar(inp):
                for _ in range(num_warmup):
                    inp_copy = inp.clone()
                    out = custom_ar.all_reduce(inp_copy, registered=False)
                torch.cuda.synchronize()

                for _ in range(num_trials):
                    inp_copy = inp.clone()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    out = custom_ar.all_reduce(inp_copy, registered=False)
                    torch.cuda.synchronize()
                    times_custom_ar.append(time.perf_counter() - start)
            
            # Restore env var
            if old_env:
                os.environ["SGLANG_PREFER_CUSTOM_ALLREDUCE_FOR_DETERMINISM"] = old_env

            if rank == 0:
                results[bs] = {
                    'naive_rs_ag': times_naive,
                    'custom_rs_ag': times_custom_rs_ag,
                    'custom_ar': times_custom_ar if times_custom_ar else None,
                }

        dist.barrier()
        dist.destroy_process_group()

        if rank == 0:
            results_queue.put(results)

    except Exception as e:
        if rank == 0:
            print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        dist.destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism size (default: 8)")
    args = parser.parse_args()
    
    world_size = args.tp
    available_gpus = torch.cuda.device_count()

    print("=" * 70)
    print("RS+AG Benchmark: Custom vs Naive vs Custom AR")
    print("=" * 70)
    print(f"Available GPUs: {available_gpus}")
    print(f"Requested TP size: {world_size}")

    if available_gpus < world_size:
        print(f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead")
        world_size = available_gpus

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this test")
        return
    
    print(f"Using TP size: {world_size}")

    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    results_queue = mp.Queue()

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(world_size, rank, port, results_queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Print results
    if not results_queue.empty():
        results = results_queue.get()
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"{'Batch Size':<12} {'Naive RS+AG':<15} {'Custom RS+AG':<15} {'Custom AR':<15} {'Speedup':<10}")
        print("-" * 70)

        for bs in sorted(results.keys()):
            naive = statistics.median(results[bs]['naive_rs_ag']) * 1000
            custom_rs_ag = statistics.median(results[bs]['custom_rs_ag']) * 1000
            custom_ar = statistics.median(results[bs]['custom_ar']) * 1000 if results[bs]['custom_ar'] else None

            speedup = naive / custom_rs_ag if custom_rs_ag > 0 else 0

            custom_ar_str = f"{custom_ar:.3f}ms" if custom_ar else "N/A"
            print(f"{bs:<12} {naive:>10.3f}ms {custom_rs_ag:>10.3f}ms {custom_ar_str:>15} {speedup:>8.2f}x")

        print(f"\n{'='*70}")
        print("Summary:")
        print("  - Custom RS+AG: Optimized, deterministic")
        print("  - Naive RS+AG: torch.distributed, deterministic but slower")
        print("  - Custom AR: Fastest but non-deterministic (uses atomics)")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

"""
Benchmark latency comparison between different all-reduce implementations.

Compares:
- NCCL all-reduce (may be non-deterministic)
- Reduce-scatter + all-gather (RS+AG, deterministic but slower)
- Deterministic 1-stage kernel (forces fixed accumulation order, deterministic)

Note: The "deterministic kernel" is NOT RS+AG. It uses the 1-stage kernel where
each GPU reads all data from all GPUs and reduces locally in a fixed order.

Usage:
    python bench_amd_deterministic_allreduce.py
"""

import multiprocessing as mp
import os
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist

# Add python directory to path to import sglang modules
script_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(script_dir, "python")
sys.path.insert(0, python_dir)

# Try to import custom all-reduce if available
try:
    import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as custom_ar_ops
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_weak_contiguous,
    )

    CUSTOM_AR_AVAILABLE = custom_ar_ops.IS_CUSTOM_AR_AVAILABLE
except (ImportError, AttributeError):
    CUSTOM_AR_AVAILABLE = False
    CustomAllreduce = None
    is_weak_contiguous = None

# Note: sglang's optimized all-reduce requires full runtime initialization
# and won't work in standalone benchmarks, so we skip it
SGLANG_AVAILABLE = False


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_custom_ar_if_available(rank, world_size, device):
    """Check if custom all-reduce is available and applicable."""
    if not CUSTOM_AR_AVAILABLE or CustomAllreduce is None:
        return False

    # Custom AR works best for single-node, even number of GPUs, world_size <= 8
    if world_size <= 8 and world_size % 2 == 0:
        return True

    return False


def reduce_scatter_then_all_gather(tensor, rank, world_size, custom_ar=None):
    """
    Deterministic all-reduce using reduce-scatter + all-gather.
    This is deterministic because it uses fixed ordering (no atomics).
    """
    total_size = tensor.numel()
    if total_size % world_size != 0:
        # Fallback to all-gather + local reduce if not divisible
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor)
        stacked = torch.stack(gather_list, dim=0)
        tensor.copy_(stacked.sum(dim=0))
        return

    chunk_size = total_size // world_size

    # Flatten to 1D
    tensor_flat = tensor.view(-1)

    # Reduce-scatter: each rank gets its chunk of the reduced result
    output_chunk = torch.empty(chunk_size, dtype=tensor.dtype, device=tensor.device)

    # Split input into chunks for reduce-scatter
    input_chunks = [
        tensor_flat[i * chunk_size : (i + 1) * chunk_size].clone()
        for i in range(world_size)
    ]

    dist.reduce_scatter(output_chunk, input_chunks)

    # All-gather: broadcast each rank's chunk to all ranks
    output_chunks = [
        torch.empty(chunk_size, dtype=tensor.dtype, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(output_chunks, output_chunk)

    # Concatenate results back
    result_flat = torch.cat(output_chunks, dim=0)
    tensor.copy_(result_flat.view(tensor.shape))


def worker(world_size, rank, port, results_queue):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    # Try to initialize custom all-reduce if available
    custom_ar = None
    use_custom_ar = init_custom_ar_if_available(rank, world_size, device)
    if use_custom_ar and CUSTOM_AR_AVAILABLE:
        try:
            # Create a gloo group for custom AR (it requires non-NCCL backend)
            # All ranks must call new_group with the same parameters
            from torch.distributed import new_group

            dist.barrier()  # Ensure all ranks are ready
            ar_group = new_group(backend="gloo")
            dist.barrier()  # Ensure group creation is complete
            custom_ar = CustomAllreduce(group=ar_group, device=device)
            if rank == 0:
                print("  Using custom all-reduce (deterministic)")
        except Exception as e:
            if rank == 0:
                print(f"  Custom AR init failed: {e}, using NCCL fallback")
            custom_ar = None
            dist.barrier()  # Ensure all ranks continue even if one fails

    # Test different batch sizes - similar to test_ar.py
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    hidden_dim = 16384  # Fixed hidden dimension

    num_trials = 10  # Same as test_ar.py

    # Different seed per rank - each GPU has DIFFERENT input (like test_ar.py)
    torch.manual_seed(42 + rank)

    results = {}

    for bs in batch_sizes:
        # Create fixed input for all trials (like test_ar.py)
        base_input = torch.randn(bs, hidden_dim, dtype=torch.bfloat16, device=device)

        dist.barrier()

        if rank == 0:
            print(f"\nBatch size {bs:4d}:")
            print(f"  Testing determinism across {num_trials} trials...")

        # Test all-reduce determinism
        results_ar = []
        latencies_ar = []
        for trial in range(num_trials):
            # Clone the same input for each trial
            inp_ar = base_input.clone()
            inp_flat_ar = inp_ar.view(-1)

            # Measure latency
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(inp_flat_ar, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ar.append(end - start)

            # Store checksum and first values (like test_ar.py)
            checksum = inp_flat_ar.sum().item()
            first_vals = inp_flat_ar[:5].clone()
            results_ar.append((checksum, first_vals))

        # Test reduce-scatter + all-gather determinism
        results_rs_ag = []
        latencies_rs_ag = []
        for trial in range(num_trials):
            # Clone the same input for each trial
            inp_rs_ag = base_input.clone()
            inp_flat_rs_ag = inp_rs_ag.view(-1)

            # Measure latency
            torch.cuda.synchronize()
            start = time.perf_counter()
            reduce_scatter_then_all_gather(
                inp_flat_rs_ag, rank, world_size, custom_ar=None
            )
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_rs_ag.append(end - start)

            # Store checksum and first values (like test_ar.py)
            checksum = inp_flat_rs_ag.sum().item()
            first_vals = inp_flat_rs_ag[:5].clone()
            results_rs_ag.append((checksum, first_vals))

        # Note: sglang's optimized all-reduce requires full runtime initialization
        # and is not tested in this standalone benchmark
        use_sglang_optimized = False
        results_optimized_rs_ag = []
        latencies_optimized_rs_ag = []

        # Test custom all-reduce determinism (if available)
        results_custom_ar = []
        latencies_custom_ar = []
        if custom_ar is not None:
            for trial in range(num_trials):
                # Clone the same input for each trial
                inp_custom = base_input.clone()
                inp_flat_custom = inp_custom.view(-1)

                # Measure latency
                torch.cuda.synchronize()
                start = time.perf_counter()
                reduce_scatter_then_all_gather(
                    inp_flat_custom, rank, world_size, custom_ar=custom_ar
                )
                torch.cuda.synchronize()
                end = time.perf_counter()
                latencies_custom_ar.append(end - start)

                # Store checksum and first values (like test_ar.py)
                checksum = inp_flat_custom.sum().item()
                first_vals = inp_flat_custom[:5].clone()
                results_custom_ar.append((checksum, first_vals))

        # Test deterministic kernel (if available)
        results_deterministic_kernel = []
        latencies_deterministic_kernel = []
        deterministic_kernel_available = False
        if custom_ar is not None and hasattr(custom_ar, "deterministic_all_reduce"):
            # Check if input size fits in buffer
            input_size_bytes = base_input.numel() * base_input.element_size()
            if input_size_bytes > custom_ar.max_size:
                if rank == 0:
                    print(
                        f"  Deterministic kernel skipped: input size ({input_size_bytes/(1024*1024):.1f} MB) > buffer size ({custom_ar.max_size/(1024*1024):.1f} MB)"
                    )
                deterministic_kernel_available = False
            else:
                try:
                    deterministic_kernel_available = True
                    for trial in range(num_trials):
                        # Clone the same input for each trial
                        inp_kernel = base_input.clone()

                        # Measure latency
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result_kernel = custom_ar.deterministic_all_reduce(
                            inp_kernel, registered=False
                        )
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        latencies_deterministic_kernel.append(end - start)

                        # Store checksum and first values
                        result_flat_kernel = result_kernel.view(-1)
                        checksum = result_flat_kernel.sum().item()
                        first_vals = result_flat_kernel[:5].clone()
                        results_deterministic_kernel.append((checksum, first_vals))
                except Exception as e:
                    if rank == 0:
                        print(
                            f"  Deterministic kernel test failed for batch size {bs}: {e}"
                        )
                    deterministic_kernel_available = False

        dist.barrier()

        if rank == 0:
            # Check determinism for all-reduce
            ar_deterministic = True
            ar_ref_sum, ar_ref_vals = results_ar[0]
            ar_variance = []
            for i, (s, vals) in enumerate(results_ar[1:], 1):
                if abs(ar_ref_sum - s) > 1e-3 or not torch.allclose(
                    ar_ref_vals, vals, rtol=1e-3
                ):
                    ar_deterministic = False
                ar_variance.append(abs(ar_ref_sum - s))

            # Check determinism for reduce-scatter + all-gather
            rs_ag_deterministic = True
            rs_ag_ref_sum, rs_ag_ref_vals = results_rs_ag[0]
            rs_ag_variance = []
            for i, (s, vals) in enumerate(results_rs_ag[1:], 1):
                if abs(rs_ag_ref_sum - s) > 1e-3 or not torch.allclose(
                    rs_ag_ref_vals, vals, rtol=1e-3
                ):
                    rs_ag_deterministic = False
                rs_ag_variance.append(abs(rs_ag_ref_sum - s))

            # Check determinism for optimized RS+AG (if available)
            optimized_rs_ag_deterministic = None
            optimized_rs_ag_max_variance = None
            lat_optimized_rs_ag_median = None
            if use_sglang_optimized and results_optimized_rs_ag:
                optimized_rs_ag_deterministic = True
                opt_rs_ag_ref_sum, opt_rs_ag_ref_vals = results_optimized_rs_ag[0]
                opt_rs_ag_variance = []
                for i, (s, vals) in enumerate(results_optimized_rs_ag[1:], 1):
                    if abs(opt_rs_ag_ref_sum - s) > 1e-3 or not torch.allclose(
                        opt_rs_ag_ref_vals, vals, rtol=1e-3
                    ):
                        optimized_rs_ag_deterministic = False
                    opt_rs_ag_variance.append(abs(opt_rs_ag_ref_sum - s))
                optimized_rs_ag_max_variance = (
                    max(opt_rs_ag_variance) if opt_rs_ag_variance else 0.0
                )
                lat_optimized_rs_ag_median = statistics.median(
                    latencies_optimized_rs_ag
                )

            # Check determinism for custom all-reduce (if available)
            custom_ar_deterministic = None
            custom_ar_max_variance = None
            lat_custom_ar_median = None
            if custom_ar is not None and results_custom_ar:
                custom_ar_deterministic = True
                custom_ar_ref_sum, custom_ar_ref_vals = results_custom_ar[0]
                custom_ar_variance = []
                for i, (s, vals) in enumerate(results_custom_ar[1:], 1):
                    if abs(custom_ar_ref_sum - s) > 1e-3 or not torch.allclose(
                        custom_ar_ref_vals, vals, rtol=1e-3
                    ):
                        custom_ar_deterministic = False
                    custom_ar_variance.append(abs(custom_ar_ref_sum - s))
                custom_ar_max_variance = (
                    max(custom_ar_variance) if custom_ar_variance else 0.0
                )
                lat_custom_ar_median = statistics.median(latencies_custom_ar)

            # Check determinism for deterministic kernel (if available)
            deterministic_kernel_deterministic = None
            deterministic_kernel_max_variance = None
            lat_deterministic_kernel_median = None
            if deterministic_kernel_available and results_deterministic_kernel:
                deterministic_kernel_deterministic = True
                kernel_ref_sum, kernel_ref_vals = results_deterministic_kernel[0]
                kernel_variance = []
                for i, (s, vals) in enumerate(results_deterministic_kernel[1:], 1):
                    if abs(kernel_ref_sum - s) > 1e-3 or not torch.allclose(
                        kernel_ref_vals, vals, rtol=1e-3
                    ):
                        deterministic_kernel_deterministic = False
                    kernel_variance.append(abs(kernel_ref_sum - s))
                deterministic_kernel_max_variance = (
                    max(kernel_variance) if kernel_variance else 0.0
                )
                lat_deterministic_kernel_median = statistics.median(
                    latencies_deterministic_kernel
                )

            # Calculate latency statistics
            lat_ar_median = statistics.median(latencies_ar)
            lat_rs_ag_median = statistics.median(latencies_rs_ag)
            overhead_rs_ag = ((lat_rs_ag_median - lat_ar_median) / lat_ar_median) * 100

            # Calculate variance statistics
            ar_max_variance = max(ar_variance) if ar_variance else 0.0
            rs_ag_max_variance = max(rs_ag_variance) if rs_ag_variance else 0.0

            results[bs] = {
                "all_reduce": {
                    "latency_median": lat_ar_median,
                    "deterministic": ar_deterministic,
                    "max_variance": ar_max_variance,
                },
                "rs_ag": {
                    "latency_median": lat_rs_ag_median,
                    "deterministic": rs_ag_deterministic,
                    "max_variance": rs_ag_max_variance,
                },
                "custom_ar": (
                    {
                        "latency_median": lat_custom_ar_median,
                        "deterministic": custom_ar_deterministic,
                        "max_variance": custom_ar_max_variance,
                    }
                    if custom_ar is not None
                    else None
                ),
                "deterministic_kernel": (
                    {
                        "latency_median": lat_deterministic_kernel_median,
                        "deterministic": deterministic_kernel_deterministic,
                        "max_variance": deterministic_kernel_max_variance,
                    }
                    if lat_deterministic_kernel_median is not None
                    else None
                ),
                "optimized_rs_ag": (
                    {
                        "latency_median": lat_optimized_rs_ag_median,
                        "deterministic": optimized_rs_ag_deterministic,
                        "max_variance": optimized_rs_ag_max_variance,
                    }
                    if lat_optimized_rs_ag_median is not None
                    else None
                ),
                "overhead_rs_ag_pct": overhead_rs_ag,
            }

            print(
                f"    All-Reduce:     {lat_ar_median*1000:.3f}ms, Deterministic: {ar_deterministic}, Max variance: {ar_max_variance:.6f}"
            )
            print(
                f"    RS+All-Gather:   {lat_rs_ag_median*1000:.3f}ms, Deterministic: {rs_ag_deterministic}, Max variance: {rs_ag_max_variance:.6f}"
            )
            if custom_ar is not None and lat_custom_ar_median is not None:
                overhead_custom = (
                    (lat_custom_ar_median - lat_ar_median) / lat_ar_median
                ) * 100
                print(
                    f"    Custom AR:       {lat_custom_ar_median*1000:.3f}ms, Deterministic: {custom_ar_deterministic}, Max variance: {custom_ar_max_variance:.6f}, Overhead: {overhead_custom:+.1f}%"
                )
            if lat_deterministic_kernel_median is not None:
                overhead_kernel = (
                    (lat_deterministic_kernel_median - lat_ar_median) / lat_ar_median
                ) * 100
                speedup_kernel_vs_rs_ag = (
                    (lat_rs_ag_median - lat_deterministic_kernel_median)
                    / lat_rs_ag_median
                ) * 100
                print(
                    f"    Deterministic Kernel: {lat_deterministic_kernel_median*1000:.3f}ms, Deterministic: {deterministic_kernel_deterministic}, Max variance: {deterministic_kernel_max_variance:.6f}, Overhead: {overhead_kernel:+.1f}%, Speedup vs RS+AG: {speedup_kernel_vs_rs_ag:+.1f}%"
                )
            if lat_optimized_rs_ag_median is not None:
                overhead_opt = (
                    (lat_optimized_rs_ag_median - lat_ar_median) / lat_ar_median
                ) * 100
                speedup_vs_rs_ag = (
                    (lat_rs_ag_median - lat_optimized_rs_ag_median) / lat_rs_ag_median
                ) * 100
                print(
                    f"    Optimized RS+AG: {lat_optimized_rs_ag_median*1000:.3f}ms, Deterministic: {optimized_rs_ag_deterministic}, Max variance: {optimized_rs_ag_max_variance:.6f}, Overhead: {overhead_opt:+.1f}%, Speedup vs RS+AG: {speedup_vs_rs_ag:+.1f}%"
                )
            print(f"    RS+AG Overhead: {overhead_rs_ag:+.1f}%")

    if rank == 0:
        results_queue.put(results)

    dist.destroy_process_group()


def main():
    world_size = 8
    available_gpus = torch.cuda.device_count()

    print("=" * 80)
    print("All-Reduce vs Reduce-Scatter + All-Gather Determinism & Latency Benchmark")
    print("=" * 80)
    print(f"Available GPUs: {available_gpus}")
    print(f"Using world_size: {world_size}")
    print(f"Hidden dimension: 16384")
    print(f"Tensor dtype: bfloat16")
    print(f"Trials per batch size: 10 (testing determinism)")
    print(f"Testing batch sizes: [1, 4, 8, 16, 32, 64, 128, 256, 512]")
    print("=" * 80)

    if available_gpus < world_size:
        print(
            f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead"
        )
        world_size = available_gpus

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this benchmark")
        return

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

    # Collect results
    if not results_queue.empty():
        results = results_queue.get()

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        header = f"{'Batch':<8} {'AR (ms)':<12} {'AR Det':<8} {'RS+AG (ms)':<15} {'RS+AG Det':<10} {'RS+AG Ovh':<12}"
        if any(r.get("custom_ar") is not None for r in results.values()):
            header += (
                f" {'Custom AR (ms)':<18} {'Custom AR Det':<15} {'Custom AR Ovh':<15}"
            )
        if any(r.get("deterministic_kernel") is not None for r in results.values()):
            header += f" {'Det Kernel (ms)':<18} {'Det Kernel Det':<15} {'Det Kernel Ovh':<15} {'Speedup':<10}"
        if any(r.get("optimized_rs_ag") is not None for r in results.values()):
            header += f" {'Opt RS+AG (ms)':<18} {'Opt RS+AG Det':<15} {'Opt RS+AG Ovh':<15} {'Speedup':<10}"
        print(header)
        print("-" * 150)

        for bs in sorted(results.keys()):
            r = results[bs]
            ar_det_str = "✓" if r["all_reduce"]["deterministic"] else "✗"
            rs_ag_det_str = "✓" if r["rs_ag"]["deterministic"] else "✗"
            line = (
                f"{bs:<8} {r['all_reduce']['latency_median']*1000:<12.3f} {ar_det_str:<8} "
                f"{r['rs_ag']['latency_median']*1000:<15.3f} {rs_ag_det_str:<10} "
                f"{r['overhead_rs_ag_pct']:<12.1f}"
            )
            if r.get("custom_ar") is not None:
                custom_ar = r["custom_ar"]
                custom_ar_det_str = "✓" if custom_ar["deterministic"] else "✗"
                custom_ar_overhead = (
                    (custom_ar["latency_median"] - r["all_reduce"]["latency_median"])
                    / r["all_reduce"]["latency_median"]
                ) * 100
                line += f" {custom_ar['latency_median']*1000:<18.3f} {custom_ar_det_str:<15} {custom_ar_overhead:<15.1f}"
            if r.get("deterministic_kernel") is not None:
                det_kernel = r["deterministic_kernel"]
                det_kernel_det_str = "✓" if det_kernel["deterministic"] else "✗"
                det_kernel_overhead = (
                    (det_kernel["latency_median"] - r["all_reduce"]["latency_median"])
                    / r["all_reduce"]["latency_median"]
                ) * 100
                speedup_kernel = (
                    (r["rs_ag"]["latency_median"] - det_kernel["latency_median"])
                    / r["rs_ag"]["latency_median"]
                ) * 100
                line += f" {det_kernel['latency_median']*1000:<18.3f} {det_kernel_det_str:<15} {det_kernel_overhead:<15.1f} {speedup_kernel:<10.1f}"
            if r.get("optimized_rs_ag") is not None:
                opt_rs_ag = r["optimized_rs_ag"]
                opt_rs_ag_det_str = "✓" if opt_rs_ag["deterministic"] else "✗"
                opt_rs_ag_overhead = (
                    (opt_rs_ag["latency_median"] - r["all_reduce"]["latency_median"])
                    / r["all_reduce"]["latency_median"]
                ) * 100
                speedup = (
                    (r["rs_ag"]["latency_median"] - opt_rs_ag["latency_median"])
                    / r["rs_ag"]["latency_median"]
                ) * 100
                line += f" {opt_rs_ag['latency_median']*1000:<18.3f} {opt_rs_ag_det_str:<15} {opt_rs_ag_overhead:<15.1f} {speedup:<10.1f}"
            print(line)

        print("=" * 80)

        # Calculate statistics
        overheads_rs_ag = [r["overhead_rs_ag_pct"] for r in results.values()]
        ar_deterministic_count = sum(
            1 for r in results.values() if r["all_reduce"]["deterministic"]
        )
        rs_ag_deterministic_count = sum(
            1 for r in results.values() if r["rs_ag"]["deterministic"]
        )
        custom_ar_deterministic_count = sum(
            1
            for r in results.values()
            if r.get("custom_ar") and r["custom_ar"]["deterministic"]
        )
        custom_ar_total_count = sum(
            1 for r in results.values() if r.get("custom_ar") is not None
        )

        deterministic_kernel_deterministic_count = sum(
            1
            for r in results.values()
            if r.get("deterministic_kernel")
            and r["deterministic_kernel"]["deterministic"]
        )
        deterministic_kernel_total_count = sum(
            1 for r in results.values() if r.get("deterministic_kernel") is not None
        )

        print(f"\nDeterminism Summary:")
        print(
            f"  All-Reduce deterministic: {ar_deterministic_count}/{len(results)} batch sizes"
        )
        print(
            f"  RS+All-Gather deterministic: {rs_ag_deterministic_count}/{len(results)} batch sizes"
        )
        if custom_ar_total_count > 0:
            print(
                f"  Custom AR deterministic: {custom_ar_deterministic_count}/{custom_ar_total_count} batch sizes"
            )
        if deterministic_kernel_total_count > 0:
            print(
                f"  Deterministic Kernel deterministic: {deterministic_kernel_deterministic_count}/{deterministic_kernel_total_count} batch sizes"
            )

        print(f"\nLatency Overhead Statistics (RS+AG vs All-Reduce):")
        avg_overhead = statistics.mean(overheads_rs_ag)
        median_overhead = statistics.median(overheads_rs_ag)
        min_overhead = min(overheads_rs_ag)
        max_overhead = max(overheads_rs_ag)
        print(f"  Average: {avg_overhead:.1f}%")
        print(f"  Median:  {median_overhead:.1f}%")
        print(f"  Min:     {min_overhead:.1f}%")
        print(f"  Max:     {max_overhead:.1f}%")

        if custom_ar_total_count > 0:
            overheads_custom = []
            for r in results.values():
                if r.get("custom_ar") is not None:
                    overhead = (
                        (
                            r["custom_ar"]["latency_median"]
                            - r["all_reduce"]["latency_median"]
                        )
                        / r["all_reduce"]["latency_median"]
                    ) * 100
                    overheads_custom.append(overhead)
            print(f"\nLatency Overhead Statistics (Custom AR vs All-Reduce):")
            print(f"  Average: {statistics.mean(overheads_custom):.1f}%")
            print(f"  Median:  {statistics.median(overheads_custom):.1f}%")
            print(f"  Min:     {min(overheads_custom):.1f}%")
            print(f"  Max:     {max(overheads_custom):.1f}%")

        if deterministic_kernel_total_count > 0:
            overheads_kernel = []
            speedups_kernel = []
            for r in results.values():
                if r.get("deterministic_kernel") is not None:
                    overhead = (
                        (
                            r["deterministic_kernel"]["latency_median"]
                            - r["all_reduce"]["latency_median"]
                        )
                        / r["all_reduce"]["latency_median"]
                    ) * 100
                    overheads_kernel.append(overhead)
                    speedup = (
                        (
                            r["rs_ag"]["latency_median"]
                            - r["deterministic_kernel"]["latency_median"]
                        )
                        / r["rs_ag"]["latency_median"]
                    ) * 100
                    speedups_kernel.append(speedup)
            print(
                f"\nLatency Overhead Statistics (Deterministic Kernel vs All-Reduce):"
            )
            print(f"  Average: {statistics.mean(overheads_kernel):.1f}%")
            print(f"  Median:  {statistics.median(overheads_kernel):.1f}%")
            print(f"  Min:     {min(overheads_kernel):.1f}%")
            print(f"  Max:     {max(overheads_kernel):.1f}%")
            print(f"\nSpeedup Statistics (Deterministic Kernel vs RS+AG):")
            print(f"  Average: {statistics.mean(speedups_kernel):.1f}%")
            print(f"  Median:  {statistics.median(speedups_kernel):.1f}%")
            print(f"  Min:     {min(speedups_kernel):.1f}%")
            print(f"  Max:     {max(speedups_kernel):.1f}%")

        # Show variance for non-deterministic cases
        print(f"\nVariance Analysis (non-deterministic cases):")
        for bs in sorted(results.keys()):
            r = results[bs]
            if not r["all_reduce"]["deterministic"]:
                print(
                    f"  Batch {bs}: All-Reduce max variance: {r['all_reduce']['max_variance']:.6f}"
                )
            if not r["rs_ag"]["deterministic"]:
                print(
                    f"  Batch {bs}: RS+All-Gather max variance: {r['rs_ag']['max_variance']:.6f}"
                )
            if r.get("custom_ar") is not None and not r["custom_ar"]["deterministic"]:
                print(
                    f"  Batch {bs}: Custom AR max variance: {r['custom_ar']['max_variance']:.6f}"
                )
            if (
                r.get("deterministic_kernel") is not None
                and not r["deterministic_kernel"]["deterministic"]
            ):
                print(
                    f"  Batch {bs}: Deterministic Kernel max variance: {r['deterministic_kernel']['max_variance']:.6f}"
                )


if __name__ == "__main__":
    main()

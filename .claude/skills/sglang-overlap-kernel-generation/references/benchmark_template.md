# Benchmark Test File Template

Generate a test file that exercises the overlap kernel with four test modes. The structure follows this pattern:

```python
"""
Test for <Op> Overlap Kernel using PyTorch Symmetric Memory.

Usage:
    # Correctness test
    <NVSHMEM_DISABLE_CUDA_VMM=0 if multimem else "">torchrun --nproc_per_node=<world_size> <test_file>.py --case correctness

    # Performance benchmark
    <NVSHMEM_DISABLE_CUDA_VMM=0 if multimem else "">torchrun --nproc_per_node=<world_size> <test_file>.py --case performance

    # Multi-size sweep
    <NVSHMEM_DISABLE_CUDA_VMM=0 if multimem else "">torchrun --nproc_per_node=<world_size> <test_file>.py --case multi_size

    # Stability (hang detection)
    <NVSHMEM_DISABLE_CUDA_VMM=0 if multimem else "">torchrun --nproc_per_node=<world_size> <test_file>.py --case stability

Note: <Add NVSHMEM_DISABLE_CUDA_VMM=0 prefix when using multimem/NVLS communication mechanism>
"""

import argparse
import os
import torch
import torch.distributed as dist


def initialize_distributed():
    """Initialize distributed environment."""
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        device_id=torch.device(f"cuda:{LOCAL_RANK}"),
    )
    assert dist.is_initialized()

    tp_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    dist.barrier(tp_group)

    return RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, tp_group

# Block size for FP8 GEMM kernels: [block_m, block_n, block_k]
BLOCK_SIZE = [64, 128, 128]

# Precision check (cosine-similarity based, from DeepGEMM)
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Cosine-similarity based diff metric. Threshold: < 0.001."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

# ---------------------------------------------------------------------------
# Reference Implementation
# ---------------------------------------------------------------------------

def <op>_reference(input_tensor, ..., pg):
    """
    Reference implementation using standard PyTorch/NCCL ops.
    This is the ground truth for correctness verification.
    """
    ...

# ---------------------------------------------------------------------------
# Performance Measurement
# ---------------------------------------------------------------------------

def perf_func(func, warmup_iters=10, iters=100, *args, **kwargs):
    """Performance measurement: warmup then timed iterations."""
    # Warmup
    for _ in range(warmup_iters):
        output = func(*args, **kwargs)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        output = func(*args, **kwargs)

    torch.cuda.synchronize()
    end = time.perf_counter()

    duration_ms = (end - start) / iters * 1000
    return output, duration_ms

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_correctness(args):
    """
    Verify overlap kernel output matches reference implementation.
    Tests: overlap kernel vs NCCL reference.
    """
    ...
    # 1. Create context
    # 2. Generate random input (seeded with 42 + rank)
    # 3. Run overlap kernel
    # 4. Run reference implementation
    # 5. torch.testing.assert_close(overlap_out, ref_out, rtol=rtol, atol=atol)
    # 6. Print PASSED/FAILED with max_diff and mean_diff on failure

def test_performance(args):
    """
    Benchmark three implementations:
    1. Standalone compute kernel
    2. Standalone comm kernel (NCCL)
    3. Overlap kernel (compute + comm overlap)

    Reports latency (us), bandwidth (GB/s), and speedup.
    """
    ...
    # For each implementation, use perf_func(func, args.warmup, args.iters, ...):
    # 1. Standalone compute: _, compute_time = perf_func(compute_only, args.warmup, args.iters, ...)
    # 2. Standalone comm: _, comm_time = perf_func(comm_only, args.warmup, args.iters, ...)
    # 3. Overlap kernel: _, overlap_time = perf_func(overlap_kernel, args.warmup, args.iters, ...)
    # 4. Print comparison table
    # 5. Calculate speedup = (compute_time + comm_time) / overlap_time

def test_multi_size(args):
    """Correctness across multiple problem shapes and dtypes."""
    ...
    # Sweep over representative (M, N, ...) configurations
    # Each config must satisfy divisibility constraints

def test_stability(args, n_iters=50):
    """Repeated correctness checks to catch non-deterministic hang/failures."""
    ...
    # Run overlap kernel n_iters times with different random inputs
    # Detects intermittent hangs (timeout) and precision drift

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Test <Op> Overlap Kernel")
    parser.add_argument("--case", type=str, default="correctness",
                        choices=["correctness", "performance", "multi_size", "stability"])
    # Shape parameters (with sensible defaults)
    # For GEMM-based kernels, DEFAULT values are: M=1024, N=7168, K=2048
    # These defaults represent realistic LLM inference workloads
    parser.add_argument("--M", type=int, default=1024, help="M dimension")
    parser.add_argument("--N", type=int, default=7168, help="N dimension")
    parser.add_argument("--K", type=int, default=2048, help="K dimension")
    # Kernel tuning parameters (block_size defined as module-level constant BLOCK_SIZE)
    parser.add_argument("--num_comm_sms", type=int, default=8, help="Number of SMs for communication")
    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    # Misc (--profile, --dtype)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    rank, local_rank, world_size, local_world_size, tp_group = initialize_distributed()
    args = parse_args()
    # Validate constraints (divisibility, etc.)
    # Dispatch to test case
    ...

if __name__ == "__main__":
    main()
```

## Performance Test Requirements

The performance test **must** include all four components separately:

1. **Compute-only**: run just the compute kernel in isolation
2. **Comm-only**: run just the collective (e.g., via NCCL) in isolation
3. **Non-overlap**: run compute + comm sequentially (no overlap)
4. **Overlap**: run the fused overlap kernel

This allows calculating **speedup** = `(compute_time + comm_time) / overlap_time` and **overlap efficiency** = `max(compute_time, comm_time) / overlap_time`.

### Compute Kernel Selection for Benchmark

The **compute-only** and **non-overlap** benchmarks **must prefer the user-provided compute kernel implementation** for timing measurement, not a PyTorch reference implementation. This ensures a fair apples-to-apples comparison with the overlap kernel, which also uses the same compute kernel internally.

**Priority order for compute kernel selection:**

1. **User-provided compute kernel** (highest priority): If the user supplies a path to a custom compute kernel (e.g., a Triton kernel function), import and call it directly for the compute-only and non-overlap measurements. This is the same compute path that the overlap kernel uses internally, so the comparison is authoritative.

2. **PyTorch reference implementation** (fallback): Only use PyTorch built-in ops (e.g., `torch.sum`, `torch.matmul`, `.mul_()`) when no user-provided compute kernel is available. The reference implementation is primarily used for **correctness verification** (comparing overlap output against ground truth), not for performance benchmarking.

**Implementation pattern when user provides a compute kernel:**

```python
# Import the user-provided compute kernel from the overlap kernel file
from <overlap_kernel_file> import (
    <user_compute_kernel>,        # e.g., topk_sum_reduce_kernel (Triton kernel launcher)
    <overlap_kernel_entry>,       # e.g., topk_sum_reduce_rs_overlap
    create_<op>_overlap_context,
)

def compute_only(x, out, *args, **kwargs):
    """Compute-only: user-provided compute kernel in isolation."""
    <user_compute_kernel>(x, out, *args, **kwargs)
    return out

def comm_only(x, output, tp_group):
    """Comm-only: NCCL collective in isolation."""
    ...

def non_overlap(x, out, output, *args, tp_group, **kwargs):
    """Non-overlap: user-provided compute kernel + NCCL comm sequentially."""
    compute_only(x, out, *args, **kwargs)
    comm_only(out, output, tp_group)
    return output
```

**Why this matters:** If the overlap kernel uses a Triton compute path but the benchmark measures the compute-only baseline with PyTorch ops, the speedup metric becomes misleading — PyTorch ops may be faster or slower than the custom kernel, distorting the overlap efficiency calculation. Using the same compute kernel in both overlap and non-overlap ensures the speedup metric reflects the genuine benefit of overlapping compute and communication.

## Output Table Format

```
  Implementation           Latency (us)    BW (GB/s)    Speedup
  -----------------------------------------------------------------------
  Compute-only             xxx.xx          xx.xx        -
  Comm-only (NCCL)         xxx.xx          xx.xx        -
  Non-overlap              xxx.xx          xx.xx        -
  Overlap kernel           xxx.xx          xx.xx        x.xxx
  -----------------------------------------------------------------------
  Speedup = (compute_time + comm_time) / overlap_time
```

## Key Conventions

- Use `torch.manual_seed(42 + rank)` for reproducible inputs
- Use CUDA events (`torch.cuda.Event(enable_timing=True)`) for timing
- Always call `torch.cuda.synchronize()` before/after timed sections
- Print results only on rank 0
- Call `ctx.finalize()` in cleanup to release symmetric memory
- Support `--profile` flag for optional torch profiler trace export
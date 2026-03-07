"""
Micro-benchmark for the SGLang Diffusion JIT CUDA RMSNorm kernel.

Compares:
  1. SGLang JIT CUDA kernel (diffusion_rmsnorm)
  2. PyTorch baseline (torch.nn.functional.rms_norm)

Adapted from: https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels

Usage:
    python scripts/bench_diffusion_rmsnorm.py

Requirements:
    pip install triton  # for triton.testing timing utilities
    # SGLang must be installed and CUDA available
"""

import time
from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# Import the JIT CUDA kernel.
# When you implement add-cuda-kernel.md, the file will be at:
#   python/sglang/jit_kernel/diffusion/rmsnorm.py
# ---------------------------------------------------------------------------
try:
    from sglang.jit_kernel.diffusion.rmsnorm import diffusion_rmsnorm

    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    print(
        "WARNING: diffusion.rmsnorm JIT kernel not available. "
        "Run after implementing add-cuda-kernel.md."
    )


def pytorch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference PyTorch implementation of RMSNorm."""
    hidden = x.shape[-1]
    return torch.nn.functional.rms_norm(
        x.float(), (hidden,), weight.float() if weight is not None else None, eps=eps
    ).to(x.dtype)


def benchmark_kernel(
    func,
    args,
    warmup: int = 20,
    iterations: int = 100,
) -> Tuple[float, float]:
    """Benchmark a kernel function. Returns (avg_ms, min_ms)."""
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        func(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times), min(times)


def run_benchmark():
    print("=" * 72)
    print("SGLang Diffusion RMSNorm Micro-Benchmark: JIT CUDA vs PyTorch")
    print("=" * 72)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute Capability: sm_{cap[0]}{cap[1]}")
    print()

    if not JIT_AVAILABLE:
        print("Skipping JIT kernel benchmark (kernel not available).")
        return

    # Determine dtype: T4 (sm_75) has no BF16
    dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
    print(f"Dtype: {dtype}")
    print()

    # Typical DiT hidden sizes for sglang diffusion models:
    #   FLUX.1-dev: hidden=3072
    #   Qwen-Image: hidden=2048
    #   Wan2.2:     hidden=4096
    configs = [
        # (batch_tokens, hidden_size, has_weight)
        (1024, 2048, True),  # Qwen-Image: 1 sample × 1024 tokens
        (4096, 2048, True),  # Qwen-Image: larger batch
        (1024, 3072, True),  # FLUX: 1 sample × 1024 tokens
        (4096, 3072, True),  # FLUX: larger
        (4096, 4096, True),  # Wan2.2
        (4096, 2048, False),  # no-weight (elementwise_affine=False)
        (16384, 3072, True),  # long sequence
    ]

    print(
        f"{'Config':<32} {'JIT(ms)':>10} {'PyTorch(ms)':>12} {'Speedup':>9} {'Weight'}"
    )
    print("-" * 72)

    total_speedup = 0
    n = 0

    for batch_tokens, hidden, has_weight in configs:
        x = torch.randn(batch_tokens, hidden, dtype=dtype, device="cuda")
        weight = torch.ones(hidden, dtype=dtype, device="cuda") if has_weight else None

        jit_avg, _ = benchmark_kernel(
            diffusion_rmsnorm, (x, weight, 1e-6), warmup=20, iterations=100
        )
        pt_avg, _ = benchmark_kernel(
            pytorch_rmsnorm, (x, weight, 1e-6), warmup=20, iterations=100
        )

        speedup = pt_avg / jit_avg
        total_speedup += speedup
        n += 1

        w_str = "yes" if has_weight else "no "
        cfg = f"[{batch_tokens}×{hidden}]"
        print(f"{cfg:<32} {jit_avg:>10.3f} {pt_avg:>12.3f} {speedup:>8.2f}x  {w_str}")

    print("-" * 72)
    print(f"{'Average Speedup':>56} {total_speedup / n:.2f}x")
    print()

    # -----------------------------------------------------------------------
    # Correctness check
    # -----------------------------------------------------------------------
    print("Correctness Check (BF16 tolerance 0.02):")
    x = torch.randn(4096, 3072, dtype=dtype, device="cuda")
    weight = torch.ones(3072, dtype=dtype, device="cuda")

    out_jit = diffusion_rmsnorm(x, weight=weight, eps=1e-6)
    out_ref = pytorch_rmsnorm(x, weight=weight, eps=1e-6)

    max_diff = (out_jit - out_ref).abs().max().item()
    rel_diff = ((out_jit - out_ref).abs() / (out_ref.abs() + 1e-8)).max().item()
    passed = max_diff < 0.02

    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Max relative diff: {rel_diff:.2e}")
    print(f"  Correctness: {'PASS ✓' if passed else 'FAIL ✗'}")
    print()

    # -----------------------------------------------------------------------
    # Memory bandwidth analysis
    # -----------------------------------------------------------------------
    print("Memory Bandwidth Analysis:")
    bt, hid = 4096, 3072
    x = torch.randn(bt, hid, dtype=dtype, device="cuda")
    weight = torch.ones(hid, dtype=dtype, device="cuda")

    bytes_per_elem = dtype.itemsize
    total_bytes = (
        bt * hid + hid + bt * hid
    ) * bytes_per_elem  # read x + read w + write out
    jit_avg, _ = benchmark_kernel(diffusion_rmsnorm, (x, weight, 1e-6))

    bandwidth_gbps = (total_bytes / 1e9) / (jit_avg / 1000)
    theoretical_bw = {
        (9, 0): 3350,  # H100: 3.35 TB/s
        (8, 0): 2000,  # A100 80GB
    }.get(
        cap, 320
    )  # T4: 320 GB/s
    efficiency = bandwidth_gbps / theoretical_bw * 100

    print(f"  Shape: [{bt} × {hid}]  dtype: {dtype}")
    print(f"  Total data: {total_bytes / 1e6:.1f} MB")
    print(f"  Achieved: {bandwidth_gbps:.1f} GB/s")
    print(f"  Theoretical ({torch.cuda.get_device_name(0)}): {theoretical_bw} GB/s")
    print(f"  Bandwidth efficiency: {efficiency:.1f}%")
    print()
    print("Target: ≥ 30% efficiency (H100/A100), ≥ 40% (T4)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
    else:
        run_benchmark()

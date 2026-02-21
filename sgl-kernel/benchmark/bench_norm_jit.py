import itertools
import os

import sgl_kernel
import torch
import triton
import triton.testing

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def torch_rmsnorm(input, weight, eps=1e-6):
    """PyTorch reference implementation of RMSNorm."""
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    output = input * torch.rsqrt(variance + eps)
    output = output * weight
    return output


def torch_fused_add_rmsnorm(input, residual, weight, eps=1e-6):
    """PyTorch reference implementation of fused add + RMSNorm."""
    residual_updated = residual + input
    variance = residual_updated.pow(2).mean(dim=-1, keepdim=True)
    output = residual_updated * torch.rsqrt(variance + eps)
    output = output * weight
    return output, residual_updated


def torch_gemma_rmsnorm(input, weight, eps=1e-6):
    """PyTorch reference implementation of Gemma RMSNorm."""
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    output = input * torch.rsqrt(variance + eps)
    output = output * (weight + 1.0)
    return output


def torch_gemma_fused_add_rmsnorm(input, residual, weight, eps=1e-6):
    """PyTorch reference implementation of Gemma fused add + RMSNorm."""
    residual_updated = residual + input
    variance = residual_updated.pow(2).mean(dim=-1, keepdim=True)
    output = residual_updated * torch.rsqrt(variance + eps)
    output = output * (weight + 1.0)
    return output, residual_updated


def calculate_diff_rmsnorm(batch_size, hidden_size):
    """Compare Torch reference and JIT for rmsnorm correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    torch_output = torch_rmsnorm(input, weight, eps)
    jit_output = sgl_kernel.rmsnorm(input, weight, eps)

    torch.testing.assert_close(torch_output, jit_output, rtol=1e-2, atol=1e-2)


def calculate_diff_fused_add_rmsnorm(batch_size, hidden_size):
    """Compare Torch reference and JIT for fused_add_rmsnorm correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    residual_torch = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    residual_jit = residual_torch.clone()
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    torch_output, _ = torch_fused_add_rmsnorm(
        input.clone(), residual_torch, weight, eps
    )

    # JIT version (in-place)
    input_jit = input.clone()
    sgl_kernel.fused_add_rmsnorm(input_jit, residual_jit, weight, eps)

    torch.testing.assert_close(torch_output, input_jit, rtol=1e-2, atol=1e-2)


def calculate_diff_gemma_rmsnorm(batch_size, hidden_size):
    """Compare Torch reference and JIT for gemma_rmsnorm correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    torch_output = torch_gemma_rmsnorm(input, weight, eps)
    jit_output = sgl_kernel.gemma_rmsnorm(input, weight, eps)

    torch.testing.assert_close(torch_output, jit_output, rtol=1e-2, atol=1e-2)


def calculate_diff_gemma_fused_add_rmsnorm(batch_size, hidden_size):
    """Compare Torch reference and JIT for gemma_fused_add_rmsnorm correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    residual_torch = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    residual_jit = residual_torch.clone()
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    torch_output, _ = torch_gemma_fused_add_rmsnorm(
        input.clone(), residual_torch, weight, eps
    )

    # JIT version (in-place)
    input_jit = input.clone()
    sgl_kernel.gemma_fused_add_rmsnorm(input_jit, residual_jit, weight, eps)

    torch.testing.assert_close(torch_output, input_jit, rtol=1e-2, atol=1e-2)


# Parameter space - simplified for CI
if IS_CI:
    batch_size_range = [32]
    hidden_size_range = [4096]
else:
    batch_size_range = [1, 32, 128, 256]
    hidden_size_range = [2048, 4096, 8192, 11008]

configs = list(itertools.product(batch_size_range, hidden_size_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sgl_jit"],
        line_names=["Torch Reference", "SGL Kernel (JIT)"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="rmsnorm-performance",
        args={},
    )
)
def benchmark_rmsnorm(batch_size, hidden_size, provider):
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    if provider == "torch":
        fn = lambda: torch_rmsnorm(input.clone(), weight, eps)
    elif provider == "sgl_jit":
        fn = lambda: sgl_kernel.rmsnorm(input.clone(), weight, eps)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sgl_jit"],
        line_names=["Torch Reference", "SGL Kernel (JIT)"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark_fused_add_rmsnorm(batch_size, hidden_size, provider):
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    if provider == "torch":
        fn = lambda: torch_fused_add_rmsnorm(
            input.clone(), residual.clone(), weight, eps
        )
    elif provider == "sgl_jit":

        def jit_fn():
            inp = input.clone()
            res = residual.clone()
            sgl_kernel.fused_add_rmsnorm(inp, res, weight, eps)
            return inp

        fn = jit_fn

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sgl_jit"],
        line_names=["Torch Reference", "SGL Kernel (JIT)"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="gemma-rmsnorm-performance",
        args={},
    )
)
def benchmark_gemma_rmsnorm(batch_size, hidden_size, provider):
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    if provider == "torch":
        fn = lambda: torch_gemma_rmsnorm(input.clone(), weight, eps)
    elif provider == "sgl_jit":
        fn = lambda: sgl_kernel.gemma_rmsnorm(input.clone(), weight, eps)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sgl_jit"],
        line_names=["Torch Reference", "SGL Kernel (JIT)"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="gemma-fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark_gemma_fused_add_rmsnorm(batch_size, hidden_size, provider):
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    input = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    eps = 1e-6

    if provider == "torch":
        fn = lambda: torch_gemma_fused_add_rmsnorm(
            input.clone(), residual.clone(), weight, eps
        )
    elif provider == "sgl_jit":

        def jit_fn():
            inp = input.clone()
            res = residual.clone()
            sgl_kernel.gemma_fused_add_rmsnorm(inp, res, weight, eps)
            return inp

        fn = jit_fn

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print("=" * 60)
    print("Running correctness checks...")
    print("=" * 60)

    # Correctness checks - simplified for CI
    if IS_CI:
        test_configs = [configs[0]]
    else:
        test_configs = configs[:3]  # Test first 3 configs

    print("\n1. Testing rmsnorm...")
    for cfg in test_configs:
        batch_size, hidden_size = cfg
        calculate_diff_rmsnorm(batch_size, hidden_size)
        print(f"  ✓ Passed: batch_size={batch_size}, hidden_size={hidden_size}")

    print("\n2. Testing fused_add_rmsnorm...")
    for cfg in test_configs:
        batch_size, hidden_size = cfg
        calculate_diff_fused_add_rmsnorm(batch_size, hidden_size)
        print(f"  ✓ Passed: batch_size={batch_size}, hidden_size={hidden_size}")

    print("\n3. Testing gemma_rmsnorm...")
    for cfg in test_configs:
        batch_size, hidden_size = cfg
        calculate_diff_gemma_rmsnorm(batch_size, hidden_size)
        print(f"  ✓ Passed: batch_size={batch_size}, hidden_size={hidden_size}")

    print("\n4. Testing gemma_fused_add_rmsnorm...")
    for cfg in test_configs:
        batch_size, hidden_size = cfg
        calculate_diff_gemma_fused_add_rmsnorm(batch_size, hidden_size)
        print(f"  ✓ Passed: batch_size={batch_size}, hidden_size={hidden_size}")

    print("\n" + "=" * 60)
    print("All correctness checks passed!")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Starting performance benchmarks...")
    print("=" * 60)

    print("\n1. Benchmarking rmsnorm...")
    benchmark_rmsnorm.run(print_data=True)

    print("\n2. Benchmarking fused_add_rmsnorm...")
    benchmark_fused_add_rmsnorm.run(print_data=True)

    print("\n3. Benchmarking gemma_rmsnorm...")
    benchmark_gemma_rmsnorm.run(print_data=True)

    print("\n4. Benchmarking gemma_fused_add_rmsnorm...")
    benchmark_gemma_fused_add_rmsnorm.run(print_data=True)

    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("=" * 60)

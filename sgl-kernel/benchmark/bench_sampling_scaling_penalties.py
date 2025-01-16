import itertools

import torch
import triton
from sgl_kernel import sampling_scaling_penalties


def sampling_scaling_penalties_naive(logits, scaling_penalties):
    return torch.where(
        logits > 0, logits / scaling_penalties, logits * scaling_penalties
    )


def sampling_scaling_penalties_kernel(logits, scaling_penalties):
    return sampling_scaling_penalties(logits, scaling_penalties)


def test_memory(func, _iter):
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated() / (2**20)
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)


def calculate_diff(batch_size, vocab_size):
    dtype = torch.bfloat16
    device = torch.device("cuda")

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    scaling_penalties = (
        torch.rand(batch_size, vocab_size, device=device, dtype=dtype) + 0.5
    )

    output_naive = sampling_scaling_penalties_naive(
        logits.clone(), scaling_penalties.clone()
    )
    output_kernel = sampling_scaling_penalties_kernel(
        logits.clone(), scaling_penalties.clone()
    )

    print(f"Naive output={output_naive}")
    print(f"Kernel output={output_kernel}")

    if torch.allclose(output_naive, output_kernel, atol=1e-2, rtol=1e-2):
        print("✅ Both implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [2**i for i in range(0, 12)]
vocab_size_range = [2**i for i in range(10, 17)]
configs = list(itertools.product(batch_size_range, vocab_size_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["naive", "kernel"],
        line_names=["PyTorch Naive", "SGL Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="sampling-scaling-penalties-performance",
        args={},
    )
)
def benchmark(batch_size, vocab_size, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    scaling_penalties = (
        torch.rand(batch_size, vocab_size, device=device, dtype=dtype) + 0.5
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sampling_scaling_penalties_naive(
                logits.clone(),
                scaling_penalties.clone(),
            ),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sampling_scaling_penalties_kernel(
                logits.clone(),
                scaling_penalties.clone(),
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["naive", "kernel"],
        line_names=["PyTorch Naive", "SGL Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="GPU memory usage (MB)",
        plot_name="sampling-scaling-penalties-memory",
        args={},
    )
)
def benchmark_memory(batch_size, vocab_size, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")

    print(
        f"Running memory benchmark with batch_size={batch_size}, vocab_size={vocab_size}, provider={provider}"
    )

    def run_kernel():
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
        scaling_penalties = (
            torch.rand(batch_size, vocab_size, device=device, dtype=dtype) + 0.5
        )

        if provider == "naive":
            return sampling_scaling_penalties_naive(logits, scaling_penalties)
        else:
            return sampling_scaling_penalties_kernel(logits, scaling_penalties)

    mem = test_memory(run_kernel, _iter=10)
    return mem


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/sampling_scaling_penalties/",
        help="Path to save sampling_scaling_penalties benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test
    calculate_diff(batch_size=4, vocab_size=4096)

    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)

    # Run memory benchmark
    benchmark_memory.run(print_data=True, save_path=args.save_path)

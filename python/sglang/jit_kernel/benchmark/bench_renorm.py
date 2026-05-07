import itertools

import sgl_kernel
import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import run_benchmark_no_cudagraph
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=5, suite="stage-b-kernel-benchmark-1-gpu-large")


def torch_top_k_renorm_probs(probs, top_k):
    """Vectorized PyTorch implementation of top-k renormalization."""
    batch_size, vocab_size = probs.shape

    # Handle scalar or tensor k
    if isinstance(top_k, int):
        k_val = min(max(top_k, 1), vocab_size)
        # Get top-k indices for all batches at once
        _, topk_indices = torch.topk(probs, k_val, dim=1, largest=True)

        # Create mask: batch_size x vocab_size
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_indices, 1.0)

        # Vectorized renormalization
        masked_probs = probs * mask
        renorm_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-10)
        return renorm_probs
    else:
        # Variable k per batch - need to handle separately
        renorm_probs = torch.zeros_like(probs)
        for i in range(batch_size):
            k_val = min(max(top_k[i].item(), 1), vocab_size)
            _, topk_indices = torch.topk(probs[i], k_val, largest=True)
            mask = torch.zeros_like(probs[i])
            mask[topk_indices] = 1.0
            masked_probs = probs[i] * mask
            renorm_probs[i] = masked_probs / (masked_probs.sum() + 1e-10)
        return renorm_probs


def torch_top_p_renorm_probs(probs, top_p, eps=1e-5):
    """Vectorized PyTorch implementation of top-p renormalization."""
    batch_size, vocab_size = probs.shape

    # Handle scalar or tensor p
    if isinstance(top_p, float):
        p_val = top_p
        # Vectorized implementation for uniform top_p
        # Sort probs in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)

        # Find cutoff: where cumsum exceeds top_p
        cutoff_mask = cumsum_probs <= p_val
        # Keep at least one token (the highest prob)
        cutoff_mask[:, 0] = True

        # Create mask in original order
        mask = torch.zeros_like(probs)
        mask.scatter_(1, sorted_indices, cutoff_mask.float())

        # Vectorized renormalization
        masked_probs = probs * mask
        renorm_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + eps)
        return renorm_probs
    else:
        # Variable p per batch - need to handle separately
        renorm_probs = torch.zeros_like(probs)
        for i in range(batch_size):
            p_val = top_p[i].item()
            sorted_prob, indices = torch.sort(probs[i], descending=False)
            cdf = torch.cumsum(sorted_prob, dim=-1)
            mask = torch.zeros(vocab_size, dtype=torch.float32, device=probs.device)
            mask.scatter_(0, indices, (cdf >= (1 - p_val) - eps).float())
            masked_probs = probs[i] * mask
            renorm_probs[i] = masked_probs / (masked_probs.sum() + eps)
        return renorm_probs


def calculate_diff_top_k_renorm(batch_size, vocab_size, k):
    """Compare Torch reference and SGLang kernel for top-k renorm correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    top_k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.int32)

    torch_output = torch_top_k_renorm_probs(probs, top_k_tensor)
    sglang_output = sgl_kernel.top_k_renorm_prob(probs, top_k_tensor)

    torch.testing.assert_close(torch_output, sglang_output, rtol=1e-3, atol=1e-3)


def calculate_diff_top_p_renorm(batch_size, vocab_size, p):
    """Compare Torch reference and SGLang kernel for top-p renorm correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    top_p_tensor = torch.full((batch_size,), p, device=device, dtype=torch.float32)

    torch_output = torch_top_p_renorm_probs(probs, top_p_tensor)
    sglang_output = sgl_kernel.top_p_renorm_prob(probs, top_p_tensor)

    torch.testing.assert_close(torch_output, sglang_output, rtol=1e-3, atol=1e-3)


# Parameter space - simplified for CI
if is_in_ci():
    batch_size_range = [16]
    vocab_size_range = [111]
    k_range = [10]
    p_range = [0.5]
else:
    batch_size_range = [16, 64, 128]
    vocab_size_range = [111, 32000, 128256]
    k_range = [10, 100, 500]
    p_range = [0.1, 0.5, 0.9]

configs_k = list(itertools.product(batch_size_range, vocab_size_range, k_range))
configs_p = list(itertools.product(batch_size_range, vocab_size_range, p_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "k"],
        x_vals=configs_k,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["Torch Reference", "SGL Kernel"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="us",
        plot_name="top-k-renorm-probs-performance",
        args={},
    )
)
def benchmark_top_k_renorm(batch_size, vocab_size, k, provider):
    # Skip invalid configurations
    if k >= vocab_size:
        return float("nan"), float("nan"), float("nan")

    torch.manual_seed(42)
    device = torch.device("cuda")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.int32)

    if provider == "torch":
        fn = lambda: torch_top_k_renorm_probs(probs.clone(), top_k_tensor)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.top_k_renorm_prob(probs.clone(), top_k_tensor)

    return run_benchmark_no_cudagraph(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "p"],
        x_vals=configs_p,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["Torch Reference", "SGL Kernel"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="top-p-renorm-probs-performance",
        args={},
    )
)
def benchmark_top_p_renorm(batch_size, vocab_size, p, provider):
    torch.manual_seed(42)
    device = torch.device("cuda")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_p_tensor = torch.full((batch_size,), p, device=device, dtype=torch.float32)

    if provider == "torch":
        fn = lambda: torch_top_p_renorm_probs(probs.clone(), top_p_tensor)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.top_p_renorm_prob(probs.clone(), top_p_tensor)

    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    print("=" * 60)
    print("Running correctness checks...")
    print("=" * 60)

    # Correctness checks - simplified for CI
    if is_in_ci():
        test_configs_k = [configs_k[0]] if configs_k else [(16, 111, 10)]
        test_configs_p = [configs_p[0]] if configs_p else [(16, 111, 0.5)]
    else:
        test_configs_k = configs_k[:3]  # Test first 3 configs
        test_configs_p = configs_p[:3]

    print("\n1. Testing top_k_renorm_probs...")
    for cfg in test_configs_k:
        batch_size, vocab_size, k = cfg
        if k < vocab_size:  # Skip invalid configs
            calculate_diff_top_k_renorm(batch_size, vocab_size, k)
            print(
                f"  ✓ Passed: batch_size={batch_size}, vocab_size={vocab_size}, k={k}"
            )

    print("\n2. Testing top_p_renorm_probs...")
    for cfg in test_configs_p:
        calculate_diff_top_p_renorm(*cfg)
        batch_size, vocab_size, p = cfg
        print(f"  ✓ Passed: batch_size={batch_size}, vocab_size={vocab_size}, p={p}")

    print("\n" + "=" * 60)
    print("All correctness checks passed!")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Starting performance benchmarks...")
    print("=" * 60)

    print("\n1. Benchmarking top_k_renorm_probs...")
    benchmark_top_k_renorm.run(print_data=True)

    print("\n2. Benchmarking top_p_renorm_probs...")
    benchmark_top_p_renorm.run(print_data=True)

    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("=" * 60)

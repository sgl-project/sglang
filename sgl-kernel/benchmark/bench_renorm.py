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


def torch_top_k_renorm_probs(probs, top_k):
    """Reference PyTorch implementation of top-k renormalization."""
    batch_size, vocab_size = probs.shape
    renorm_probs = torch.zeros_like(probs)

    for i in range(batch_size):
        k_val = top_k[i].item() if top_k.dim() > 0 else top_k
        k_val = min(k_val, vocab_size)  # Clamp k to vocab_size

        # Get top-k values
        sorted_prob, _ = torch.sort(probs[i], descending=True)
        pivot = sorted_prob[k_val - 1]
        mask = (probs[i] >= pivot).float()

        # Renormalize
        masked_probs = probs[i] * mask
        renorm_probs[i] = masked_probs / masked_probs.sum()

    return renorm_probs


def torch_top_p_renorm_probs(probs, top_p, eps=1e-5):
    """Reference PyTorch implementation of top-p renormalization."""
    batch_size, vocab_size = probs.shape
    renorm_probs = torch.zeros_like(probs)

    for i in range(batch_size):
        p_val = top_p[i].item() if top_p.dim() > 0 else top_p

        # Sort and compute cumulative sum
        sorted_prob, indices = torch.sort(probs[i], descending=False)
        cdf = torch.cumsum(sorted_prob, dim=-1)

        # Create mask for top-p
        mask = torch.zeros(vocab_size, dtype=torch.float32, device=probs.device)
        mask.scatter_(0, indices, (cdf >= (1 - p_val) - eps).float())

        # Renormalize
        masked_probs = probs[i] * mask
        renorm_probs[i] = masked_probs / (masked_probs.sum() + eps)

    return renorm_probs


def torch_top_k_mask_logits(logits, top_k):
    """Reference PyTorch implementation of top-k logits masking."""
    batch_size, vocab_size = logits.shape
    masked_logits = torch.full_like(logits, float('-inf'))

    for i in range(batch_size):
        k_val = top_k[i].item() if top_k.dim() > 0 else top_k
        k_val = min(k_val, vocab_size)  # Clamp k to vocab_size

        # Get top-k values
        sorted_logits, _ = torch.sort(logits[i], descending=True)
        pivot = sorted_logits[k_val - 1]
        mask = (logits[i] >= pivot)

        # Mask logits
        masked_logits[i] = torch.where(mask, logits[i], torch.tensor(float('-inf'), device=logits.device))

    return masked_logits


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


def calculate_diff_top_k_mask(batch_size, vocab_size, k):
    """Compare Torch reference and SGLang kernel for top-k mask correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    logits = torch.randn(batch_size, vocab_size, device=device) * 5
    top_k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.int32)

    torch_output = torch_top_k_mask_logits(logits, top_k_tensor)
    sglang_output = sgl_kernel.top_k_mask_logits(logits, top_k_tensor)

    torch.testing.assert_close(torch_output, sglang_output, rtol=1e-3, atol=1e-3)


# Parameter space - simplified for CI
if IS_CI:
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
        line_names=["Torch Reference", "SGL Kernel (FlashInfer)"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="us",
        plot_name="top-k-renorm-probs-performance",
        args={},
    )
)
def benchmark_top_k_renorm(batch_size, vocab_size, k, provider):
    # Skip invalid configurations
    if k >= vocab_size:
        return float('nan'), float('nan'), float('nan')

    torch.manual_seed(42)
    device = torch.device("cuda")

    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.int32)

    if provider == "torch":
        fn = lambda: torch_top_k_renorm_probs(probs.clone(), top_k_tensor)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.top_k_renorm_prob(probs.clone(), top_k_tensor)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "p"],
        x_vals=configs_p,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["Torch Reference", "SGL Kernel (FlashInfer)"],
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

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "k"],
        x_vals=configs_k,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["Torch Reference", "SGL Kernel (FlashInfer)"],
        styles=[("red", "-"), ("orange", "-")],
        ylabel="us",
        plot_name="top-k-mask-logits-performance",
        args={},
    )
)
def benchmark_top_k_mask(batch_size, vocab_size, k, provider):
    # Skip invalid configurations
    if k >= vocab_size:
        return float('nan'), float('nan'), float('nan')

    torch.manual_seed(42)
    device = torch.device("cuda")

    logits = torch.randn(batch_size, vocab_size, device=device) * 5
    top_k_tensor = torch.full((batch_size,), k, device=device, dtype=torch.int32)

    if provider == "torch":
        fn = lambda: torch_top_k_mask_logits(logits.clone(), top_k_tensor)
    elif provider == "sglang":
        fn = lambda: sgl_kernel.top_k_mask_logits(logits.clone(), top_k_tensor)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print("=" * 60)
    print("Running correctness checks...")
    print("=" * 60)

    # Correctness checks - simplified for CI
    if IS_CI:
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
            print(f"  ✓ Passed: batch_size={batch_size}, vocab_size={vocab_size}, k={k}")

    print("\n2. Testing top_p_renorm_probs...")
    for cfg in test_configs_p:
        calculate_diff_top_p_renorm(*cfg)
        batch_size, vocab_size, p = cfg
        print(f"  ✓ Passed: batch_size={batch_size}, vocab_size={vocab_size}, p={p}")

    print("\n3. Testing top_k_mask_logits...")
    for cfg in test_configs_k:
        batch_size, vocab_size, k = cfg
        if k < vocab_size:  # Skip invalid configs
            calculate_diff_top_k_mask(batch_size, vocab_size, k)
            print(f"  ✓ Passed: batch_size={batch_size}, vocab_size={vocab_size}, k={k}")

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

    print("\n3. Benchmarking top_k_mask_logits...")
    benchmark_top_k_mask.run(print_data=True)

    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("=" * 60)

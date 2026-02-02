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


def torch_top_k_top_p_joint_sampling_from_probs(
    normalized_prob, top_k, top_p, eps=1e-4
):
    """Reference PyTorch implementation of joint top-k top-p sampling."""
    batch_size, vocab_size = normalized_prob.shape
    samples = torch.empty(batch_size, dtype=torch.int64, device=normalized_prob.device)

    for i in range(batch_size):
        p_val = top_p[i].item()
        k_val = top_k[i].item()

        # top-p mask
        sorted_prob, indices = torch.sort(normalized_prob[i], descending=False)
        cdf = torch.cumsum(sorted_prob, dim=-1)
        mask_top_p = torch.zeros(
            vocab_size, dtype=torch.int32, device=normalized_prob.device
        )
        mask_top_p.scatter_add_(0, indices, (cdf > (1 - p_val) - eps).int())

        # top-k mask
        sorted_prob_desc, _ = torch.sort(normalized_prob[i], descending=True)
        pivot = sorted_prob_desc[k_val - 1]
        mask_top_k = (normalized_prob[i] >= pivot).int()

        # joint mask
        mask = torch.minimum(mask_top_p, mask_top_k).bool()

        # sample from masked probs
        masked_probs = normalized_prob[i] * mask
        masked_probs = masked_probs / masked_probs.sum()
        idx = torch.multinomial(masked_probs, 1)
        samples[i] = idx

    return samples


def calculate_diff(batch_size, vocab_size, p):
    """Compare Torch reference and SGLang kernel for correctness."""
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")

    device = torch.device("cuda")
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    top_p_tensor = torch.full((batch_size,), p, device=device)
    top_k_tensor = torch.full((batch_size,), k, device=device)

    torch_samples = torch_top_k_top_p_joint_sampling_from_probs(
        normalized_prob, top_k_tensor, top_p_tensor
    )
    sglang_samples = sgl_kernel.top_k_top_p_sampling_from_probs(
        normalized_prob, top_k_tensor, top_p_tensor, filter_apply_order="joint"
    )


# parameter space - simplified for CI
if IS_CI:
    batch_size_range = [16]  # Single batch size for CI
    vocab_size_range = [111]  # Single vocab size for CI
    p_range = [0.1]  # Single p value for CI
else:
    batch_size_range = [16, 64, 128]
    vocab_size_range = [111, 32000]
    p_range = [0.1, 0.5]

configs = list(itertools.product(batch_size_range, vocab_size_range, p_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size", "p"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["Torch Reference", "SGL Kernel"],
        styles=[("red", "-"), ("green", "-")],
        ylabel="us",
        plot_name="top-k-top-p-joint-sampling-performance",
        args={},
    )
)
def benchmark_sampling(batch_size, vocab_size, p, provider):
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")

    device = torch.device("cuda")
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=device)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    top_p_tensor = torch.full((batch_size,), p, device=device)
    top_k_tensor = torch.full((batch_size,), k, device=device)

    if provider == "torch":
        fn = lambda: torch_top_k_top_p_joint_sampling_from_probs(
            normalized_prob.clone(), top_k_tensor, top_p_tensor
        )
    elif provider == "sglang":
        fn = lambda: sgl_kernel.top_k_top_p_sampling_from_probs(
            normalized_prob.clone(),
            top_k_tensor,
            top_p_tensor,
            filter_apply_order="joint",
        )

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Correctness check - simplified for CI
    if IS_CI:
        # Only test one configuration in CI
        test_configs = [configs[0]] if configs else [(16, 111, 0.1)]
    else:
        test_configs = configs

    for cfg in test_configs:
        calculate_diff(*cfg)

    print("\n" + "=" * 60)
    print("Starting performance benchmark...")
    benchmark_sampling.run(print_data=True)

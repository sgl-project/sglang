import itertools
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.jit_kernel.hadamard import hadamard_transform

# Optional: scipy reference
try:
    from scipy.linalg import hadamard

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


# CI environment uses simplified parameters
if IS_CI:
    batch_sizes = [16]
    dim_range = [1024]
else:
    batch_sizes = [1, 16, 64, 256]
    dim_range = [64, 256, 1024, 4096, 8192, 16384, 32768]

configs = list(itertools.product(batch_sizes, dim_range))

available_providers = ["jit_kernel"]
available_names = ["JIT Kernel"]
available_styles = [("red", "-")]

if SCIPY_AVAILABLE:
    available_providers.insert(0, "naive")
    available_names.insert(0, "Naive (scipy)")
    available_styles.insert(0, ("blue", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "dim"],
        x_vals=[list(c) for c in configs],
        line_arg="provider",
        line_vals=available_providers,
        line_names=available_names,
        styles=available_styles,
        ylabel="us",
        plot_name="hadamard-transform-performance",
        args={},
    )
)
def benchmark(batch_size, dim, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    scale = 1.0 / math.sqrt(dim)

    x = torch.randn(batch_size, dim, device=device, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "naive":
        # Precompute Hadamard matrix on GPU to avoid CPU-GPU transfer during
        # CUDA graph capture.
        log_dim = math.ceil(math.log2(dim)) if dim > 0 else 0
        dim_padded = 2**log_dim if dim > 0 else 1
        H = torch.tensor(
            hadamard(dim_padded, dtype=float), dtype=dtype, device=device
        )

        def naive_fn():
            xc = x.clone()
            flat = xc.reshape(-1, dim)
            if dim != dim_padded:
                flat = F.pad(flat, (0, dim_padded - dim))
            out = F.linear(flat, H) * scale
            return out[..., :dim].reshape(xc.shape)

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            naive_fn,
            quantiles=quantiles,
        )
    elif provider == "jit_kernel":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: hadamard_transform(x.clone(), scale=scale),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print("=" * 80)
    print("Benchmarking Fast Hadamard Transform (JIT Kernel)")
    print("=" * 80)
    benchmark.run(print_data=True)

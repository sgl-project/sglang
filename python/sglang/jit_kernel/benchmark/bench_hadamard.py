import itertools
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.hadamard import hadamard_transform

IS_CI = is_in_ci()

# AOT kernel: might not be available in all environments.
# This is used for performance baseline comparison.
try:
    from sgl_kernel import hadamard_transform as hadamard_transform_aot

    AOT_AVAILABLE = True
except Exception:
    AOT_AVAILABLE = False

# Naive reference implementation using scipy hadamard matrix.
try:
    from scipy.linalg import hadamard

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

DTYPE = torch.bfloat16
DEVICE = "cuda"

if IS_CI:
    batch_sizes = [16]
    dim_range = [1024]
else:
    batch_sizes = [1, 16, 64, 256]
    dim_range = [64, 256, 1024, 4096, 8192, 16384, 32768]


# Naive reference implementation using precomputed scipy hadamard matrix.
def torch_hadamard_transform(x, scale, H, dim, dim_padded):
    flat = x.reshape(-1, dim)
    if dim != dim_padded:
        flat = F.pad(flat, (0, dim_padded - dim))
    out = F.linear(flat, H) * scale
    return out[..., :dim].reshape(x.shape)


available_providers = ["jit_kernel"]
available_names = ["JIT Kernel"]
available_styles = [("red", "-")]

if AOT_AVAILABLE:
    available_providers.insert(0, "aot_kernel")
    available_names.insert(0, "AOT Kernel")
    available_styles.insert(0, ("green", "-"))

if SCIPY_AVAILABLE:
    available_providers.append("naive")
    available_names.append("Naive (scipy)")
    available_styles.append(("blue", "-"))

configs = list(itertools.product(batch_sizes, dim_range))


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
def benchmark(batch_size: int, dim: int, provider: str) -> Tuple[float, float, float]:
    scale = 1.0 / math.sqrt(dim)
    x = torch.randn(batch_size, dim, device=DEVICE, dtype=DTYPE)

    FN_MAP = {
        "jit_kernel": lambda: hadamard_transform(x.clone(), scale=scale),
    }
    if AOT_AVAILABLE:
        FN_MAP["aot_kernel"] = lambda: hadamard_transform_aot(
            x.clone(), scale=scale
        )
    if SCIPY_AVAILABLE:
        # Precompute Hadamard matrix on GPU to avoid CPU-GPU transfer
        # during CUDA graph capture.
        log_dim = math.ceil(math.log2(dim)) if dim > 0 else 0
        dim_padded = 2**log_dim if dim > 0 else 1
        H = torch.tensor(
            hadamard(dim_padded, dtype=float), dtype=DTYPE, device=DEVICE
        )
        FN_MAP["naive"] = lambda: torch_hadamard_transform(
            x.clone(), scale, H, dim, dim_padded
        )

    fn = FN_MAP[provider]
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print("=" * 80)
    print("Benchmarking Fast Hadamard Transform")
    print("=" * 80)
    benchmark.run(print_data=True)

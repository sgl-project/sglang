import math

import torch
import torch.nn.functional as F

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE
from sglang.jit_kernel.hadamard import hadamard_transform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

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


# Naive reference implementation using precomputed scipy hadamard matrix.
def torch_hadamard_transform(x, scale, H, dim, dim_padded):
    flat = x.reshape(-1, dim)
    if dim != dim_padded:
        flat = F.pad(flat, (0, dim_padded - dim))
    out = F.linear(flat, H) * scale
    return out[..., :dim].reshape(x.shape)


LINE_VALS = ["jit_kernel"]
if AOT_AVAILABLE:
    LINE_VALS.insert(0, "aot_kernel")
if SCIPY_AVAILABLE:
    LINE_VALS.append("naive")


@marker.parametrize("batch_size", [1, 16, 64, 256], [16])
@marker.parametrize("dim", [64, 256, 1024, 4096, 8192, 16384, 32768], [1024])
@marker.benchmark("provider", LINE_VALS)
def benchmark(batch_size: int, dim: int, provider: str):
    if provider == "aot_kernel" and not AOT_AVAILABLE:
        marker.skip("sgl_kernel AOT not available")
    if provider == "naive" and not SCIPY_AVAILABLE:
        marker.skip("scipy not available")

    scale = 1.0 / math.sqrt(dim)
    x = torch.randn(batch_size, dim, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

    if provider == "jit_kernel":
        return marker.do_bench(
            lambda x: hadamard_transform(x, scale=scale),
            input_args=(x,),
            graph_clone_args=(0,),
        )
    if provider == "aot_kernel":
        return marker.do_bench(
            lambda x: hadamard_transform_aot(x, scale=scale),
            input_args=(x,),
            graph_clone_args=(0,),
        )

    # naive (scipy): precompute Hadamard matrix on GPU to avoid CPU-GPU
    # transfer during CUDA graph capture.
    log_dim = math.ceil(math.log2(dim)) if dim > 0 else 0
    dim_padded = 2**log_dim if dim > 0 else 1
    H = torch.tensor(
        hadamard(dim_padded, dtype=float),
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    return marker.do_bench(
        lambda x: torch_hadamard_transform(x, scale, H, dim, dim_padded),
        input_args=(x,),
        graph_clone_args=(0,),
        memory_args=(x,),
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Benchmarking Fast Hadamard Transform")
    print("=" * 80)
    benchmark.run()

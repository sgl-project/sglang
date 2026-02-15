"""Benchmark for HiCache JIT kernel performance.

This benchmark tests the performance of KV cache transfer operations
between GPU and CPU (host pinned memory), comparing:
- SGL AOT Kernel: Pre-compiled transfer_kv kernels from sgl_kernel
- SGL JIT Kernel: JIT-compiled hicache kernels
- PyTorch Indexing: Plain PyTorch index copy
- PyTorch 2 Stream: PyTorch implementation using 2 CUDA streams

Tests cover:
- One Layer: CPU->GPU
- All Layer: GPU->CPU

Note: Uses do_bench instead of do_bench_cudagraph since CUDA graph
capture doesn't support CPU-GPU memory transfers.
"""

import itertools
from typing import Tuple

import torch
import triton
import triton.testing
from sgl_kernel import transfer_kv_all_layer, transfer_kv_per_layer

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DTYPE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
    transfer_hicache_all_layer,
    transfer_hicache_one_layer,
)


def sglang_aot_transfer_one(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_size: int,
) -> None:
    """SGL AOT Kernel for single layer transfer."""
    transfer_kv_per_layer(
        k_cache_src,
        k_cache_dst,
        v_cache_src,
        v_cache_dst,
        indices_src,
        indices_dst,
        item_size,
    )


def sglang_jit_transfer_one(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    element_dim: int,
) -> None:
    """SGL JIT Kernel for single layer transfer."""
    transfer_hicache_one_layer(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
        element_dim=element_dim,
    )


def sglang_aot_transfer_all(
    k_ptrs_dst: torch.Tensor,
    v_ptrs_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptrs_src: torch.Tensor,
    v_ptrs_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_size: int,
    num_layers: int,
) -> None:
    """SGL AOT Kernel for all layer transfer."""
    transfer_kv_all_layer(
        k_ptrs_src,
        k_ptrs_dst,
        v_ptrs_src,
        v_ptrs_dst,
        indices_src,
        indices_dst,
        item_size,
        num_layers,
    )


def sglang_jit_transfer_all(
    k_ptrs_dst: torch.Tensor,
    v_ptrs_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptrs_src: torch.Tensor,
    v_ptrs_src: torch.Tensor,
    indices_src: torch.Tensor,
    stride_bytes: int,
    element_size: int,
) -> None:
    """SGL JIT Kernel for all layer transfer."""
    transfer_hicache_all_layer(
        k_ptrs_dst,
        v_ptrs_dst,
        indices_dst,
        k_ptrs_src,
        v_ptrs_src,
        indices_src,
        kv_cache_src_stride_bytes=stride_bytes,
        kv_cache_dst_stride_bytes=stride_bytes,
        element_size=element_size,
    )


def pytorch_transfer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst_on_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src_on_src: torch.Tensor,
) -> None:
    """PyTorch indexing baseline."""
    dst_device = k_cache_dst.device
    k_cache_dst[indices_dst_on_dst] = k_cache_src[indices_src_on_src].to(dst_device)
    v_cache_dst[indices_dst_on_dst] = v_cache_src[indices_src_on_src].to(dst_device)


alt_stream = torch.cuda.Stream()


def torch_streams_transfer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst_on_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src_on_src: torch.Tensor,
) -> None:
    """PyTorch 2 Stream baseline."""
    dst_device = k_cache_dst.device
    current_stream = torch.cuda.current_stream()
    alt_stream.wait_stream(current_stream)
    k_cache_dst[indices_dst_on_dst] = k_cache_src[indices_src_on_src].to(dst_device)
    with torch.cuda.stream(alt_stream):
        v_cache_dst[indices_dst_on_dst] = v_cache_src[indices_src_on_src].to(dst_device)
    current_stream.wait_stream(alt_stream)


# Benchmark configuration
GPU_CACHE_SIZE = 32 * 1024  # 32K tokens on GPU
HOST_CACHE_SIZE = 128 * 1024  # 128K tokens on CPU
NUM_LAYERS = 8

BS_RANGE = get_benchmark_range(
    full_range=[2**n for n in range(0, 15)],
    ci_range=[16],
)
ELEMENT_SIZE_RANGE = get_benchmark_range(
    full_range=[64, 128, 256, 512, 1024],
    ci_range=[1024],
)

LINE_VALS = ["aot", "jit", "pytorch", "torch_streams"]
LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel", "PyTorch", "PyTorch 2 Stream"]
STYLES = [("orange", "-"), ("blue", "--"), ("red", ":"), ("green", "-.")]

CONFIGS = list(itertools.product(ELEMENT_SIZE_RANGE, BS_RANGE))


# =============================================================================
# One Layer Benchmarks
# =============================================================================


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["element_size", "batch_size"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="hicache-one-layer-h2d",
        args={},
    )
)
def benchmark_one_layer_h2d(
    element_size: int, batch_size: int, provider: str
) -> Tuple[float, float, float]:
    """One Layer: Host (CPU) -> Device (GPU)."""
    k_cache_src = torch.randn(
        (HOST_CACHE_SIZE, element_size),
        dtype=DEFAULT_DTYPE,
        device="cpu",
        pin_memory=True,
    )
    v_cache_src = torch.randn(
        (HOST_CACHE_SIZE, element_size),
        dtype=DEFAULT_DTYPE,
        device="cpu",
        pin_memory=True,
    )
    k_cache_dst = torch.randn(
        (GPU_CACHE_SIZE, element_size), dtype=DEFAULT_DTYPE, device="cuda"
    )
    v_cache_dst = torch.randn(
        (GPU_CACHE_SIZE, element_size), dtype=DEFAULT_DTYPE, device="cuda"
    )

    indices_src_gpu = torch.randperm(HOST_CACHE_SIZE, device="cuda")[:batch_size]
    indices_dst_gpu = torch.randperm(GPU_CACHE_SIZE, device="cuda")[:batch_size]
    indices_src_cpu = indices_src_gpu.cpu()
    torch.cuda.synchronize()

    element_bytes = element_size * k_cache_src.element_size()

    FN_MAP = {
        "aot": lambda: sglang_aot_transfer_one(
            k_cache_dst,
            v_cache_dst,
            indices_dst_gpu,
            k_cache_src,
            v_cache_src,
            indices_src_gpu,
            element_bytes,
        ),
        "jit": lambda: sglang_jit_transfer_one(
            k_cache_dst,
            v_cache_dst,
            indices_dst_gpu,
            k_cache_src,
            v_cache_src,
            indices_src_gpu,
            element_size,
        ),
        "pytorch": lambda: pytorch_transfer(
            k_cache_dst,
            v_cache_dst,
            indices_dst_gpu,
            k_cache_src,
            v_cache_src,
            indices_src_cpu,
        ),
        "torch_streams": lambda: torch_streams_transfer(
            k_cache_dst,
            v_cache_dst,
            indices_dst_gpu,
            k_cache_src,
            v_cache_src,
            indices_src_cpu,
        ),
    }

    if provider == "jit" and not can_use_hicache_jit_kernel(element_size=element_bytes):
        return (float("nan"), float("nan"), float("nan"))

    ms, min_ms, max_ms = triton.testing.do_bench(
        FN_MAP[provider], quantiles=DEFAULT_QUANTILES
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


# =============================================================================
# All Layer Benchmarks
# =============================================================================


def _create_ptr_tensor(tensors, device="cuda"):
    """Create a tensor of data pointers."""
    return torch.tensor(
        [t.data_ptr() for t in tensors],
        dtype=torch.uint64,
        device=device,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["element_size", "batch_size"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="hicache-all-layer-d2h",
        args={},
    )
)
def benchmark_all_layer_d2h(
    element_size: int, batch_size: int, provider: str
) -> Tuple[float, float, float]:
    """All Layer: Device (GPU) -> Host (CPU)."""
    k_caches_src = torch.randn(
        (NUM_LAYERS, GPU_CACHE_SIZE, element_size), dtype=DEFAULT_DTYPE, device="cuda"
    )
    v_caches_src = torch.randn(
        (NUM_LAYERS, GPU_CACHE_SIZE, element_size), dtype=DEFAULT_DTYPE, device="cuda"
    )
    k_caches_dst = torch.randn(
        (NUM_LAYERS, HOST_CACHE_SIZE, element_size),
        dtype=DEFAULT_DTYPE,
        device="cpu",
        pin_memory=True,
    )
    v_caches_dst = torch.randn(
        (NUM_LAYERS, HOST_CACHE_SIZE, element_size),
        dtype=DEFAULT_DTYPE,
        device="cpu",
        pin_memory=True,
    )

    indices_src_gpu = torch.randperm(GPU_CACHE_SIZE, device="cuda")[:batch_size]
    indices_dst_gpu = torch.randperm(HOST_CACHE_SIZE, device="cuda")[:batch_size]
    indices_dst_cpu = indices_dst_gpu.cpu()
    torch.cuda.synchronize()

    element_bytes = element_size * k_caches_src.element_size()

    k_ptrs_src = _create_ptr_tensor([k_caches_src[i] for i in range(NUM_LAYERS)])
    v_ptrs_src = _create_ptr_tensor([v_caches_src[i] for i in range(NUM_LAYERS)])
    k_ptrs_dst = _create_ptr_tensor([k_caches_dst[i] for i in range(NUM_LAYERS)])
    v_ptrs_dst = _create_ptr_tensor([v_caches_dst[i] for i in range(NUM_LAYERS)])

    FN_MAP = {
        "aot": lambda: sglang_aot_transfer_all(
            k_ptrs_dst,
            v_ptrs_dst,
            indices_dst_gpu,
            k_ptrs_src,
            v_ptrs_src,
            indices_src_gpu,
            element_bytes,
            NUM_LAYERS,
        ),
        "jit": lambda: sglang_jit_transfer_all(
            k_ptrs_dst,
            v_ptrs_dst,
            indices_dst_gpu,
            k_ptrs_src,
            v_ptrs_src,
            indices_src_gpu,
            element_bytes,
            element_bytes,
        ),
        "pytorch": lambda: [
            pytorch_transfer(
                k_caches_dst[i],
                v_caches_dst[i],
                indices_dst_cpu,
                k_caches_src[i],
                v_caches_src[i],
                indices_src_gpu,
            )
            for i in range(NUM_LAYERS)
        ],
        "torch_streams": lambda: [
            torch_streams_transfer(
                k_caches_dst[i],
                v_caches_dst[i],
                indices_dst_cpu,
                k_caches_src[i],
                v_caches_src[i],
                indices_src_gpu,
            )
            for i in range(NUM_LAYERS)
        ],
    }

    if provider == "jit" and not can_use_hicache_jit_kernel(element_size=element_bytes):
        return (float("nan"), float("nan"), float("nan"))

    ms, min_ms, max_ms = triton.testing.do_bench(
        FN_MAP[provider], quantiles=DEFAULT_QUANTILES
    )
    return (
        1000 * ms / NUM_LAYERS,
        1000 * max_ms / NUM_LAYERS,
        1000 * min_ms / NUM_LAYERS,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("One Layer: Host -> Device (CPU -> GPU)")
    print("=" * 60)
    benchmark_one_layer_h2d.run(print_data=True)

    print("\n" + "=" * 60)
    print("All Layer: Device -> Host (GPU -> CPU) [per-layer avg]")
    print("=" * 60)
    benchmark_all_layer_d2h.run(print_data=True)

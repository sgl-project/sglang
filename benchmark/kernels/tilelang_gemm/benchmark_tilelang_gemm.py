#!/usr/bin/env python3
"""
Benchmark TileLang GEMM vs Baseline (DeepGEMM on Hopper, Triton on Ada)

This script compares the performance of TileLang FP8 blockwise GEMM against
architecture-specific baselines:
- Hopper (sm90): DeepGEMM as baseline
- Ada (sm89): Triton as baseline

Usage:
    # Benchmark specific (N, K)
    python benchmark_tilelang_gemm.py --N 4096 --K 8192

    # Benchmark all available configs
    python benchmark_tilelang_gemm.py --all

    # Output to CSV
    python benchmark_tilelang_gemm.py --N 4096 --K 8192 --output results.csv
"""

import argparse
import csv
import logging
from typing import Dict, List, Optional, Tuple

import torch
import triton
from tqdm import tqdm

from sglang.srt.layers import tilelang_gemm_wrapper
from sglang.srt.layers.tilelang_gemm_wrapper.core.config_loader import DEFAULT_M_VALUES


# Detect GPU architecture
def get_gpu_arch() -> str:
    """Detect GPU architecture and return 'hopper', 'ada', or 'unknown'."""
    if not torch.cuda.is_available():
        return "unknown"

    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor

    if sm_version >= 90:
        return "hopper"  # sm90+
    elif sm_version == 89:
        return "ada"  # sm89
    else:
        return "unknown"


GPU_ARCH = get_gpu_arch()

# Import baseline based on architecture
if GPU_ARCH == "hopper":
    from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

    if ENABLE_JIT_DEEPGEMM:
        try:
            from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

            from sglang.srt.layers import deep_gemm_wrapper

            BASELINE_AVAILABLE = True
            BASELINE_NAME = "DeepGEMM"
        except ImportError:
            BASELINE_AVAILABLE = False
            BASELINE_NAME = "DeepGEMM (unavailable)"
    else:
        BASELINE_AVAILABLE = False
        BASELINE_NAME = "DeepGEMM (disabled)"
elif GPU_ARCH == "ada":
    try:
        from sglang.srt.layers.quantization.fp8_kernel import (
            w8a8_block_fp8_matmul_triton,
        )

        BASELINE_AVAILABLE = True
        BASELINE_NAME = "Triton"
    except ImportError:
        BASELINE_AVAILABLE = False
        BASELINE_NAME = "Triton (unavailable)"
else:
    BASELINE_AVAILABLE = False
    BASELINE_NAME = f"Unknown arch ({GPU_ARCH})"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    """Calculate TFLOPS."""
    return 2.0 * M * N * K / latency_ms / 1e9


def prepare_data(M: int, N: int, K: int):
    """Prepare FP8 test data using sglang's quantization functions."""
    from sglang.srt.layers.tilelang_gemm_wrapper.core.quant_utils import (
        prepare_gemm_inputs,
    )

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    A_fp8, B_fp8, A_scale, B_scale = prepare_gemm_inputs(A, B)

    # Prepare DeepGEMM scale if on Hopper
    if GPU_ARCH == "hopper" and BASELINE_AVAILABLE:
        A_scale_deepgemm = get_mn_major_tma_aligned_tensor(A_scale.clone())
    else:
        A_scale_deepgemm = None

    return A_fp8, B_fp8, A_scale, B_scale, A_scale_deepgemm


def benchmark_tilelang(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    M: int,
    N: int,
    rep: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark TileLang kernel using entrypoint API."""

    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        tilelang_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, rep=rep, quantiles=quantiles
    )
    return ms, min_ms, max_ms


def benchmark_deepgemm(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    M: int,
    N: int,
    rep: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark DeepGEMM kernel (Hopper baseline)."""

    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    def fn():
        deep_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, rep=rep, quantiles=quantiles
    )
    return ms, min_ms, max_ms


def benchmark_triton(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
    rep: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark Triton kernel (Ada baseline)."""

    block_size = [128, 128]

    def fn():
        return w8a8_block_fp8_matmul_triton(
            A_fp8, B_fp8, A_scale, B_scale, block_size, output_dtype=torch.bfloat16
        )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, rep=rep, quantiles=quantiles
    )
    return ms, min_ms, max_ms


def run_benchmark(
    N: int,
    K: int,
    m_values: List[int],
    rep: int = 100,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Run benchmark comparing TileLang vs architecture-specific baseline."""

    # Check TileLang availability
    if not tilelang_gemm_wrapper.is_available():
        logger.error("TileLang GEMM is not available")
        return []

    # Check if config exists for this (N, K)
    available_configs = tilelang_gemm_wrapper.list_available_configs()
    if (N, K) not in available_configs:
        logger.error(f"Config not found for N={N}, K={K}")
        logger.error(f"Run tuning first: python tune_tilelang_gemm.py --N {N} --K {K}")
        return []

    results = []

    for M in tqdm(m_values, desc="Benchmarking TileLang"):
        try:
            # Prepare data
            A_fp8, B_fp8, A_scale, B_scale, A_scale_deepgemm = prepare_data(M, N, K)

            # Benchmark TileLang
            try:
                tl_ms, _, _ = benchmark_tilelang(
                    A_fp8, B_fp8, A_scale, B_scale, M, N, rep
                )
                tl_tflops = tflops(M, N, K, tl_ms)
                info = tilelang_gemm_wrapper.get_kernel_info(M, N, K)
                kernel_type = info["kernel_type"] if info else "unknown"
            except Exception as e:
                logger.warning(f"M={M}: TileLang failed: {e}")
                continue

            # Benchmark baseline based on architecture
            if BASELINE_AVAILABLE:
                try:
                    if GPU_ARCH == "hopper":
                        bl_ms, _, _ = benchmark_deepgemm(
                            A_fp8, B_fp8, A_scale_deepgemm, B_scale, M, N, rep
                        )
                    elif GPU_ARCH == "ada":
                        bl_ms, _, _ = benchmark_triton(
                            A_fp8, B_fp8, A_scale, B_scale, M, N, K, rep
                        )
                    else:
                        bl_ms = float("nan")
                    bl_tflops = tflops(M, N, K, bl_ms)
                    speedup = bl_ms / tl_ms
                except Exception as e:
                    logger.warning(f"M={M}: {BASELINE_NAME} failed: {e}")
                    bl_ms = bl_tflops = speedup = float("nan")
            else:
                bl_ms = bl_tflops = speedup = float("nan")

            results.append(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "tl_ms": tl_ms,
                    "bl_ms": bl_ms,
                    "tl_tflops": tl_tflops,
                    "bl_tflops": bl_tflops,
                    "speedup": speedup,
                    "kernel_type": kernel_type,
                }
            )

        except Exception as e:
            logger.warning(f"M={M}: Error: {e}")

    # Print all results at the end (to avoid interleaving with compilation logs)
    print(f"\n{'='*100}")
    print(f"Benchmark: TileLang vs {BASELINE_NAME}")
    print(f"GPU Architecture: {GPU_ARCH.upper()}")
    print(f"N={N}, K={K}")
    print(f"{BASELINE_NAME} available: {BASELINE_AVAILABLE}")
    print(f"{'='*100}\n")

    header = (
        f"{'M':>6} | {'TileLang (ms)':>14} | {BASELINE_NAME + ' (ms)':>14} | "
        f"{'TL TFLOPS':>10} | {'BL TFLOPS':>10} | "
        f"{'Speedup':>8} | {'Kernel Type':>15}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['M']:>6} | {r['tl_ms']:>14.4f} | {r['bl_ms']:>14.4f} | "
            f"{r['tl_tflops']:>10.2f} | {r['bl_tflops']:>10.2f} | "
            f"{r['speedup']:>7.2f}x | {r['kernel_type']:>15}"
        )

    print("-" * len(header))

    # Save to file
    if output_file and results:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TileLang GEMM vs Baseline (DeepGEMM on Hopper, Triton on Ada)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--N", type=int, help="N dimension")
    parser.add_argument("--K", type=int, help="K dimension")
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=DEFAULT_M_VALUES,
        help=f"M values to benchmark (default: {DEFAULT_M_VALUES})",
    )
    parser.add_argument(
        "--rep", type=int, default=100, help="Benchmark repetitions (default: 100)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output CSV file"
    )

    args = parser.parse_args()

    # Print architecture info
    print(f"Detected GPU Architecture: {GPU_ARCH.upper()}")
    print(f"Baseline: {BASELINE_NAME}")
    print()

    run_benchmark(args.N, args.K, args.m_values, args.rep, args.output)


if __name__ == "__main__":
    main()

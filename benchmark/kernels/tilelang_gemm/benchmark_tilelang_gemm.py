#!/usr/bin/env python3
"""
Benchmark TileLang GEMM vs DeepGEMM vs SGL-Kernel

This script compares the performance of TileLang FP8 blockwise GEMM against DeepGEMM
and SGL-Kernel baselines.

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
import os
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
import triton

from sglang.srt.layers.tilelang_gemm_wrapper.core.config_loader import DEFAULT_M_VALUES
from sglang.srt.layers import tilelang_gemm_wrapper

# Optional deep_gemm import
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

if ENABLE_JIT_DEEPGEMM:
    try:
        from sglang.srt.layers import deep_gemm_wrapper
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor
    except ImportError:
        get_mn_major_tma_aligned_tensor = None
        ENABLE_JIT_DEEPGEMM = False

# Optional sgl_kernel import
try:
    from sgl_kernel import fp8_blockwise_scaled_mm
    SGL_KERNEL_AVAILABLE = True
except ImportError:
    fp8_blockwise_scaled_mm = None
    SGL_KERNEL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
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
    
    # Prepare DeepGEMM scale if available
    if ENABLE_JIT_DEEPGEMM:
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
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=quantiles)
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
    """Benchmark DeepGEMM kernel."""
    
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    
    def fn():
        deep_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C)
    
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=quantiles)
    return ms, min_ms, max_ms


def benchmark_sglkernel(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    M: int,
    N: int,
    rep: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark SGL-Kernel fp8_blockwise_scaled_mm."""
    # Prepare data in the format expected by sgl_kernel
    # A_scale needs to be transposed and made contiguous, then transposed back
    A_scale_sgl = A_scale.t().contiguous().t()
    # B and B_scale need to be transposed (column major layout)
    B_fp8_sgl = B_fp8.t()
    B_scale_sgl = B_scale.t()
    
    def fn():
        return fp8_blockwise_scaled_mm(A_fp8, B_fp8_sgl, A_scale_sgl, B_scale_sgl, torch.bfloat16)
    
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=quantiles)
    return ms, min_ms, max_ms


def run_benchmark(
    N: int,
    K: int,
    m_values: List[int],
    rep: int = 100,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Run benchmark comparing TileLang vs DeepGEMM vs SGL-Kernel."""
    
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
            
            # Benchmark DeepGEMM
            if ENABLE_JIT_DEEPGEMM:
                try:
                    dg_ms, _, _ = benchmark_deepgemm(
                        A_fp8, B_fp8, A_scale_deepgemm, B_scale, M, N, rep
                    )
                    dg_tflops = tflops(M, N, K, dg_ms)
                    speedup_vs_dg = dg_ms / tl_ms
                except Exception as e:
                    dg_ms = dg_tflops = speedup_vs_dg = float('nan')
            else:
                dg_ms = dg_tflops = speedup_vs_dg = float('nan')
            
            # Benchmark SGL-Kernel
            if SGL_KERNEL_AVAILABLE:
                try:
                    sk_ms, _, _ = benchmark_sglkernel(
                        A_fp8, B_fp8, A_scale, B_scale, M, N, rep
                    )
                    sk_tflops = tflops(M, N, K, sk_ms)
                    speedup_vs_sk = sk_ms / tl_ms
                except Exception as e:
                    logger.warning(f"M={M}: SGL-Kernel failed: {e}")
                    sk_ms = sk_tflops = speedup_vs_sk = float('nan')
            else:
                sk_ms = sk_tflops = speedup_vs_sk = float('nan')
            
            results.append({
                "M": M, "N": N, "K": K,
                "tl_ms": tl_ms, "dg_ms": dg_ms, "sk_ms": sk_ms,
                "tl_tflops": tl_tflops, "dg_tflops": dg_tflops, "sk_tflops": sk_tflops,
                "speedup_vs_dg": speedup_vs_dg, "speedup_vs_sk": speedup_vs_sk,
                "kernel_type": kernel_type,
            })
            
        except Exception as e:
            logger.warning(f"M={M}: Error: {e}")
    
    # Print all results at the end (to avoid interleaving with compilation logs)
    print(f"\n{'='*120}")
    print(f"Benchmark: TileLang vs DeepGEMM vs SGL-Kernel")
    print(f"N={N}, K={K}")
    print(f"DeepGEMM available: {ENABLE_JIT_DEEPGEMM}")
    print(f"SGL-Kernel available: {SGL_KERNEL_AVAILABLE}")
    print(f"{'='*120}\n")
    
    header = (
        f"{'M':>6} | {'TileLang (ms)':>14} | {'DeepGEMM (ms)':>14} | {'SGL-Kernel (ms)':>15} | "
        f"{'TL TFLOPS':>10} | {'DG TFLOPS':>10} | {'SK TFLOPS':>10} | "
        f"{'TL/DG':>7} | {'TL/SK':>7} | {'Kernel Type':>15}"
    )
    print(header)
    print("-" * len(header))
    
    for r in results:
        print(
            f"{r['M']:>6} | {r['tl_ms']:>14.4f} | {r['dg_ms']:>14.4f} | {r['sk_ms']:>15.4f} | "
            f"{r['tl_tflops']:>10.2f} | {r['dg_tflops']:>10.2f} | {r['sk_tflops']:>10.2f} | "
            f"{r['speedup_vs_dg']:>6.2f}x | {r['speedup_vs_sk']:>6.2f}x | {r['kernel_type']:>15}"
        )
    
    print("-" * len(header))
    
    # Save to file
    if output_file and results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TileLang GEMM vs DeepGEMM vs SGL-Kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--N", type=int, help="N dimension")
    parser.add_argument("--K", type=int, help="K dimension")
    parser.add_argument(
        "--all", action="store_true",
        help="Benchmark all available configurations"
    )
    parser.add_argument(
        "--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES,
        help=f"M values to benchmark (default: {DEFAULT_M_VALUES})"
    )
    parser.add_argument(
        "--rep", type=int, default=100,
        help="Benchmark repetitions (default: 100)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output CSV file"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Benchmark all available configs
        shapes = tilelang_gemm_wrapper.list_available_configs()
        
        if not shapes:
            logger.error("No configurations found. Run tuning first.")
            sys.exit(1)
        
        for N, K in sorted(shapes):
            run_benchmark(N, K, args.m_values, args.rep, args.output)
    
    elif args.N and args.K:
        run_benchmark(args.N, args.K, args.m_values, args.rep, args.output)
    
    else:
        parser.error("Either --N and --K, or --all must be specified")


if __name__ == "__main__":
    main()

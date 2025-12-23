#!/usr/bin/env python3
"""
Benchmark TileLang GEMM vs DeepGEMM

This script compares the performance of TileLang FP8 blockwise GEMM against DeepGEMM.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import triton

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Force enable TileLang GEMM
os.environ["SGLANG_ENABLE_TILELANG_GEMM"] = "1"

# Default M values
DEFAULT_M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


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
    
    return A_fp8, B_fp8, A_scale, B_scale


def benchmark_tilelang(
    wrapper,
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    M: int,
    N: int,
    rep: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark TileLang kernel."""
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    
    def fn():
        wrapper.gemm(A_fp8, B_fp8, A_scale, B_scale, C)
    
    quantiles = [0.5, 0.2, 0.8]
    try:
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=quantiles)
    except Exception:
        # Fallback if cudagraph not available
        ms, min_ms, max_ms = triton.testing.do_bench(fn, rep=rep, quantiles=quantiles)
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
    from sglang.srt.layers import deep_gemm_wrapper
    
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    
    def fn():
        deep_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C)
    
    quantiles = [0.5, 0.2, 0.8]
    try:
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=quantiles)
    except Exception:
        ms, min_ms, max_ms = triton.testing.do_bench(fn, rep=rep, quantiles=quantiles)
    return ms, min_ms, max_ms


def run_benchmark(
    N: int,
    K: int,
    m_values: List[int],
    config_dir: Optional[str] = None,
    rep: int = 100,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Run benchmark comparing TileLang vs DeepGEMM."""
    from sglang.srt.layers.tilelang_gemm_wrapper.core import TileLangGEMMWrapper
    from sglang.srt.layers import deep_gemm_wrapper
    
    # Initialize TileLang wrapper
    wrapper = TileLangGEMMWrapper(config_dir=config_dir)
    
    # Check if config exists
    if not wrapper.config_loader.config_exists(N, K):
        logger.error(f"Config not found for N={N}, K={K}")
        logger.error(f"Run tuning first: python tune_tilelang_gemm.py --N {N} --K {K}")
        return []
    
    # Check DeepGEMM availability
    deepgemm_available = deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
    
    print(f"\n{'='*90}")
    print(f"Benchmark: TileLang vs DeepGEMM")
    print(f"N={N}, K={K}")
    print(f"DeepGEMM available: {deepgemm_available}")
    print(f"{'='*90}\n")
    
    # Print header
    header = (
        f"{'M':>6} | {'TileLang (ms)':>14} | {'DeepGEMM (ms)':>14} | "
        f"{'TL TFLOPS':>10} | {'DG TFLOPS':>10} | {'Speedup':>8} | {'Kernel Type':>15}"
    )
    print(header)
    print("-" * len(header))
    
    results = []
    
    for M in m_values:
        try:
            # Prepare data
            A_fp8, B_fp8, A_scale, B_scale = prepare_data(M, N, K)
            
            # Benchmark TileLang
            try:
                tl_ms, _, _ = benchmark_tilelang(
                    wrapper, A_fp8, B_fp8, A_scale, B_scale, M, N, rep
                )
                tl_tflops = tflops(M, N, K, tl_ms)
                info = wrapper.get_kernel_info(M, N, K)
                kernel_type = info["kernel_type"]
            except Exception as e:
                logger.warning(f"M={M}: TileLang failed: {e}")
                continue
            
            # Benchmark DeepGEMM
            if deepgemm_available:
                try:
                    dg_ms, _, _ = benchmark_deepgemm(
                        A_fp8, B_fp8, A_scale, B_scale, M, N, rep
                    )
                    dg_tflops = tflops(M, N, K, dg_ms)
                    speedup = dg_ms / tl_ms
                except Exception as e:
                    dg_ms = dg_tflops = speedup = float('nan')
            else:
                dg_ms = dg_tflops = speedup = float('nan')
            
            # Print result
            print(
                f"{M:>6} | {tl_ms:>14.4f} | {dg_ms:>14.4f} | "
                f"{tl_tflops:>10.2f} | {dg_tflops:>10.2f} | "
                f"{speedup:>7.2f}x | {kernel_type:>15}"
            )
            
            results.append({
                "M": M, "N": N, "K": K,
                "tl_ms": tl_ms, "dg_ms": dg_ms,
                "tl_tflops": tl_tflops, "dg_tflops": dg_tflops,
                "speedup": speedup, "kernel_type": kernel_type,
            })
            
        except Exception as e:
            logger.warning(f"M={M}: Error: {e}")
    
    print("-" * len(header))
    
    # Summary
    if results:
        valid_speedups = [r["speedup"] for r in results if r["speedup"] == r["speedup"]]
        if valid_speedups:
            avg_speedup = sum(valid_speedups) / len(valid_speedups)
            print(f"\nAverage Speedup: {avg_speedup:.2f}x (TileLang vs DeepGEMM)")
    
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
        description="Benchmark TileLang GEMM vs DeepGEMM",
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
        "--config-dir", type=str, default=None,
        help="TileLang config directory"
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
        from sglang.srt.layers.tilelang_gemm_wrapper.core import TileLangGEMMWrapper
        wrapper = TileLangGEMMWrapper(config_dir=args.config_dir)
        shapes = wrapper.list_available_configs()
        
        if not shapes:
            logger.error("No configurations found. Run tuning first.")
            sys.exit(1)
        
        for N, K in sorted(shapes):
            run_benchmark(N, K, args.m_values, args.config_dir, args.rep, args.output)
    
    elif args.N and args.K:
        run_benchmark(args.N, args.K, args.m_values, args.config_dir, args.rep, args.output)
    
    else:
        parser.error("Either --N and --K, or --all must be specified")


if __name__ == "__main__":
    main()

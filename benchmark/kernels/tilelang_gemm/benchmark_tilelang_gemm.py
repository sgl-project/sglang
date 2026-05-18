"""Benchmark TileLang FP8 GEMM against the Triton FP8 baseline."""

from __future__ import annotations

import argparse
import csv
import logging
from typing import Iterable, List

import torch
import triton

logger = logging.getLogger(__name__)


def _tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    return 2.0 * M * N * K / latency_ms / 1e9


def _per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    M, N = x.shape
    padded_m = triton.cdiv(M, 128) * 128
    padded_n = triton.cdiv(N, 128) * 128
    x_padded = torch.zeros(
        (padded_m, padded_n), dtype=x.dtype, device=x.device
    )
    x_padded[:M, :N] = x
    x_view = x_padded.view(padded_m // 128, 128, padded_n // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:M, :N].contiguous(), (x_amax / 448.0).view(
        padded_m // 128, padded_n // 128
    )


def _prepare_data(M: int, N: int, K: int):
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    A_fp8, A_scale = sglang_per_token_group_quant_fp8(
        A.contiguous(), group_size=128, column_major_scales=False
    )
    B_fp8, B_scale = _per_block_cast_to_fp8(B.contiguous())
    return A_fp8, A_scale, B_fp8, B_scale


def _benchmark_one(
    M: int,
    N: int,
    K: int,
    rep: int,
    skip_baseline: bool,
) -> dict:
    from sglang.srt.layers import tilelang_gemm_wrapper

    A_fp8, A_scale, B_fp8, B_scale = _prepare_data(M, N, K)
    C_tl = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    tilelang_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C_tl)

    def tilelang_run():
        tilelang_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C_tl)

    tl_ms, _, _ = triton.testing.do_bench(
        tilelang_run, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    kernel_info = tilelang_gemm_wrapper.get_kernel_info(M, N, K)

    result = {
        "M": M,
        "N": N,
        "K": K,
        "tilelang_ms": tl_ms,
        "tilelang_tflops": _tflops(M, N, K, tl_ms),
        "kernel_type": kernel_info["kernel_type"],
        "baseline_ms": float("nan"),
        "baseline_tflops": float("nan"),
        "speedup": float("nan"),
        "allclose": "",
        "max_diff": float("nan"),
    }

    if skip_baseline:
        return result

    from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton

    C_ref = w8a8_block_fp8_matmul_triton(
        A_fp8, B_fp8, A_scale, B_scale, [128, 128], output_dtype=torch.bfloat16
    )

    def triton_run():
        return w8a8_block_fp8_matmul_triton(
            A_fp8, B_fp8, A_scale, B_scale, [128, 128], output_dtype=torch.bfloat16
        )

    baseline_ms, _, _ = triton.testing.do_bench(
        triton_run, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    max_diff = (C_tl - C_ref).abs().max().item()
    result.update(
        {
            "baseline_ms": baseline_ms,
            "baseline_tflops": _tflops(M, N, K, baseline_ms),
            "speedup": baseline_ms / tl_ms,
            "allclose": torch.allclose(C_tl, C_ref, rtol=1e-2, atol=1e-2),
            "max_diff": max_diff,
        }
    )
    return result


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_shape_values(values: Iterable[str]) -> list[tuple[int, int]]:
    shapes = []
    for value in values:
        try:
            N, K = (int(part) for part in value.split(",", 1))
        except Exception as err:
            raise ValueError(f"Expected N,K shape, got {value}") from err
        shapes.append((N, K))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", required=True, help="N,K shape")
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--config-path", help="Selected-config JSON file or directory")
    parser.add_argument("--export-config-path", help="Export selected configs after benchmark")
    parser.add_argument("--output", "-o", help="CSV output path")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from sglang.srt.layers import tilelang_gemm_wrapper

    if args.config_path:
        tilelang_gemm_wrapper.load_selected_configs(args.config_path)

    rows = []
    for N, K in _parse_shape_values(args.shape):
        for M in args.m_values:
            logger.info("Benchmarking M=%s, N=%s, K=%s", M, N, K)
            row = _benchmark_one(M, N, K, args.rep, args.skip_baseline)
            rows.append(row)
            logger.info("%s", row)

    if args.output:
        _write_csv(args.output, rows)

    if args.export_config_path:
        tilelang_gemm_wrapper.export_selected_configs(args.export_config_path)


if __name__ == "__main__":
    main()

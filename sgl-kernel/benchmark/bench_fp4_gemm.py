import argparse
import copy
import csv
import itertools

import pytest
import torch
import triton
from flashinfer import mm_fp4
from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def get_weight_shapes(args):
    models_tps = args.tp_sizes

    if models_tps == [4]:
        return [[1024, 3584], [7168, 256], [7168, 2304], [9216, 3584]]

    if models_tps == [8]:
        return [[512, 3584], [7168, 128], [7168, 1152], [4608, 3584]]
    return [
        [1024, 3584],
        [7168, 256],
        [7168, 2304],
        [9216, 3584],
        [512, 3584],
        [7168, 128],
        [7168, 1152],
        [4608, 3584],
    ]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            3072,
            4096,
            8192,
            16384,
        ],
        # x_vals = [64],
        x_log=False,
        line_arg="provider",
        line_vals=["cutlass", "cudnn", "trtllm"],
        line_names=["baseline cutlass fp4", "cudnn fp4", "trtllm fp4"],
        styles=[("red", "solid"), ("blue", "solid"), ("green", "solid")],
        ylabel="latency (ms)",
        plot_name="fp4_gemm_benchmark",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, dtype, correctness, csv_file):
    M = batch_size
    packed_k = K
    K = 2 * packed_k
    a_dtype = torch.randn((M, K), dtype=dtype, device="cuda")
    b_dtype = torch.randn((N, K), dtype=dtype, device="cuda")
    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)

    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = scaled_fp4_quant(a_dtype, a_global_scale)
    # print("a_fp4", a_fp4)
    b_fp4, b_scale_interleaved = scaled_fp4_quant(b_dtype, b_global_scale)
    res_fi = torch.empty((M, N), dtype=dtype, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cutlass":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: cutlass_scaled_fp4_mm(
                a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
            ),
            quantiles=quantiles,
        )
    if provider == "cudnn":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: mm_fp4(
                a_fp4,
                b_fp4.T,
                a_scale_interleaved,
                b_scale_interleaved.T,
                alpha,
                dtype,
                res_fi,
            ),
            quantiles=quantiles,
        )
    if provider == "trtllm":
        a_scale_interleaved = a_scale_interleaved.to(torch.uint8)
        b_scale_interleaved = b_scale_interleaved.to(torch.uint8)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: mm_fp4(
                a_fp4,
                b_fp4.T,
                a_scale_interleaved,
                b_scale_interleaved.T,
                alpha,
                dtype,
                res_fi,
                backend="trtllm",
            ),
            quantiles=quantiles,
        )
    if correctness:
        res_cutlass = cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
        )
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_scale_interleaved,
            b_scale_interleaved.T,
            alpha,
            dtype,
            res_fi,
            backend="cudnn",
        )
        assert torch.allclose(
            res_fi, res_cutlass, atol=1e-3, rtol=1e-3
        ), "cudnn fp4 doesn't match cutlass fp4"
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_scale_interleaved,
            b_scale_interleaved.T,
            alpha,
            dtype,
            res_fi,
            backend="trtllm",
        )
        assert torch.allclose(
            res_fi, res_cutlass, atol=1e-3, rtol=1e-3
        ), "trtllm fp4 doesn't match cutlass fp4"

    if csv_file:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([provider, M, N, K, ms])

    return ms, min_ms, max_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="List of tensor parallel sizes",
    )
    parser.add_argument(
        "--dtype",
        type=torch.dtype,
        default=torch.bfloat16,
        help="Data type",
    )
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Check correctness",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results_cutlass_cudnn.csv",
        help="CSV file to save results",
    )
    args = parser.parse_args()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["provider", "m", "n", "k", "time_ms"])

    NKs = get_weight_shapes(args)
    for N, K in NKs:
        print(f"DeepSeek-R1-0528-FP4 N={N} K={K}: ")
        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path="bench_fp4_res",
            N=N,
            K=K,
            dtype=args.dtype,
            correctness=args.correctness,
            csv_file=args.csv,
        )

    print("Benchmark finished!")

import argparse
import csv
import os
from typing import List, Tuple

import torch
import triton
from flashinfer import mm_fp4
from flashinfer.testing import bench_gpu_time_with_cupti
from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

from sglang.srt.utils import get_device_capability, is_sm100_supported

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Weight shapes are in the format: ([K, N], TP_SPLIT_DIM)
# TP split dim 0 means split K by tp size; dim 1 means split N by tp size.
DEEPSEEK_R1_MODEL = "deepseek-ai/DeepSeek-R1-0528-FP4"

WEIGHT_SHAPES = {
    "meta-llama/Llama-3.1-8B-Instruct": [
        ([4096, 6144], 1),
        ([4096, 4096], 0),
        ([4096, 28672], 1),
        ([14336, 4096], 0),
    ],
    "meta-llama/Llama-3.3-70B-Instruct": [
        ([8192, 10240], 1),
        ([8192, 8192], 0),
        ([8192, 57344], 1),
        ([28672, 8192], 0),
    ],
}

DEEPSEEK_R1_WEIGHT_SHAPES = {
    4: [[1024, 3584], [7168, 256], [7168, 2304], [9216, 3584]],
    8: [[512, 3584], [7168, 128], [7168, 1152], [4608, 3584]],
}


def _bench_cudagraph_with_cupti(fn, quantiles):
    times_ms = bench_gpu_time_with_cupti(fn=fn, use_cuda_graph=True)
    if not times_ms:
        return 0.0, 0.0, 0.0
    quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)
    times_tensor = torch.tensor(times_ms, dtype=torch.float32)
    qs = torch.quantile(times_tensor, quantiles_tensor).tolist()
    return qs[0], qs[1], qs[2]


def get_weight_shapes(args) -> List[Tuple[int, int, str]]:
    shapes: List[Tuple[int, int, str]] = []
    for model in args.models:
        if model == DEEPSEEK_R1_MODEL:
            for tp_size in args.tp_sizes:
                if tp_size in DEEPSEEK_R1_WEIGHT_SHAPES:
                    selected = DEEPSEEK_R1_WEIGHT_SHAPES[tp_size]
                else:
                    selected = (
                        DEEPSEEK_R1_WEIGHT_SHAPES[4] + DEEPSEEK_R1_WEIGHT_SHAPES[8]
                    )
                for n, packed_k in selected:
                    shapes.append((n, packed_k, model))
            continue

        if model not in WEIGHT_SHAPES:
            raise ValueError(f"Unsupported model: {model}")
        for tp_size in args.tp_sizes:
            for k_n, tp_split_dim in WEIGHT_SHAPES[model]:
                k, n = k_n
                if tp_split_dim == 0:
                    k = k // tp_size
                else:
                    n = n // tp_size
                packed_k = k // 2
                shapes.append((n, packed_k, model))
    return shapes


# CI environment uses simplified parameters
if IS_CI:
    batch_sizes = [1, 8]  # Simplified for CI
else:
    batch_sizes = [
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
    ]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=batch_sizes,
        # x_vals = [64],
        x_log=False,
        line_arg="provider",
        line_vals=["sglang_cutlass", "cutlass", "cudnn", "trtllm", "cute-dsl", "auto"],
        line_names=[
            "sglang cutlass fp4",
            "flashinfer cutlass fp4",
            "cudnn fp4",
            "trtllm fp4",
            "cute-dsl fp4",
            "auto fp4 (cudnn/cutlass)",
        ],
        styles=[
            ("red", "solid"),
            ("orange", "solid"),
            ("blue", "solid"),
            ("green", "solid"),
            ("brown", "solid"),
            ("purple", "solid"),
        ],
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
    if provider == "sglang_cutlass":
        ms, min_ms, max_ms = _bench_cudagraph_with_cupti(
            lambda: cutlass_scaled_fp4_mm(
                a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
            ),
            quantiles=quantiles,
        )
    if provider == "cutlass":
        ms, min_ms, max_ms = _bench_cudagraph_with_cupti(
            lambda: mm_fp4(
                a_fp4,
                b_fp4.T,
                a_scale_interleaved,
                b_scale_interleaved.T,
                alpha,
                dtype,
                res_fi,
                backend="cutlass",
            ),
            quantiles=quantiles,
        )
    if provider == "cudnn":
        ms, min_ms, max_ms = _bench_cudagraph_with_cupti(
            lambda: mm_fp4(
                a_fp4,
                b_fp4.T,
                a_scale_interleaved,
                b_scale_interleaved.T,
                alpha,
                dtype,
                res_fi,
                backend="cudnn",
            ),
            quantiles=quantiles,
        )
    if provider == "trtllm":
        a_scale_interleaved = a_scale_interleaved.to(torch.uint8)
        b_scale_interleaved = b_scale_interleaved.to(torch.uint8)
        ms, min_ms, max_ms = _bench_cudagraph_with_cupti(
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
    if provider == "cute-dsl":
        ms, min_ms, max_ms = _bench_cudagraph_with_cupti(
            lambda: mm_fp4(
                a_fp4,
                b_fp4.T,
                a_scale_interleaved,
                b_scale_interleaved.T,
                alpha,
                dtype,
                res_fi,
                backend="cute-dsl",
            ),
            quantiles=quantiles,
        )
    if provider == "auto":
        ms, min_ms, max_ms = _bench_cudagraph_with_cupti(
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
        "--models",
        nargs="+",
        type=str,
        default=[DEEPSEEK_R1_MODEL],
        help="List of models to benchmark. Supported: Llama 8B/70B and deepseek-ai/DeepSeek-R1-0528-FP4.",
    )
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
        help="Output data type",
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

    # Simplify for CI environment
    if IS_CI:
        args.tp_sizes = [args.tp_sizes[0]]  # Use only first TP size

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["provider", "m", "n", "k", "time_ms"])

    # FP4 operations require Blackwell SM100 support
    major, minor = get_device_capability()
    if not is_sm100_supported():
        print("Skipping FP4 GEMM benchmark")
        if major is not None:
            print(
                f"FP4 operations require SM100 (Blackwell), but found sm{major}{minor}"
            )
        else:
            print("Could not determine device capability")
    else:
        NKs = get_weight_shapes(args)

        # Limit iterations in CI
        if IS_CI:
            NKs = NKs[:2]  # Only test first 2 shapes in CI

        for N, K, model_name in NKs:
            print(f"{model_name} N={N} packed_k={K}: ")
            benchmark.run(
                print_data=True,
                N=N,
                K=K,
                dtype=args.dtype,
                correctness=args.correctness,
                csv_file=args.csv,
            )
        print("Benchmark finished!")

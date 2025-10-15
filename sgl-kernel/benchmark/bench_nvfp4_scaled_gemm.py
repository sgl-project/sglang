import argparse
import copy
import itertools
import os

import torch
import triton
from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

from sglang.srt.utils import get_device_capability

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Weight Shapes are in the format
# ([K, N], TP_SPLIT_DIM)
# Example:
#  A shape of ([14336, 4096], 0) indicates the following GEMM shape,
#   - TP1 : K = 14336, N = 4096
#   - TP2 : K = 7168, N = 4096
#  A shape of ([4096, 6144], 1) indicates the following GEMM shape,
#   - TP1 : K = 4096, N = 6144
#   - TP4 : K = 4096, N = 1536

# TP1 shapes
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
    "mistralai/Mistral-Large-Instruct-2407": [
        ([12288, 14336], 1),
        ([12288, 12288], 0),
        ([12288, 57344], 1),
        ([28672, 12288], 0),
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        ([3584, 4608], 1),
        ([3584, 3584], 0),
        ([3584, 37888], 1),
        ([18944, 3584], 0),
    ],
    "Qwen/Qwen2.5-32B-Instruct": [
        ([5120, 7168], 1),
        ([5120, 5120], 0),
        ([5120, 55296], 1),
        ([27648, 5120], 0),
    ],
    "Qwen/Qwen2.5-72B-Instruct": [
        ([8192, 10240], 1),
        ([8192, 8192], 0),
        ([8192, 59136], 1),
        ([29568, 8192], 0),
    ],
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": [
        ([2048, 3072], 1),
        ([2048, 4096], 1),
        ([2048, 2048], 0),
        ([2048, 576], 0),
        ([2048, 21888], 1),
        ([10944, 2048], 0),
        ([2048, 2816], 1),
        ([1408, 2048], 0),
    ],
}


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=[
            "sglang-fp4-fp16",
            "sglang-fp4-bf16",
        ],
        line_names=[
            "sglang-fp4-fp16",
            "sglang-fp4-bf16",
        ],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="fp4 block scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    # M, N, K = batch_size, 4096, 8192
    run_step = 100
    dtype = torch.float16 if "fp16" in provider else torch.bfloat16
    M = batch_size
    a = torch.randn((M, K), dtype=dtype, device="cuda")
    b = torch.randn((N, K), dtype=dtype, device="cuda")
    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = scaled_fp4_quant(a, a_global_scale)
    b_fp4, b_scale_interleaved = scaled_fp4_quant(b, b_global_scale)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Bridging the gap between CPU and GPU
    for _ in range(25):
        c = a @ b.t()
    # Warmup
    for _ in range(5):
        cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
        )
    start_event.record()
    for _ in range(run_step):
        cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
        )
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event) / run_step

    tflops = lambda ms: (2 * M * N * K) * 1e-9 / ms
    return tflops(ms)


def prepare_shapes(args):
    KN_model_names = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        assert model in WEIGHT_SHAPES
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KN.append(model)
            KN_model_names.append(KN)
    return KN_model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        help="List of models to benchmark",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="List of tensor parallel sizes",
    )
    args = parser.parse_args()

    # Check architecture compatibility - FP4 operations require sm100a/sm103a
    major, minor = get_device_capability()
    if major is None or major < 10:  # Requires compute capability 10.0+ (sm100a/sm103a)
        print("Skipping NVIDIA FP4 scaled GEMM benchmark")
        if major is not None:
            print(f"FP4 operations require sm100a/sm103a, but found sm{major}{minor}")
        else:
            print("Could not determine device capability")
    else:
        KN_model_names = prepare_shapes(args)

        # Limit iterations in CI
        if IS_CI:
            KN_model_names = KN_model_names[:2]  # Only test first 2 shapes in CI

        for K, N, model_name in KN_model_names:
            print(f"{model_name} N={N} K={K}: ")
            benchmark.run(print_data=True, N=N, K=K)
            print("Benchmark finished!")

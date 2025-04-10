import argparse
import copy
import itertools

import deep_gemm
import torch
import triton
from deep_gemm import get_col_major_tma_aligned_tensor
from sgl_kernel import fp8_blockwise_scaled_mm
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm

from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul


def get_weight_shapes(args):
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    # NOTE(HandH1998): The weight shapes only works for DeepSeek-V3. Modify them, if you tune for another different model.
    # cannot TP
    total = [
        (512 + 64, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (7168, 16384),
        (7168, 18432),
    ]
    # N can TP
    n_tp = [
        (18432 * 2, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (24576, 1536),
        (4096, 7168),
    ]
    # K can TP
    k_tp = [(7168, 18432), (7168, 16384), (7168, 2048)]
    # only support Deepseek-V3
    SUPPORT_MODEL = ["deepseek-ai/DeepSeek-V3"]

    weight_shapes = []
    for model, tp_size in models_tps:
        assert model in SUPPORT_MODEL
        for t in total:
            new_t = [t[0], t[1], model]
            weight_shapes.append(new_t)
        for n_t in n_tp:
            new_t = [n_t[0] // tp_size, n_t[1], model]
            weight_shapes.append(new_t)
        for k_t in k_tp:
            new_t = [k_t[0], k_t[1] // tp_size, model]
            weight_shapes.append(new_t)
    return weight_shapes


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def fp8_gemm_deepgemm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
):
    """DeepGEMM implementation of FP8 GEMM"""
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    # Run DeepGEMM kernel
    deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (y_fp8, y_scale), out)
    return out


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=["vllm", "sgl-kernel", "triton", "deepgemm"],
        line_names=["vllm", "sgl-kernel", "sglang triton", "deepgemm"],
        styles=[("blue", "-"), ("orange", "-"), ("red", "-"), ("yellow", "-")],
        ylabel="GB/s",
        plot_name="fp8 blockwise scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)
    scale_a_shape = scale_shape(a_fp8.shape, scale_a_group_shape)
    scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)

    scale_a = torch.randn(scale_a_shape, device="cuda", dtype=torch.float32)
    scale_b = torch.randn(scale_b_shape, device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "sgl-kernel":
        scale_a = scale_a.t().contiguous().t()
        b_fp8, scale_b = b_fp8.t(), scale_b.t()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fp8_blockwise_scaled_mm(
                a_fp8, b_fp8, scale_a, scale_b, torch.float16
            ),
            quantiles=quantiles,
        )
    if provider == "vllm":
        scale_a = scale_a.t().contiguous().t()
        b_fp8, scale_b = b_fp8.t(), scale_b.t()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, torch.float16),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: w8a8_block_fp8_matmul(
                a_fp8, b_fp8, scale_a, scale_b, [128, 128], torch.float16
            ),
            quantiles=quantiles,
        )
    if provider == "deepgemm":
        scale_a_col_major = get_col_major_tma_aligned_tensor(scale_a.clone())
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fp8_gemm_deepgemm(
                a_fp8, scale_a_col_major, b_fp8, scale_b, M, N, K
            ),
            quantiles=quantiles,
        )
    return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["deepseek-ai/DeepSeek-V3"],
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

    NK_model_names = get_weight_shapes(args)
    for N, K, model_name in NK_model_names:
        if N % 128 != 0 or K % 128 != 0:
            print(f"Skip {N=}, {K=} now")
            continue
        print(f"{model_name} N={N} K={K}: ")
        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path="bench_fp8_blockwise_res",
            N=N,
            K=K,
        )

    print("Benchmark finished!")

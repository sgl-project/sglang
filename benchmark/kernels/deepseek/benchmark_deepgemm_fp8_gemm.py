from typing import Tuple

import deep_gemm
import tilelang
import tilelang.language as T
import torch
import triton
from deep_gemm import ceil_div, get_col_major_tma_aligned_tensor
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_block_fp8_matmul as vllm_w8a8_block_fp8_matmul,
)

from sglang.srt.layers.quantization.fp8_kernel import (
    w8a8_block_fp8_matmul_deepgemm as w8a8_block_fp8_matmul,
)


# Adapted from https://github.com/tile-ai/tilelang/blob/a8cfdce92795cb861c9033573534653ee040b5ed/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py#L1
def tl_gemm(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        "e4m3_float8",
    ], "Currently only e4m3_float8 is supported"
    assert out_dtype in [
        "bfloat16",
        "float16",
    ], "Currently only bfloat16 and float16 are supported"

    TILE_SIZE = (128, 128, 128)
    block_M = TILE_SIZE[0]
    block_N = TILE_SIZE[1]
    block_K = TILE_SIZE[2]

    A_shape = (M, K)
    Scales_A_shape = (M, T.ceildiv(K, block_K))
    B_shape = (N, K)
    Scales_B_shape = (T.ceildiv(N, block_N), T.ceildiv(K, block_K))
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (block_M, block_N)

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, in_dtype),
        scales_a: T.Buffer(Scales_A_shape, "float32"),
        B: T.Buffer(B_shape, in_dtype),
        scales_b: T.Buffer(Scales_B_shape, "float32"),
        C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), "float32")
            C_local = T.alloc_fragment(C_shared_shape, accum_dtype)
            C_local_accum = T.alloc_fragment(C_shared_shape, accum_dtype)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                # Load A into shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Load B into shared memory
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Load scale into shared memory
                Scale_B = scales_b[bx, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


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


def fp8_gemm_sglang(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
):
    """SGLang implementation of FP8 GEMM"""
    block_size = [128, 128]  # Matches the block size in per_block_cast_to_fp8

    # Run SGLang kernel
    out = w8a8_block_fp8_matmul(
        x_fp8, y_fp8, x_scale, y_scale, block_size, torch.bfloat16
    )
    return out


def fp8_gemm_vllm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    y_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    m: int,
    n: int,
    k: int,
):
    """vLLM implementation of FP8 GEMM"""
    block_size = [128, 128]  # Matches the block size in per_block_cast_to_fp8

    # Run vLLM kernel
    out = vllm_w8a8_block_fp8_matmul(
        x_fp8, y_fp8, x_scale, y_scale, block_size, torch.bfloat16
    )
    return out


def calculate_diff(m: int, n: int, k: int):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    x_fp8, x_scale = per_token_cast_to_fp8(x.clone())
    y_fp8, y_scale = per_block_cast_to_fp8(y.clone())
    x_scale_col_major = get_col_major_tma_aligned_tensor(x_scale.clone())

    out_deepgemm = fp8_gemm_deepgemm(
        x_fp8.clone(),
        x_scale_col_major.clone(),
        y_fp8.clone(),
        y_scale.clone(),
        m,
        n,
        k,
    )
    out_sglang = fp8_gemm_sglang(
        x_fp8.clone(), x_scale.clone(), y_fp8.clone(), y_scale.clone(), m, n, k
    )

    tilelang_func = tl_gemm(m, n, k, "e4m3_float8", "bfloat16", "float32")
    tilelang_kernel = tilelang.compile(tilelang_func, out_idx=[-1])
    out_tilelang = tilelang_kernel(
        x_fp8.clone(), x_scale.clone(), y_fp8.clone(), y_scale.clone()
    )

    diff_sglang_deepgemm = torch.abs(out_deepgemm - out_sglang).mean().item()
    diff_tilelang_deepgemm = torch.abs(out_deepgemm - out_tilelang).mean().item()
    diff_tilelang_sglang = torch.abs(out_tilelang - out_sglang).mean().item()

    print(f"Shape m={m}, n={n}, k={k}:")
    print(f"DeepGEMM output: {out_deepgemm[0, 0:5]}")
    print(f"SGLang output: {out_sglang[0, 0:5]}")
    print(f"TileLang output: {out_tilelang[0, 0:5]}")
    print(f"Mean absolute difference (SGLang-DeepGEMM): {diff_sglang_deepgemm}")
    print(f"Mean absolute difference (TileLang-DeepGEMM): {diff_tilelang_deepgemm}")
    print(f"Mean absolute difference (TileLang-SGLang): {diff_tilelang_sglang}")

    sglang_deepgemm_match = torch.allclose(
        out_deepgemm, out_sglang, atol=1e-2, rtol=1e-2
    )
    tilelang_deepgemm_match = torch.allclose(
        out_deepgemm, out_tilelang, atol=1e-2, rtol=1e-2
    )
    tilelang_sglang_match = torch.allclose(
        out_tilelang, out_sglang, atol=1e-2, rtol=1e-2
    )

    if sglang_deepgemm_match and tilelang_deepgemm_match and tilelang_sglang_match:
        print("✅ All implementations match\n")
    else:
        print("❌ Some implementations differ:")
        print(f"  - SGLang vs DeepGEMM: {'✅' if sglang_deepgemm_match else '❌'}")
        print(f"  - TileLang vs DeepGEMM: {'✅' if tilelang_deepgemm_match else '❌'}")
        print(f"  - TileLang vs SGLang: {'✅' if tilelang_sglang_match else '❌'}\n")


def get_weight_shapes(tp_size):
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

    weight_shapes = []
    for t in total:
        weight_shapes.append(t)
    for n_t in n_tp:
        new_t = (n_t[0] // tp_size, n_t[1])
        weight_shapes.append(new_t)
    for k_t in k_tp:
        new_t = (k_t[0], k_t[1] // tp_size)
        weight_shapes.append(new_t)

    return weight_shapes


def create_benchmark_configs(tp_size):
    configs = []
    weight_shapes = get_weight_shapes(tp_size)
    batch_sizes = [8, 16, 32, 64, 128, 256, 1024, 2048, 4096]

    for n, k in weight_shapes:
        for m in batch_sizes:
            configs.append((m, n, k, tp_size))

    return configs


def get_benchmark(tp_size):
    all_configs = create_benchmark_configs(tp_size)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "tp_size"],
            x_vals=[list(config) for config in all_configs],
            line_arg="provider",
            line_vals=["deepgemm", "sglang", "tilelang"],
            line_names=["DeepGEMM", "SGLang", "TileLang"],
            styles=[("blue", "-"), ("red", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"fp8-gemm-performance-comparison-tp{tp_size}",
            args={},
        )
    )
    def benchmark(m, n, k, tp_size, provider):
        print(f"Shape (m={m}, n={n}, k={k}, tp={tp_size}), Provider: {provider}")
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        # Preprocess data before benchmarking
        x_fp8, x_scale = per_token_cast_to_fp8(x)
        y_fp8, y_scale = per_block_cast_to_fp8(y)
        x_scale_col_major = get_col_major_tma_aligned_tensor(x_scale.clone())

        quantiles = [0.5, 0.2, 0.8]

        if provider == "deepgemm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_deepgemm(
                    x_fp8.clone(),
                    x_scale_col_major.clone(),
                    y_fp8.clone(),
                    y_scale.clone(),
                    m,
                    n,
                    k,
                ),
                quantiles=quantiles,
            )
        elif provider == "sglang":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_sglang(
                    x_fp8.clone(),
                    x_scale.clone(),
                    y_fp8.clone(),
                    y_scale.clone(),
                    m,
                    n,
                    k,
                ),
                quantiles=quantiles,
            )
        else:  # tilelang
            tilelang_func = tl_gemm(m, n, k, "e4m3_float8", "bfloat16", "float32")
            tilelang_kernel = tilelang.compile(tilelang_func, out_idx=[-1])
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: tilelang_kernel(
                    x_fp8.clone(),
                    x_scale.clone(),
                    y_fp8.clone(),
                    y_scale.clone(),
                ),
                quantiles=quantiles,
            )

        # Calculate TFLOPS
        flops = 2 * m * n * k  # multiply-adds
        tflops = flops / (ms * 1e-3) / 1e12

        # Print shape-specific results with TFLOPS
        print(f"Time: {ms*1000:.2f} ms, TFLOPS: {tflops:.2f}")
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/fp8_gemm/",
        help="Path to save fp8 gemm benchmark results",
    )
    parser.add_argument(
        "--run_correctness",
        action="store_true",
        default=True,
        help="Whether to run correctness test",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size to benchmark (default: 1)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Enable TF32, adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_core.py#L148
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run correctness tests on a few examples
    if args.run_correctness:
        print("Running correctness tests...")
        calculate_diff(64, 512, 7168)  # Small test
        calculate_diff(64, 7168, 16384)  # Medium test
        calculate_diff(64, 18432, 7168)  # Large test

    # Get the benchmark function with the specified tp_size
    benchmark = get_benchmark(args.tp_size)

    print(f"Running performance benchmark for TP size = {args.tp_size}...")
    benchmark.run(print_data=True, save_path=args.save_path)

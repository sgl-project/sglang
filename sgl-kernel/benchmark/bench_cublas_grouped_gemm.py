import argparse

import torch
import triton
import triton.language as tl
from sgl_kernel import cublas_grouped_gemm

WEIGHT_CONFIGS = {
    "DeepSeek-V2-Lite": {
        "num_routed_experts": 64,
        "ffn_shapes": [
            [2048, 2816],
            [1408, 2048],
        ],
    },
    "DeepSeek-V2": {
        "num_routed_experts": 160,
        "ffn_shapes": [
            [5120, 3072],
            [1536, 5120],
        ],
    },
}


# This Triton Grouped Gemm Kernel is adapted from
# https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # Factors for multiplication.
    alphas,
    betas,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # load multiplication factors
        alpha = tl.load(alphas + g)
        beta = tl.load(betas + g)
        # iterate through the tiles in the current gemm problem
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                a = tl.load(
                    a_ptrs,
                    mask=(offs_am[:, None] < gm)
                    and (offs_k[None, :] < gk - kk * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < gk - kk * BLOCK_SIZE_K)
                    and (offs_bn[None, :] < gn),
                    other=0.0,
                )
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            accumulator *= alpha
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            output_mask = (offs_am[:, None] < gm) and (offs_bn[None, :] < gn)
            c += beta * tl.load(c_ptrs, mask=output_mask)
            tl.store(c_ptrs, c, mask=output_mask)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def triton_perf_fn(group_A, group_B, group_C, dtype):
    # We put the process of matrix lengths and pointers here out of fairness,
    # since cublas_grouped_gemm kernel also does these work.
    group_size = len(group_A)
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    alphas = [1.0] * group_size
    betas = [0.0] * group_size
    for i in range(group_size):
        M, N, K = group_A[i].shape[0], group_B[i].shape[1], group_A[i].shape[1]
        g_sizes += [M, N, K]
        g_lds += [K, N, N]
        A_addrs.append(group_A[i].data_ptr())
        B_addrs.append(group_B[i].data_ptr())
        C_addrs.append(group_C[i].data_ptr())

    d_a_ptrs = torch.tensor(A_addrs, device="cuda")
    d_b_ptrs = torch.tensor(B_addrs, device="cuda")
    d_c_ptrs = torch.tensor(C_addrs, device="cuda")
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")
    d_alphas = torch.tensor(alphas, dtype=torch.float32, device="cuda")
    d_betas = torch.tensor(betas, dtype=torch.float32, device="cuda")

    NUM_SM = 128
    grid = (NUM_SM,)
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        d_alphas,
        d_betas,
        group_size,
        NUM_SM=NUM_SM,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )


def cublas_perf_fn(group_A, group_B, group_C, dtype):
    cublas_grouped_gemm(group_A, group_B, group_C, dtype)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        line_vals=[
            "triton",
            "cublas",
        ],
        line_names=[
            "triton",
            "cublas",
        ],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="gbps",
        plot_name="grouped gemm",
        args={},
    )
)
def benchmark(M, provider, N, K):
    group_size = 20  # Number of used experts per gpu is usually around 20
    group_A = []
    group_B_row_major = []
    group_B_col_major = []
    group_C = []
    dtype = torch.float16
    for i in range(group_size):
        A = torch.rand((M, K), device="cuda", dtype=dtype)
        B_row_major = torch.rand((K, N), device="cuda", dtype=dtype)
        B_col_major = torch.rand((N, K), device="cuda", dtype=dtype)
        C = torch.empty((M, N), device="cuda", dtype=dtype)
        group_A.append(A)
        group_B_row_major.append(B_row_major)
        group_B_col_major.append(B_col_major)
        group_C.append(C)

    quantiles = [0.5, 0.2, 0.8]
    if "triton" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(group_A, group_B_row_major, group_C, dtype),
            quantiles=quantiles,
        )
    elif "cublas" in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cublas_perf_fn(group_A, group_B_col_major, group_C, dtype),
            quantiles=quantiles,
        )

    gbps = (
        lambda ms: group_size
        * (2 * M * N * K + 2 * M * N)
        * group_A[0].element_size()
        * 1e-9
        / (ms * 1e-3)
    )
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["DeepSeek-V2"],
        help="List of models to benchmark",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor parallel size",
    )
    args = parser.parse_args()
    for model in args.models:
        assert model in WEIGHT_CONFIGS
        num_experts_per_device = (
            WEIGHT_CONFIGS[model]["num_routed_experts"] // args.tp_size
        )
        for K, N in WEIGHT_CONFIGS[model]["ffn_shapes"]:
            print(
                f"{model} N={N} K={K} tp_size={args.tp_size} "
                f"group_size=num_experts_per_device={num_experts_per_device}: "
            )
            benchmark.run(
                print_data=True,
                show_plots=True,
                save_path="bench_grouped_gemm_res",
                N=N,
                K=K,
            )

    print("Benchmark finished!")

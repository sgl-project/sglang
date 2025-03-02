import itertools
from typing import Tuple

import deep_gemm
import numpy as np
import torch
import triton
import triton.language as tl
from deep_gemm import calc_diff, ceil_div, get_col_major_tma_aligned_tensor


def construct_grouped(
    x: torch.Tensor, y: torch.Tensor, num_groups: int, is_masked: bool
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    # Verify input shapes
    m, k = x.shape
    n, k_y = y.shape
    assert k == k_y, f"Incompatible shapes: x({m}, {k}), y({n}, {k_y})"
    assert m % num_groups == 0, f"m({m}) must be divisible by num_groups({num_groups})"
    assert m % 4 == 0, f"TMA alignment error: {m}"

    # Reshape inputs for grouped processing
    m_per_group = m // num_groups
    x = x.view(num_groups, m_per_group, k)
    y = y.unsqueeze(0).expand(num_groups, n, k)

    # Initialize output tensors
    out = torch.empty((num_groups, m_per_group, n), device="cuda", dtype=torch.bfloat16)
    ref_out = torch.einsum("gmk,gnk->gmn", x, y)

    x_fp8 = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, m_per_group, k // 128), device="cuda", dtype=torch.float
        ),
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, (n + 127) // 128, k // 128), device="cuda", dtype=torch.float
        ),
    )
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier for testing
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))

    return x_fp8, y_fp8, out, ref_out


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


# Reference: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
@triton.jit
def fp8_gemm_group_triton_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fp8_gemm_group_triton(a, b, num_groups):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape
    # Transpose b to match kernel expectations
    b = b.T.contiguous()
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fp8_gemm_group_triton_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=num_groups,
    )
    return c


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
    num_groups = [4, 8]
    for n, k in weight_shapes:
        for m in batch_sizes:
            for num_groups in num_groups:
                configs.append((m, n, k, num_groups, tp_size))

    return configs


def calculate_diff(m: int, n: int, k: int, num_groups: int):
    print(f"Shape (m={m}, n={n}, k={k}")
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    x_fp8, y_fp8, out, out_torch = construct_grouped(x, y, num_groups, is_masked=False)
    m_per_group = m // num_groups
    out_deepgemm = out.clone()
    m_indices = torch.arange(0, num_groups, device="cuda", dtype=torch.int)
    m_indices = (
        m_indices.unsqueeze(-1).expand(num_groups, m_per_group).contiguous().view(-1)
    )

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        (x_fp8[0].clone(), x_fp8[1].clone()),
        (y_fp8[0].clone(), y_fp8[1].clone()),
        out_deepgemm,
        m_indices,
    )
    torch.cuda.synchronize()

    # without fp8
    out_triton = fp8_gemm_group_triton(x.clone(), y.clone(), num_groups)
    torch.cuda.synchronize()

    diff_torch_deepgemm = torch.abs(out_torch - out_deepgemm).mean().item()
    diff_torch_triton = torch.abs(out_torch - out_triton).mean().item()

    print(f"Shape m={m}, n={n}, k={k}:")
    print(f"Torch output: {out_torch[0, 0:10]}")
    print(f"DeepGEMM output: {out_deepgemm[0, 0:10]}")
    print(f"Triton output: {out_triton[0, 0:10]}")
    print(f"Mean absolute difference (Torch-DeepGEMM): {diff_torch_deepgemm}")
    print(f"Mean absolute difference (Torch-Triton): {diff_torch_triton}")

    deepgemm_torch_diff = calc_diff(out_deepgemm, out_torch)
    triton_torch_diff = calc_diff(out_triton, out_torch)
    deepgemm_triton_diff = calc_diff(out_deepgemm, out_triton)

    diff_threshold = 0.001
    all_match = (
        deepgemm_torch_diff < diff_threshold
        and triton_torch_diff < diff_threshold
        and deepgemm_triton_diff < diff_threshold
    )
    if all_match:
        print("✅ All implementations match\n")
    else:
        print("❌ Some implementations differ:")
        print(
            f"  - Torch vs DeepGEMM: {'✅' if deepgemm_torch_diff < diff_threshold else '❌'}"
            f"  - Torch vs Triton: {'✅' if triton_torch_diff < diff_threshold else '❌'}"
            f"  - DeepGEMM vs Triton: {'✅' if deepgemm_triton_diff < diff_threshold else '❌'}"
        )


def get_benchmark(tp_size):
    all_configs = create_benchmark_configs(tp_size)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "num_groups", "tp_size"],
            x_vals=[config for config in all_configs],
            line_arg="provider",
            line_vals=["deepgemm", "triton", "torch"],
            line_names=["DeepGEMM", "Triton", "Torch"],
            styles=[("blue", "-"), ("red", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"fp8-group-gemm-performance-comparison-tp{tp_size}",
            args={},
        )
    )
    def benchmark(m, n, k, num_groups, tp_size, provider):
        print(
            f"Shape (m={m}, n={n}, k={k}, tp={tp_size}, num_groups={num_groups}, Provider: {provider}"
        )
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        x_fp8, y_fp8, out, out_torch = construct_grouped(
            x, y, num_groups, is_masked=False
        )
        m_per_group = m // num_groups
        m_indices = torch.arange(0, num_groups, device="cuda", dtype=torch.int)
        m_indices = (
            m_indices.unsqueeze(-1)
            .expand(num_groups, m_per_group)
            .contiguous()
            .view(-1)
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "deepgemm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    x_fp8,
                    y_fp8,
                    out,
                    m_indices,
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_group_triton(
                    x_fp8,
                    y_fp8,
                    num_groups,
                ),
                quantiles=quantiles,
            )

        # Calculate TFLOPS
        flops = 2 * m * n * k  # multiply-adds
        tflops = flops / (ms * 1e-3) / 1e12

        # Print shape-specific results with TFLOPS
        print(f"Time: {ms:.2f} ms, TFLOPS: {tflops:.2f}")
        return (
            ms,
            max_ms,
            min_ms,
        )  # return in seconds for consistency with triton benchmark

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/deepseek/fp8_group_gemm/",
        help="Path to save deepgemm fp8 group gemm benchmark results",
    )
    parser.add_argument(
        "--run_correctness",
        action="store_true",
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
    # if args.run_correctness:
    if True:
        print("Running correctness tests...")
        calculate_diff(8192, 7168, 4096, 4)
        calculate_diff(8192, 2048, 7168, 4)
        calculate_diff(4096, 7168, 4096, 8)
        calculate_diff(4096, 2048, 7168, 8)

    # Get the benchmark function with the specified tp_size
    # benchmark = get_benchmark(args.tp_size)

    # print(f"Running performance benchmark for TP size = {args.tp_size}...")
    # benchmark.run(print_data=True, save_path=args.save_path)

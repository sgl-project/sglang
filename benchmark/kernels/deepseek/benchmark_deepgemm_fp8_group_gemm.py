from typing import Tuple

import deep_gemm
import torch
import triton
import triton.language as tl
from deep_gemm import calc_diff
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

# Import shared functionality from the regular GEMM benchmark
from sglang.benchmark.kernels.deepseek.benchmark_deepgemm_fp8_gemm import (
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)


def construct_grouped_and_flat_fp8(
    x: torch.Tensor, y: torch.Tensor, num_groups: int, is_masked: bool
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],  # grouped x_fp8
    Tuple[torch.Tensor, torch.Tensor],  # grouped y_fp8
    Tuple[torch.Tensor, torch.Tensor],  # flat x_fp8
    Tuple[torch.Tensor, torch.Tensor],  # flat y_fp8
    torch.Tensor,  # output
    torch.Tensor,  # reference output
]:
    # Verify input shapes
    m, k = x.shape
    n, k_y = y.shape
    assert k == k_y, f"Incompatible shapes: x({m}, {k}), y({n}, {k_y})"
    assert m % num_groups == 0, f"m({m}) must be divisible by num_groups({num_groups})"
    assert m % 4 == 0, f"TMA alignment error: {m}"

    # Reshape inputs for grouped processing
    m_per_group = m // num_groups
    x_grouped = x.view(num_groups, m_per_group, k)
    y_grouped = y.unsqueeze(0).expand(num_groups, n, k)

    # Initialize output tensors
    out = torch.empty((num_groups, m_per_group, n), device="cuda", dtype=torch.bfloat16)
    ref_out = torch.einsum("gmk,gnk->gmn", x_grouped, y_grouped)

    # Quantize grouped tensors
    x_fp8_grouped = (
        torch.empty_like(x_grouped, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, m_per_group, k // 128), device="cuda", dtype=torch.float
        ),
    )
    y_fp8_grouped = (
        torch.empty_like(y_grouped, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, (n + 127) // 128, k // 128), device="cuda", dtype=torch.float
        ),
    )
    for i in range(num_groups):
        x_fp8_grouped[0][i], x_fp8_grouped[1][i] = per_token_cast_to_fp8(x_grouped[i])
        y_fp8_grouped[0][i], y_fp8_grouped[1][i] = per_block_cast_to_fp8(y_grouped[i])

    # Quantize flat tensors
    x_fp8_flat = per_token_cast_to_fp8(x)
    y_fp8_flat = per_block_cast_to_fp8(y)

    # For non-masked input, merge the group and M dims in output
    if not is_masked:
        x_fp8_grouped = (
            x_fp8_grouped[0].view(-1, k),
            per_token_cast_to_fp8(x_grouped.view(-1, k))[1],
        )
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier for testing
    x_fp8_grouped = (
        x_fp8_grouped[0],
        get_mn_major_tma_aligned_tensor(x_fp8_grouped[1]),
    )
    x_fp8_flat = (x_fp8_flat[0], get_mn_major_tma_aligned_tensor(x_fp8_flat[1]))

    return x_fp8_grouped, y_fp8_grouped, x_fp8_flat, y_fp8_flat, out, ref_out


# Since we don't have a group gemm kernel in SGLang/vLLM, we implemented a
# custom kernel based on the Triton tutorial.
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
@triton.jit
def fp8_gemm_group_triton_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Pointers to scaling factors
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Strides for scaling factors
    stride_a_scale_m,
    stride_a_scale_k,
    stride_b_scale_n,
    stride_b_scale_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B with FP8 inputs and scaling factors.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    Note: Block sizes must be multiples of 32 for optimal TMA performance.
    """
    # Map program ids to the block of C it should compute
    pid_group = tl.program_id(axis=0)  # Group ID
    pid_n = tl.program_id(axis=1)  # N dimension ID

    # Compute the M block ID within this group
    group_size_m = min(M - pid_group * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m_within_group = tl.program_id(axis=2) % group_size_m
    pid_m = pid_group * GROUP_SIZE_M + pid_m_within_group

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k_block * BLOCK_SIZE_K

        # Load the next block of A and B, with masks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_offset, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_offset, other=0.0)

        # Calculate indices for scaling factors for this K block
        a_scale_ptrs = a_scale_ptr + (
            offs_am * stride_a_scale_m + k_block * stride_a_scale_k
        )
        b_scale_ptrs = b_scale_ptr + (
            pid_n * stride_b_scale_n + k_block * stride_b_scale_k
        )

        # Perform matrix multiplication in FP8
        res = tl.dot(a, b)

        # Load scaling factors for the current block
        a_scale = tl.load(a_scale_ptrs)[:, None]  # [BLOCK_SIZE_M, 1]
        b_scale = tl.load(b_scale_ptrs)

        # Apply scaling factors to the accumulated result
        accumulator += res * a_scale * b_scale

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert to bfloat16 for output
    c = accumulator.to(tl.bfloat16)

    # Write back the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fp8_gemm_group_triton(a_tuple, b_tuple, c, num_groups):
    """
    Perform matrix multiplication with FP8 inputs and proper scaling.

    Args:
        a_tuple: Tuple of (quantized_tensor, scale_factors) for input A
        b_tuple: Tuple of (quantized_tensor, scale_factors) for input B
        c: Output tensor in BF16 format
        num_groups: Number of groups for grouped GEMM

    Returns:
        Result tensor in BF16 format
    """
    # Unpack the tuples
    a, a_scale = a_tuple
    b, b_scale = b_tuple

    M, K = a.shape
    _, N = b.shape

    # Configure block sizes - must be multiples of 32 for TMA alignment
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    # Calculate grid dimensions
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_groups_grid = triton.cdiv(num_pid_m, num_groups)

    # 3D grid launch - (group, n_blocks, m_blocks_per_group)
    grid = (num_groups_grid, num_pid_n, min(num_groups, num_pid_m))

    fp8_gemm_group_triton_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.stride(0),
        1,  # Stride in the K dimension may be 1
        b_scale.stride(0),
        1 if b_scale.dim() > 1 else 0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=num_groups,
    )

    return c


def fp8_gemm_group_deepgemm(x_fp8_grouped, y_fp8_grouped, out, m_indices):
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        x_fp8_grouped,
        y_fp8_grouped,
        out,
        m_indices,
    )
    return out


def calculate_diff(m: int, n: int, k: int, num_groups: int):
    print(f"Shape (m={m}, n={n}, k={k}")
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    x_fp8_grouped, y_fp8_grouped, x_fp8_flat, y_fp8_flat, out, out_torch = (
        construct_grouped_and_flat_fp8(x, y, num_groups, is_masked=False)
    )
    m_per_group = m // num_groups
    out_deepgemm = out.clone()
    m_indices = torch.arange(0, num_groups, device="cuda", dtype=torch.int)
    m_indices = (
        m_indices.unsqueeze(-1).expand(num_groups, m_per_group).contiguous().view(-1)
    )

    fp8_gemm_group_deepgemm(
        x_fp8_grouped,
        y_fp8_grouped,
        out_deepgemm,
        m_indices,
    )
    torch.cuda.synchronize()

    # Prepare inputs for Triton
    a, a_scale = x_fp8_flat
    b, b_scale = y_fp8_flat
    b = b.T.contiguous()
    # Ensure scales are in the right format and contiguous
    a_scale, b_scale = a_scale.contiguous(), b_scale.contiguous()
    M, _ = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    out_triton = fp8_gemm_group_triton((a, a_scale), (b, b_scale), c, num_groups)
    torch.cuda.synchronize()

    diff_torch_deepgemm = torch.abs(out_torch - out_deepgemm).mean().item()
    diff_torch_triton = torch.abs(out_torch - out_triton).mean().item()
    diff_deepgemm_triton = torch.abs(out_deepgemm - out_triton).mean().item()

    print(f"Shape m={m}, n={n}, k={k}:")
    print(f"Torch output: {out_torch[0, 0:5]}")
    print(f"DeepGEMM output: {out_deepgemm[0, 0:5]}")
    print(f"Triton output: {out_triton[0, 0:5]}")
    print(f"Mean absolute difference (Torch-DeepGEMM): {diff_torch_deepgemm}")
    print(f"Mean absolute difference (Torch-Triton): {diff_torch_triton}")
    print(f"Mean absolute difference (DeepGEMM-Triton): {diff_deepgemm_triton}")

    deepgemm_torch_diff = calc_diff(out_deepgemm, out_torch)
    triton_torch_diff = calc_diff(out_triton, out_torch)
    deepgemm_triton_diff = calc_diff(out_deepgemm, out_triton)

    DIFF_THRESHOLD = 0.001
    all_match = (
        deepgemm_torch_diff < DIFF_THRESHOLD
        and triton_torch_diff < DIFF_THRESHOLD
        and deepgemm_triton_diff < DIFF_THRESHOLD
    )
    if all_match:
        print("✅ All implementations match\n")
    else:
        print("❌ Some implementations differ:")
        print(
            f"  - Torch vs DeepGEMM: {'✅' if deepgemm_torch_diff < DIFF_THRESHOLD else '❌'}"
            f"  - Torch vs Triton: {'✅' if triton_torch_diff < DIFF_THRESHOLD else '❌'}"
            f"  - DeepGEMM vs Triton: {'✅' if deepgemm_triton_diff < DIFF_THRESHOLD else '❌'}"
        )


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
    batch_sizes = [2048, 4096]
    group_sizes = [4, 8]
    for n, k in weight_shapes:
        for m in batch_sizes:
            for num_groups in group_sizes:
                configs.append((m, n, k, num_groups, tp_size))

    return configs


def get_benchmark(tp_size):
    all_configs = create_benchmark_configs(tp_size)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k", "num_groups", "tp_size"],
            x_vals=[config for config in all_configs],
            line_arg="provider",
            line_vals=["deepgemm", "triton"],
            line_names=["DeepGEMM", "Triton"],
            styles=[("blue", "-"), ("red", "-")],
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
        x_fp8_grouped, y_fp8_grouped, x_fp8_flat, y_fp8_flat, out, out_torch = (
            construct_grouped_and_flat_fp8(x, y, num_groups, is_masked=False)
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
                lambda: fp8_gemm_group_deepgemm(
                    x_fp8_grouped,
                    y_fp8_grouped,
                    out,
                    m_indices,
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            # Prepare inputs for Triton
            # We did it outside of the lambda function to make it fair comparison like deepgemm
            a, a_scale = x_fp8_flat
            b, b_scale = y_fp8_flat
            b = b.T.contiguous()
            # Ensure scales are in the right format and contiguous
            a_scale, b_scale = a_scale.contiguous(), b_scale.contiguous()
            M, _ = a.shape
            _, N = b.shape
            c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fp8_gemm_group_triton(
                    (a, a_scale),
                    (b, b_scale),
                    c,
                    num_groups,
                ),
                quantiles=quantiles,
            )

        # Calculate TFLOPS
        flops = 2 * m * n * k  # multiply-adds
        tflops = flops / (ms * 1e-3) / 1e12

        print(f"Time: {ms*1000:.2f} ms, TFLOPS: {tflops:.2f}")
        return ms * 1000, max_ms * 1000, min_ms * 1000  # convert to ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/fp8_group_gemm/",
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
    if args.run_correctness:
        print("Running correctness tests...")
        calculate_diff(8192, 7168, 4096, 4)
        calculate_diff(8192, 2048, 7168, 4)
        calculate_diff(4096, 7168, 4096, 8)
        calculate_diff(4096, 2048, 7168, 8)
        calculate_diff(4096, 576, 7168, 8)

    # Get the benchmark function with the specified tp_size
    benchmark = get_benchmark(args.tp_size)

    print(f"Running performance benchmark for TP size = {args.tp_size}...")
    benchmark.run(print_data=True, save_path=args.save_path)

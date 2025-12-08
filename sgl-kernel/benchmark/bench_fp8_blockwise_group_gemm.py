import argparse
import os

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)
import random
from dataclasses import dataclass
from typing import List, Tuple

import deep_gemm
import torch
from sgl_kernel import fp8_blockwise_scaled_grouped_mm


def get_m_alignment_for_contiguous_layout():
    return 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


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


def construct_contiguous_grouped(
    num_groups: int, expected_m_per_group: int, k: int, n: int
) -> Tuple[
    int,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    alignment = get_m_alignment_for_contiguous_layout()
    group_ms = [int(expected_m_per_group) for _ in range(num_groups)]
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])

    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        start = aligned_end

    assert m % 4 == 0, f"TMA alignment error: {m}"
    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, ceil_div(n, 128), k // 128), device="cuda", dtype=torch.float
        ),
    )
    for i in range(num_groups):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    return m, x_fp8, y_fp8, m_indices, out


def bench_deepgemm(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, int]:
    # construct tensors
    m, x_fp8, y_fp8, m_indices, out = construct_contiguous_grouped(
        num_groups, expected_m_per_group, k, n
    )

    def run_deepgemm():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(x_fp8, y_fp8, out, m_indices)

    # warmup
    for _ in range(num_warmup):
        run_deepgemm()
    torch.cuda.synchronize()

    # run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies: list[float] = []
    start_event.record()
    for _ in range(num_run):
        run_deepgemm()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    avg = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return avg, m


def bench_cutlass(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, int]:
    device = "cuda"
    alignment = 16
    n_g = ceil_div(n, alignment) * alignment
    k_g = ceil_div(k, alignment) * alignment
    out_dtype = torch.bfloat16

    expert_offsets = torch.zeros((num_groups + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_groups, 3), device=device, dtype=torch.int32)
    layout_sfa = torch.zeros((num_groups, 5), device=device, dtype=torch.int32)
    layout_sfb = torch.zeros((num_groups, 5), device=device, dtype=torch.int32)

    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []

    # TODO(@TianQiLin666666): Unique group_ms in all bench function
    group_ms = [
        alignment * ceil_div(int(expected_m_per_group), alignment)
        for _ in range(num_groups)
    ]
    for g in range(num_groups):
        m_g = group_ms[g]
        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][:] = torch.tensor([m_g, n_g, k_g], device=device)

        a_g, a_scale = per_token_cast_to_fp8(torch.randn((m_g, k_g), device=device))
        b_g, b_scale = per_block_cast_to_fp8(torch.randn((n_g, k_g), device=device).t())
        a_tensors.append(a_g)
        b_tensors.append(b_g)
        a_scales_tensors.append(a_scale)
        b_scales_tensors.append(b_scale)

    a_stack = torch.empty(
        (expert_offsets[-1], k_g), device=device, dtype=torch.float8_e4m3fn
    )
    b_stack = torch.empty(
        (num_groups, n_g, k_g), device=device, dtype=torch.float8_e4m3fn
    )

    for g in range(num_groups):
        a_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_tensors[g]
        b_stack[g] = b_tensors[g].t()
    b_stack = b_stack.transpose(1, 2)

    a_scale_stack = torch.empty(
        (expert_offsets[-1], k_g // 128), device=device, dtype=torch.float32
    )
    b_scale_stack = torch.empty(
        (num_groups, n_g // 128, k_g // 128), device=device, dtype=torch.float32
    )

    for g in range(num_groups):
        a_scale_stack[expert_offsets[g] : expert_offsets[g + 1]] = a_scales_tensors[g]
        b_scale_stack[g] = b_scales_tensors[g].t()
    b_scale_stack = b_scale_stack.transpose(1, 2)

    c_out = torch.empty((expert_offsets[-1], n_g), device=device, dtype=out_dtype)
    a_strides = torch.full(
        (num_groups,), a_stack.stride(0), device=device, dtype=torch.int64
    )
    c_strides = torch.full(
        (num_groups,), c_out.stride(0), device=device, dtype=torch.int64
    )
    workspace = torch.empty((1024 * 1024 * 1024), device=device, dtype=torch.uint8)
    a_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    b_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    out_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    a_scales_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)
    b_scales_ptrs = torch.empty((num_groups,), device=device, dtype=torch.int64)

    def run_cutlass():
        fp8_blockwise_scaled_grouped_mm(
            c_out,
            a_ptrs,
            b_ptrs,
            out_ptrs,
            a_scales_ptrs,
            b_scales_ptrs,
            a_stack,
            b_stack,
            a_scale_stack,
            b_scale_stack,
            a_strides,
            a_strides,
            c_strides,
            layout_sfa,
            layout_sfb,
            problem_sizes,
            expert_offsets[:-1],
            workspace,
        )

    # warmup
    for _ in range(num_warmup):
        run_cutlass()
    torch.cuda.synchronize()

    # run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        run_cutlass()
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    avg = start_event.elapsed_time(end_event) / num_run * 1000  # us

    return avg, expert_offsets[-1]


def bench_sglang_triton(
    expected_m_per_group: int,
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, int]:
    pass


benchmark_kernels = {
    "deepgemm": bench_deepgemm,
    "cutlass": bench_cutlass,
    # "triton": bench_sglang_triton,
}


@dataclass
class ShapeArg:
    expected_m_per_group: int
    n: int
    k: int
    num_groups: int


def benchmark_one_shape(
    shape_args: List[ShapeArg],
    num_warmup: int,
    num_run: int,
):
    for shape in shape_args:
        print(
            f"\nBenchmark: expected_m_per_group={shape.expected_m_per_group}, n={shape.n}, k={shape.k}, num_groups={shape.num_groups}"
        )
        for kernel_name, kernel_func in benchmark_kernels.items():
            average_time, m = kernel_func(
                shape.expected_m_per_group,
                shape.n,
                shape.k,
                shape.num_groups,
                num_warmup,
                num_run,
            )
            print(f"{kernel_name}: {average_time} us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-run", type=int, default=10)

    # CI environment uses simplified parameters
    if IS_CI:
        shape_args = [
            # Only test one simple shape in CI
            ShapeArg(expected_m_per_group=128, n=512, k=7168, num_groups=256),
        ]
    else:
        shape_args = [
            # Prefill, DeepSeek-R1, gateup, chunk_size = 4096, TP = 8
            ShapeArg(expected_m_per_group=128, n=512, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 8192, TP = 8
            ShapeArg(expected_m_per_group=256, n=512, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 8192, TP = 16
            ShapeArg(expected_m_per_group=256, n=256, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 16384, TP = 16
            ShapeArg(expected_m_per_group=512, n=256, k=7168, num_groups=256),
            # Decode, DeepSeek-R1, gateup, bs = 32, TP = 8
            ShapeArg(expected_m_per_group=1, n=512, k=7168, num_groups=256),
            # Decode, DeepSeek-R1, gateup, bs = 64, TP = 16
            ShapeArg(expected_m_per_group=2, n=256, k=7168, num_groups=256),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 8192, EP = 8
            ShapeArg(expected_m_per_group=256, n=4096, k=7168, num_groups=32),
            # Prefill, DeepSeek-R1, gateup, chunk_size = 16384, EP = 16
            ShapeArg(expected_m_per_group=512, n=4096, k=7168, num_groups=16),
            # Decode, DeepSeek-R1, gateup, bs = 128, EP = 8
            ShapeArg(expected_m_per_group=4, n=4096, k=7168, num_groups=32),
            # Decode, DeepSeek-R1, gateup, bs = 256, EP = 16
            ShapeArg(expected_m_per_group=8, n=4096, k=7168, num_groups=16),
            # Prefill, Qwen3-235B-A22B-FP8, gateup, chunk_size = 16384, TP = 4
            ShapeArg(expected_m_per_group=1024, n=768, k=4096, num_groups=128),
            # Prefill, Qwen3-235B-A22B-FP8, down, chunk_size = 16384, TP = 4
            ShapeArg(expected_m_per_group=1024, n=4096, k=384, num_groups=128),
            # Decode, Qwen3-235B-A22B-FP8, gateup, bs = 256, TP = 4
            ShapeArg(expected_m_per_group=16, n=768, k=4096, num_groups=128),
            # Decode, Qwen3-235B-A22B-FP8, down, bs = 256, TP = 4
            ShapeArg(expected_m_per_group=16, n=4096, k=384, num_groups=128),
        ]
    args = parser.parse_args()
    benchmark_one_shape(shape_args, args.num_warmup, args.num_run)


if __name__ == "__main__":
    main()

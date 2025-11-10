import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from sgl_kernel import (
    es_fp8_blockwise_scaled_grouped_mm,
    fp8_blockwise_scaled_grouped_mm,
)

random.seed(28)


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


def create_unbalanced_expert_token_distribution(max_num_experts):
    ratios = [random.random() for _ in range(max_num_experts)]

    def convert_to_tokens(ratio: float):
        if ratio <= 0.7:
            return random.randint(1, 32)
        elif ratio > 0.7 and ratio <= 0.85:
            return random.randint(32, 64)
        elif ratio > 0.85 and ratio <= 0.95:
            return random.randint(64, 128)
        elif ratio > 0.95:
            return random.randint(128, 1024)
        else:
            return 128

    group_ms = [convert_to_tokens(ratio) for ratio in ratios]
    return group_ms


group_ms = create_unbalanced_expert_token_distribution(8192)
# group_ms = [128 for _ in range(8192)]
# group_ms = [128 if i % 2 == 0 else 64 for i in range(8192)]


def bench_es(
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, int]:
    device = "cuda"
    alignment = 128
    n_g = ceil_div(n, alignment) * alignment
    k_g = ceil_div(k, alignment) * alignment
    out_dtype = torch.bfloat16

    expert_offsets = torch.zeros((num_groups + 1), device=device, dtype=torch.int32)
    problem_sizes = torch.zeros((num_groups, 3), device=device, dtype=torch.int32)

    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    if False:
        print("Token Distributtion: ", group_ms[0:num_groups])
        print("Token Count: ", sum(group_ms[0:num_groups]))
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
    d_strides = torch.full(
        (num_groups,), c_out.stride(0), device=device, dtype=torch.int64
    )
    workspace = torch.empty((1024 * 1024 * 1024), device=device, dtype=torch.uint8)

    def run_cutlass():
        es_fp8_blockwise_scaled_grouped_mm(
            c_out,
            a_stack,
            b_stack,
            a_scale_stack,
            b_scale_stack,
            a_strides,
            a_strides,
            d_strides,
            problem_sizes,
            expert_offsets[:-1],
            workspace,
        )

    run_cutlass()
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


def bench_sgl(
    n: int,
    k: int,
    num_groups: int,
    num_warmup: int,
    num_run: int,
) -> Tuple[float, int]:
    device = "cuda"
    alignment = 128
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


benchmark_kernels = {"es": bench_es, "sgl-kernel": bench_sgl}


@dataclass
class ShapeArg:
    n: int
    k: int
    num_groups: int


def benchmark_one_shape(
    shape_args: List[ShapeArg],
    num_warmup: int,
    num_run: int,
):
    for shape in shape_args:
        print(f"\nBenchmark: n={shape.n}, k={shape.k}, num_groups={shape.num_groups}")
        for kernel_name, kernel_func in benchmark_kernels.items():
            average_time, m = kernel_func(
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
    parser.add_argument("--num-run", type=int, default=20)
    shape_args = [
        # Prefill, DeepSeek-R1, gateup, chunk_size = 4096, TP = 8
        ShapeArg(n=512, k=7168, num_groups=256),
        # Prefill, DeepSeek-R1, down, chunk_size = 4096, TP = 8
        ShapeArg(n=7168, k=256, num_groups=256),
        # Prefill, Qwen3-235B-A22B-FP8, gateup, TP = 4
        ShapeArg(n=768, k=4096, num_groups=128),
        # Prefill, Qwen3-235B-A22B-FP8, down, TP = 4
        ShapeArg(n=4096, k=384, num_groups=128),
        # Decode, DeepSeek-R1, gateup, bs = 128, EP = 8
        ShapeArg(n=4096, k=7168, num_groups=32),
        # Decode, DeepSeek-R1, gateup, bs = 256, EP = 16
        ShapeArg(n=4096, k=7168, num_groups=16),
    ]
    args = parser.parse_args()
    benchmark_one_shape(shape_args, args.num_warmup, args.num_run)


if __name__ == "__main__":
    main()

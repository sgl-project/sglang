import argparse
from typing import Tuple

import torch
import deep_gemm

FP8_MAX = 448.0


def per_token_cast_to_fp8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert x.shape[1] % 128 == 0

    m, k = x.shape
    view = x.view(m, -1, 128)

    amax = (
        view.abs()
        .float()
        .amax(dim=2)
        .clamp_min(1e-4)
    )

    fp8 = (
        view * (FP8_MAX / amax.unsqueeze(2))
    ).to(torch.float8_e4m3fn).view(m, k)

    scale = amax / FP8_MAX
    return fp8, scale


def per_block_cast_to_fp8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2

    m, k = x.shape
    padded_m = deep_gemm.ceil_div(m, 128) * 128
    padded_k = deep_gemm.ceil_div(k, 128) * 128

    padded = torch.zeros(
        (padded_m, padded_k),
        dtype=x.dtype,
        device=x.device,
    )
    padded[:m, :k] = x

    view = padded.view(
        padded_m // 128,
        128,
        padded_k // 128,
        128,
    )

    amax = (
        view.abs()
        .float()
        .amax(dim=(1, 3), keepdim=True)
        .clamp_min(1e-4)
    )

    fp8 = (
        view * (FP8_MAX / amax)
    ).to(torch.float8_e4m3fn)

    fp8 = fp8.view(padded_m, padded_k)[:m, :k].contiguous()
    scale = (amax / FP8_MAX).view(
        padded_m // 128,
        padded_k // 128,
    )

    return fp8, scale


def construct_grouped(
    groups: int,
    max_m: int,
    n: int,
    k: int,
):
    x = torch.randn(
        (groups, max_m, k),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (groups, n, k),
        device="cuda",
        dtype=torch.bfloat16,
    )

    out = torch.empty(
        (groups, max_m, n),
        device="cuda",
        dtype=torch.bfloat16,
    )

    x_data = torch.empty_like(
        x,
        dtype=torch.float8_e4m3fn,
    )
    x_scale = torch.empty(
        (groups, max_m, k // 128),
        device="cuda",
        dtype=torch.float32,
    )

    weight_data = torch.empty_like(
        weight,
        dtype=torch.float8_e4m3fn,
    )
    weight_scale = torch.empty(
        (groups, deep_gemm.ceil_div(n, 128), k // 128),
        device="cuda",
        dtype=torch.float32,
    )

    for group in range(groups):
        x_data[group], x_scale[group] = per_token_cast_to_fp8(
            x[group]
        )
        weight_data[group], weight_scale[group] = per_block_cast_to_fp8(
            weight[group]
        )

    x_scale = deep_gemm.get_col_major_tma_aligned_tensor(x_scale)

    return (
        (x_data, x_scale),
        (weight_data, weight_scale),
        out,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--max-m", type=int, default=256)
    parser.add_argument("--valid-m", type=int, default=64)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if args.valid_m > args.max_m:
        raise ValueError("--valid-m must be <= --max-m")
    if args.max_m % 4 != 0:
        raise ValueError("--max-m must be divisible by 4")
    if args.k % 128 != 0:
        raise ValueError("--k must be divisible by 128")

    print("GPU:", torch.cuda.get_device_name(0))
    print("DeepGEMM:", deep_gemm.__file__)
    print(
        f"groups={args.groups}, max_m={args.max_m}, "
        f"valid_m={args.valid_m}, N={args.n}, K={args.k}"
    )

    lhs, rhs, out = construct_grouped(
        args.groups,
        args.max_m,
        args.n,
        args.k,
    )

    masked_m = torch.full(
        (args.groups,),
        args.valid_m,
        device="cuda",
        dtype=torch.int32,
    )

    def run() -> None:
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            args.valid_m,
        )

    print("Compiling/warming up DeepGEMM...")
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(
        f"deepgemm_masked/"
        f"g{args.groups}_m{args.valid_m}_n{args.n}_k{args.k}"
    )

    start.record()
    for _ in range(args.iterations):
        run()
    end.record()

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    total_ms = start.elapsed_time(end)
    average_us = total_ms * 1000.0 / args.iterations

    flops = (
        2
        * args.groups
        * args.valid_m
        * args.n
        * args.k
    )
    tflops = flops / (average_us * 1e-6) / 1e12

    print(f"Average: {average_us:.2f} us")
    print(f"Throughput: {tflops:.2f} TFLOPS")
    print("Output checksum:", out.float().mean().item())


if __name__ == "__main__":
    main()

import itertools
from typing import Optional

import torch
import triton.testing
from sgl_kernel import (
    sgl_per_token_group_quant_fp8,
    sgl_silu_and_mul_per_token_group_quant_fp8,
    silu_and_mul,
)

num_tokens_range = [2**i for i in range(2, 15)]
hidden_dim_range = [2**i for i in range(7, 14)]
configs = list(itertools.product(num_tokens_range, hidden_dim_range))
fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
    if x.shape[0] > 0:
        sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max)

    return x_q, x_s


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert (
        input.shape[:-1] == output.shape[:-1]
    ), f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert (
        input.shape[-1] == 2 * output.shape[-1]
    ), f"{input.shape[-1]} != {2 * output.shape[-1]}"


def sglang_silu_and_mul_per_token_group_quant_fp8(
    input: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
):
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    x_q = torch.empty(
        input.shape[:-1] + (input.shape[-1] // 2,),
        device=input.device,
        dtype=torch.float8_e4m3fn,
    )
    _check_shape(input, x_q)
    assert (
        x_q.shape[-1] % group_size == 0
    ), "the last dimension of `x_q` cannot be divisible by `group_size`"
    assert x_q.is_contiguous(), "`x_q` is not contiguous"
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x_q.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x_q.shape[:-2] + (x_q.shape[-1] // group_size, aligned_size),
                device=x_q.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x_q.shape[-2], :]
        else:
            x_s = torch.empty(
                (x_q.shape[-1] // group_size,) + x_q.shape[:-1],
                device=x_q.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x_q.shape[:-1] + (x_q.shape[-1] // group_size,),
            device=x_q.device,
            dtype=torch.float32,
        )
    if x_q.shape[0] > 0:
        sgl_silu_and_mul_per_token_group_quant_fp8(
            input, x_q, x_s, group_size, eps, fp8_min, fp8_max
        )

    return x_q, x_s


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "hidden_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["fused", "split", "fused_cudagraph", "split_cudagraph"],
        line_names=["fused", "split", "fused_cudagraph", "split_cudagraph"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("orange", "-")],
        ylabel="us",
        plot_name="silu_and_mul and group quant fp8 operator performance for fuse and split",
        args={},
    )
)
def benchmark(num_tokens, hidden_dim, provider):
    quantiles = [0.5, 0.2, 0.8]
    device = "cuda"
    gateup_output = torch.randn(
        num_tokens, hidden_dim * 2, device=device, dtype=torch.bfloat16
    )
    down_input = torch.empty(
        num_tokens, hidden_dim, device=device, dtype=torch.bfloat16
    )
    scale_block_size = 128
    if provider == "fused":

        def fused(gateup_output, scale_block_size):
            return sglang_silu_and_mul_per_token_group_quant_fp8(
                gateup_output, scale_block_size
            )

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused(gateup_output, scale_block_size),
            quantiles=quantiles,
        )
    elif provider == "fused_cudagraph":

        def fused(gateup_output, scale_block_size):
            return sglang_silu_and_mul_per_token_group_quant_fp8(
                gateup_output, scale_block_size
            )

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: fused(gateup_output, scale_block_size),
            quantiles=quantiles,
        )
    elif provider == "split":

        def split(gateup_output, down_input, scale_block_size):
            silu_and_mul(gateup_output, down_input)
            return sglang_per_token_group_quant_fp8(down_input, scale_block_size)

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: split(gateup_output, down_input, scale_block_size),
            quantiles=quantiles,
        )
    elif provider == "split_cudagraph":

        def split(gateup_output, down_input, scale_block_size):
            silu_and_mul(gateup_output, down_input)
            return sglang_per_token_group_quant_fp8(down_input, scale_block_size)

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: split(gateup_output, down_input, scale_block_size),
            quantiles=quantiles,
        )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)

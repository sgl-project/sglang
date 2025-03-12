import itertools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.testing
from sgl_kernel import sgl_per_token_quant_fp8
from vllm import _custom_ops as ops

from sglang.srt.utils import is_hip

is_hip_ = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn

@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    fp8_min,
    fp8_max,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / fp8_max
    x_q = tl.clamp(x / scale_x, fp8_min, fp8_max).to(xq_ptr.dtype.element_ty)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def sgl_triton_per_token_quant_fp8(x) -> Tuple[torch.Tensor, torch.Tensor]:
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=fp8_type_)
    scales = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    finfo = torch.finfo(fp8_type_)
    fp8_max = finfo.max

    if is_hip_:
        fp8_max = 224.0

    fp8_min = -fp8_max
    assert x.is_contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
    )

    return x_q, scales


def sglang_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    sgl_per_token_quant_fp8(input, output, scale)

    return output, scale


def calculate_diff(batch_size: int, seq_len: int):
    """Calculate difference between SGLang triton and cuda implementations."""
    device = torch.device("cuda")
    x = torch.rand((batch_size, seq_len), dtype=torch.float16, device=device)

    sgl_triton_out, sgl_triton_scale = sgl_triton_per_token_quant_fp8(x)
    sglang_out, sglang_scale = sglang_per_token_quant_fp8(x)

    scale_diff = torch.abs(sgl_triton_scale - sglang_scale).mean().item()
    output_diff = torch.abs(sgl_triton_out.float() - sglang_out.float()).mean().item()

    if torch.allclose(
        sgl_triton_out.to(torch.float32), sglang_out.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(sgl_triton_scale, sglang_scale, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [16, 32, 64, 128]
seq_len_range = [64, 128, 256, 512, 1024, 2048, 4096]

configs = list(itertools.product(batch_size_range, seq_len_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_triton", "sglang"],
        line_names=["SGL Triton", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-dynamic-quant-fp8-performance",
        args={},
    )
)
def benchmark_quantization(batch_size, seq_len, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(batch_size * seq_len, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sgl_triton":
        fn = lambda: sgl_triton_per_token_quant_fp8(x.clone())
    elif provider == "sglang":
        fn = lambda: sglang_per_token_quant_fp8(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=4, seq_len=4096)
    benchmark_quantization.run(print_data=True)

import itertools

import torch
import triton
from sgl_kernel import sgl_per_token_group_quant_8bit

from sglang.srt.layers.quantization.int8_kernel import per_token_group_quant_int8
from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8


def sglang_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
    eps: float = 1e-10,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if dtype == torch.int8:
        iinfo = torch.iinfo(dtype)
        max_8bit = iinfo.max
        min_8bit = iinfo.min
    else:
        f8_info = torch.finfo(dtype)
        max_8bit = f8_info.max
        min_8bit = f8_info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    sgl_per_token_group_quant_8bit(x, x_q, x_s, group_size, eps, min_8bit, max_8bit)

    return x_q, x_s


def calculate_diff(batch_size, seq_len, group_size, dtype):
    device = torch.device("cuda")
    hidden_dim = group_size * 2

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)

    if dtype == torch.int8:
        x_q_triton, x_s_triton = per_token_group_quant_int8(x.clone(), group_size)
    else:
        x_q_triton, x_s_triton = per_token_group_quant_fp8(x.clone(), group_size)
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(x.clone(), group_size, dtype=dtype)

    if torch.allclose(
        x_q_triton.to(torch.float32), x_q_sglang.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(x_s_triton, x_s_sglang, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1, 2, 4, 8, 16, 32, 64]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
group_size_range = [128]  # For DeepSeek V3/R1
dtype_range = [torch.int8, torch.float8_e4m3fn]

configs = list(itertools.product(batch_size_range, seq_len_range, group_size_range, dtype_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size", "dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "sglang"],
        line_names=["Triton", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-group-quant-8bit-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, dtype, provider):
    device = torch.device("cuda")
    hidden_dim = 7168

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        if dtype == torch.int8:
            fn = lambda: per_token_group_quant_int8(x.clone(), group_size)
        else:
            fn = lambda: per_token_group_quant_fp8(x.clone(), group_size)
    elif provider == "sglang":
        fn = lambda: sglang_per_token_group_quant_8bit(x.clone(), group_size, dtype=dtype)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":

    calculate_diff(batch_size=4, seq_len=128, group_size=64, dtype=torch.int8)
    calculate_diff(batch_size=4, seq_len=128, group_size=64, dtype=torch.float8_e4m3fn)

    benchmark.run(print_data=True)

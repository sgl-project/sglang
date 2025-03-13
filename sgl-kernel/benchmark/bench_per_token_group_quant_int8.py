import itertools

import torch
import triton
from sgl_kernel import sgl_per_token_group_quant_int8

from sglang.srt.layers.quantization.int8_kernel import per_token_group_quant_int8


def sglang_per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    sgl_per_token_group_quant_int8(x, x_q, x_s, group_size, eps, int8_min, int8_max)

    return x_q, x_s


def calculate_diff(batch_size, seq_len, group_size):
    dtype = torch.float16
    device = torch.device("cuda")
    hidden_dim = group_size * 2

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    x_q_triton, x_s_triton = per_token_group_quant_int8(x.clone(), group_size)
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_int8(x.clone(), group_size)

    if torch.allclose(
        x_q_triton.to(torch.float32), x_q_sglang.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(x_s_triton, x_s_sglang, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1, 2, 4, 8, 16, 32, 64]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
group_size_range = [128]  # For DeepSeek V3/R1

configs = list(itertools.product(batch_size_range, seq_len_range, group_size_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "sglang"],
        line_names=["Triton", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-group-quant-fp8-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    hidden_dim = 7168

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: per_token_group_quant_int8(x.clone(), group_size)
    elif provider == "sglang":
        fn = lambda: sglang_per_token_group_quant_int8(x.clone(), group_size)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":

    calculate_diff(batch_size=4, seq_len=128, group_size=64)

    benchmark.run(print_data=True)

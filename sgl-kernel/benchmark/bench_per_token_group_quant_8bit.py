import itertools
from typing import Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel import sgl_per_token_group_quant_fp8, sgl_per_token_group_quant_int8

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


@triton.jit
def _per_token_group_quant_8bit(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Collums of input
    N,
    # Avoid to divide zero
    eps,
    # Information for 8bit data type (int8 or fp8_type_)
    max_8bit,
    min_8bit,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.
    This function converts the tensor values into 8bit values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / max_8bit
    y_q = tl.clamp(y / y_s, min_8bit, max_8bit).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def triton_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn` is supported for now.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if dst_dtype == torch.int8:
        iinfo = torch.iinfo(dst_dtype)
        max_8bit = iinfo.max
        min_8bit = iinfo.min
    else:
        finfo = torch.finfo(dst_dtype)
        max_8bit = finfo.max
        min_8bit = finfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dst_dtype)
    M = x.numel() // group_size
    N = group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_8bit[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        max_8bit,
        min_8bit,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


def sglang_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=dst_dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    if dst_dtype == torch.int8:
        iinfo = torch.iinfo(dst_dtype)
        int8_max = iinfo.max
        int8_min = iinfo.min
        sgl_per_token_group_quant_int8(x, x_q, x_s, group_size, eps, int8_min, int8_max)
    else:
        f8_info = torch.finfo(dst_dtype)
        fp8_max = f8_info.max
        fp8_min = f8_info.min
        sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max)

    return x_q, x_s


def calculate_diff(batch_size, seq_len, group_size, dst_dtype):
    device = torch.device("cuda")
    hidden_dim = 7168

    x = torch.randn(
        batch_size * seq_len, hidden_dim, device=device, dtype=torch.float16
    )

    x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(
        x.clone(), group_size, dst_dtype
    )
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(
        x.clone(), group_size, dst_dtype
    )

    if torch.allclose(
        x_q_triton.to(torch.float32), x_q_sglang.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(x_s_triton, x_s_sglang, rtol=1e-3, atol=1e-5):
        print(f"✅ {dst_dtype} implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1, 2, 4, 8, 16, 32, 64]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
group_size_range = [128]  # For DeepSeek V3/R1
dst_dtype_range = [torch.int8, fp8_type_]

configs = list(
    itertools.product(
        batch_size_range, seq_len_range, group_size_range, dst_dtype_range
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size", "dst_dtype"],
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
def benchmark(batch_size, seq_len, group_size, dst_dtype, provider):
    device = torch.device("cuda")
    hidden_dim = 7168

    x = torch.randn(
        batch_size * seq_len, hidden_dim, device=device, dtype=torch.float16
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: triton_per_token_group_quant_8bit(x.clone(), group_size, dst_dtype)
    elif provider == "sglang":
        fn = lambda: sglang_per_token_group_quant_8bit(x.clone(), group_size, dst_dtype)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":

    calculate_diff(batch_size=4, seq_len=128, group_size=64, dst_dtype=torch.int8)
    calculate_diff(batch_size=4, seq_len=128, group_size=64, dst_dtype=fp8_type_)

    benchmark.run(print_data=True)

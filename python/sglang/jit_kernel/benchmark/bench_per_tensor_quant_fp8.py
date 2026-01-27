from typing import Optional, Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8

try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    ops = None
    VLLM_AVAILABLE = False

try:
    from sglang.srt.utils import is_hip

    _is_hip = is_hip()
except ImportError:
    _is_hip = False

IS_CI = is_in_ci()

fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


def vllm_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not VLLM_AVAILABLE:
        return sglang_scaled_fp8_quant(input, scale)
    return ops.scaled_fp8_quant(input, scale)


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        is_static = False
    per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


def calculate_diff(batch_size: int, seq_len: int):
    device = torch.device("cuda")
    x = torch.rand((batch_size, seq_len), dtype=torch.bfloat16, device=device)

    if not VLLM_AVAILABLE:
        print("vLLM not available, skipping comparison")
        return

    vllm_out, vllm_scale = vllm_scaled_fp8_quant(x)
    sglang_out, sglang_scale = sglang_scaled_fp8_quant(x)

    vllm_out = vllm_out.to(torch.float32)
    sglang_out = sglang_out.to(torch.float32)

    triton.testing.assert_close(vllm_out, sglang_out, rtol=1e-3, atol=1e-3)
    triton.testing.assert_close(vllm_scale, sglang_scale, rtol=1e-3, atol=1e-3)


if IS_CI:
    element_range = [16384]
else:
    element_range = [2**n for n in range(10, 20)]


if VLLM_AVAILABLE:
    line_vals = ["vllm", "sglang"]
    line_names = ["VLLM", "SGL Kernel"]
    styles = [("blue", "-"), ("green", "-")]
else:
    line_vals = ["sglang"]
    line_names = ["SGL Kernel"]
    styles = [("green", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["element_count"],
        x_vals=element_range,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="per-tensor-quant-fp8-performance",
        args={},
    )
)
def benchmark(element_count, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(element_count, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        fn = lambda: vllm_scaled_fp8_quant(x.clone())
    elif provider == "sglang":
        fn = lambda: sglang_scaled_fp8_quant(x.clone())
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=4, seq_len=4096)
    benchmark.run(print_data=True)

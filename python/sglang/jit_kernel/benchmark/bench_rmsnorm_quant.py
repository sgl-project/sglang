import itertools

import torch
import triton
import triton.testing
from flashinfer import rmsnorm_quant as fi_rmsnorm_quant
from sgl_kernel import rms_norm_static_fp8_quant

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)

# from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm


DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn
# maximum value for e4m3fn for clamping in kernel
FP8_E4M3_MAX = 448.0
# FP8 is low precision, so the tolerance needs to be higher
TOLERANCE = {"atol": 1.5e-1, "rtol": 1.5e-1}
FP_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}


def scaled_fp8_conversion_ref(
    val: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """Helper function matching the scaled_fp8_conversion device function."""
    quant_scale = 1.0 / scale

    x = val * quant_scale

    r = torch.clamp(x, min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)

    if r.dtype != fp8_dtype:
        return r.to(fp8_dtype)
    return r


def sglang_aot_rmsnorm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    rms_norm_static_fp8_quant(out, input, weight, scale)


# def sglang_jit_rmsnorm(
#     input: torch.Tensor,
#     weight: torch.Tensor,
# ) -> None:
#     jit_rmsnorm(input, weight, output=input)


def flashinfer_rmsnorm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    fi_rmsnorm_quant(out, input, weight, scale)


@torch.compile()
def torch_impl_rmsnorm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    mean = input.float().pow(2).mean(dim=-1, keepdim=True)
    norm = (mean + eps).rsqrt()
    out.copy_(
        scaled_fp8_conversion_ref(
            (input.float() * norm * weight.float()), scale, FP8_DTYPE
        )
    )


BS_LIST = get_benchmark_range(
    full_range=[2**n for n in range(0, 14)],
    ci_range=[16],
)
HIDDEN_SIZE_LIST = get_benchmark_range(
    full_range=[1536, 3072, 4096, 5120, 8192],
    ci_range=[512, 2048],
)

LINE_VALS = ["aot", "flashinfer", "torch"]
LINE_NAMES = ["SGL AOT Kernel", "FlashInfer", "PyTorch"]
STYLES = [("orange", "-"), ("green", "-."), ("red", ":")]

configs = list(itertools.product(HIDDEN_SIZE_LIST, BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="rmsnorm-performance",
        args={},
    )
)
def benchmark(hidden_size: int, batch_size: int, provider: str):
    input = torch.randn(
        (batch_size, hidden_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    weight = torch.randn(hidden_size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    scale = torch.tensor([4.0], dtype=torch.float32, device=DEFAULT_DEVICE)
    out = torch.empty((batch_size, hidden_size), dtype=FP8_DTYPE, device=DEFAULT_DEVICE)
    FN_MAP = {
        "aot": sglang_aot_rmsnorm_quant,
        # "jit": sglang_jit_rmsnorm,
        "flashinfer": flashinfer_rmsnorm_quant,
        "torch": torch_impl_rmsnorm_quant,
    }
    fn = lambda: FN_MAP[provider](input, weight, scale, out.clone())
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)

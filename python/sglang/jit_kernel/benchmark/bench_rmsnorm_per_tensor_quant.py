import itertools

import torch
import triton
import triton.testing
from flashinfer import rmsnorm_quant as fi_rmsnorm_quant

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
from sglang.jit_kernel.norm import (
    rmsnorm_per_tensor_quant as jit_rmsnorm_per_tensor_quant,
)
from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8 as jit_quant

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


def sglang_jit_rmsnorm_per_tensor_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    jit_rmsnorm_per_tensor_quant(out, input, weight, scale)


def sglang_unfused_jit(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    temp = torch.empty_like(input)
    jit_rmsnorm(input, weight, out=temp)
    jit_quant(temp, out, scale, is_static=True)


def flashinfer_rmsnorm_per_tensor_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    fi_rmsnorm_quant(out, input, weight, scale)


@torch.compile()
def torch_impl_rmsnorm_per_tensor_quant(
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

LINE_VALS = [
    "fused_jit",
    "fused_flashinfer",
    "unfused_jit",
    "unfused_torch",
]
LINE_NAMES = [
    "SGL JIT Fused",
    "FlashInfer Fused",
    "SGL JIT Unfused",
    "PyTorch Unfused",
]
STYLES = [
    ("blue", "--"),
    ("purple", "-."),
    ("green", "-."),
    ("red", ":"),
]

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
        "fused_jit": sglang_jit_rmsnorm_per_tensor_quant,
        "fused_flashinfer": flashinfer_rmsnorm_per_tensor_quant,
        "unfused_jit": sglang_unfused_jit,
        "unfused_torch": torch_impl_rmsnorm_per_tensor_quant,
    }
    fn = lambda: FN_MAP[provider](input, weight, scale, out.clone())
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)

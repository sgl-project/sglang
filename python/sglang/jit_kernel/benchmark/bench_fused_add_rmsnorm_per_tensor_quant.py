import itertools

import torch
import triton
import triton.testing
from flashinfer import fused_add_rmsnorm_quant as fi_fused_add_rmsnorm_quant

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.jit_kernel.norm import (
    fused_add_rmsnorm_per_tensor_quant as jit_fused_add_rmsnorm_per_tensor_quant,
)
from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8 as jit_quant

FP8_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = 448.0


def sglang_jit_fused_add_rmsnorm_per_tensor_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    jit_fused_add_rmsnorm_per_tensor_quant(out, input, residual, weight, scale)


def flashinfer_fused_add_rmsnorm_per_tensor_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    fi_fused_add_rmsnorm_quant(out, input, residual, weight, scale)


def sglang_unfused_jit(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    jit_fused_add_rmsnorm(input, residual, weight)
    jit_quant(input, out, scale, is_static=True)


@torch.compile()
def torch_impl_fused_add_rmsnorm_per_tensor_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    residual.add_(input)
    x = residual.float()
    mean = x.pow(2).mean(dim=-1, keepdim=True)
    norm = (mean + eps).rsqrt()
    normed = x * norm * weight.float()
    inv_scale = 1.0 / scale
    out.copy_((normed * inv_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_DTYPE))


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
        plot_name="fused-add-rmsnorm-per-tensor-quant-performance",
        args={},
    )
)
def benchmark(hidden_size: int, batch_size: int, provider: str):
    input = torch.randn(
        (batch_size, hidden_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    residual = torch.randn(
        (batch_size, hidden_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    weight = torch.randn(hidden_size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    scale = torch.tensor([4.0], dtype=torch.float32, device=DEFAULT_DEVICE)
    out = torch.empty((batch_size, hidden_size), dtype=FP8_DTYPE, device=DEFAULT_DEVICE)
    FN_MAP = {
        "fused_jit": sglang_jit_fused_add_rmsnorm_per_tensor_quant,
        "fused_flashinfer": flashinfer_fused_add_rmsnorm_per_tensor_quant,
        "unfused_jit": sglang_unfused_jit,
        "unfused_torch": torch_impl_fused_add_rmsnorm_per_tensor_quant,
    }
    fn = lambda: FN_MAP[provider](
        input.clone(), residual.clone(), weight, scale, out.clone()
    )
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)

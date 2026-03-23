from typing import Tuple

import torch
import triton.testing

from sglang.jit_kernel.benchmark.utils import run_benchmark_no_cudagraph
from sglang.jit_kernel.diffusion.triton.norm import norm_infer
from sglang.jit_kernel.diffusion.triton.scale_shift import (
    fuse_layernorm_scale_shift_gate_select01_kernel,
    fuse_residual_layernorm_scale_shift_gate_select01_kernel,
)
from sglang.utils import is_in_ci

if is_in_ci():
    B_RANGE, S_RANGE, D_RANGE = [1], [128], [3072]
else:
    B_RANGE, S_RANGE, D_RANGE = [1, 2], [128, 512, 2048], [1024, 1536, 3072]

DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-6
LINE_VALS = ["split", "fused"]
LINE_NAMES = ["Triton Norm + Torch Select", "Fused Triton"]
STYLES = [("red", "-"), ("blue", "--")]
CONFIG = [(b, s, d) for b in B_RANGE for s in S_RANGE for d in D_RANGE]


def _make_common_inputs(batch_size: int, seq_len: int, hidden_size: int):
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    index = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int32, device=DEVICE)
    scale0 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    shift0 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    gate0 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    scale1 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    shift1 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    gate1 = torch.randn(batch_size, hidden_size, dtype=DTYPE, device=DEVICE)
    return x, weight, bias, index, scale0, shift0, gate0, scale1, shift1, gate1


def _apply_select01_modulation(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
):
    idx = index.bool().unsqueeze(-1)
    scale = torch.where(idx, scale1.unsqueeze(1), scale0.unsqueeze(1))
    shift = torch.where(idx, shift1.unsqueeze(1), shift0.unsqueeze(1))
    gate = torch.where(idx, gate1.unsqueeze(1), gate0.unsqueeze(1))
    return x * (1 + scale) + shift, gate


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D"],
        x_vals=CONFIG,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="qwen_image_layernorm_scale_shift_gate_select01",
        args={},
    )
)
def bench_layernorm_scale_shift_gate_select01(
    B: int, S: int, D: int, provider: str
) -> Tuple[float, float, float]:
    x, weight, bias, index, scale0, shift0, gate0, scale1, shift1, gate1 = (
        _make_common_inputs(B, S, D)
    )

    if provider == "split":

        def fn():
            normalized = norm_infer(
                x.view(-1, x.shape[-1]),
                weight,
                bias,
                eps=EPS,
                is_rms_norm=False,
            ).view_as(x)
            return _apply_select01_modulation(
                normalized, scale0, shift0, gate0, scale1, shift1, gate1, index
            )

    else:

        def fn():
            return fuse_layernorm_scale_shift_gate_select01_kernel(
                x,
                weight=weight,
                bias=bias,
                scale0=scale0,
                shift0=shift0,
                gate0=gate0,
                scale1=scale1,
                shift1=shift1,
                gate1=gate1,
                index=index,
                eps=EPS,
            )

    return run_benchmark_no_cudagraph(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D"],
        x_vals=CONFIG,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="qwen_image_residual_layernorm_scale_shift_gate_select01",
        args={},
    )
)
def bench_residual_layernorm_scale_shift_gate_select01(
    B: int, S: int, D: int, provider: str
) -> Tuple[float, float, float]:
    x, weight, bias, index, scale0, shift0, gate0, scale1, shift1, gate1 = (
        _make_common_inputs(B, S, D)
    )
    residual = torch.randn_like(x)
    residual_gate = torch.randn_like(x)

    if provider == "split":

        def fn():
            residual_out = residual + residual_gate * x
            normalized = norm_infer(
                residual_out.view(-1, residual_out.shape[-1]),
                weight,
                bias,
                eps=EPS,
                is_rms_norm=False,
            ).view_as(residual_out)
            return _apply_select01_modulation(
                normalized, scale0, shift0, gate0, scale1, shift1, gate1, index
            )

    else:

        def fn():
            return fuse_residual_layernorm_scale_shift_gate_select01_kernel(
                x,
                residual=residual,
                residual_gate=residual_gate,
                weight=weight,
                bias=bias,
                scale0=scale0,
                shift0=shift0,
                gate0=gate0,
                scale1=scale1,
                shift1=shift1,
                gate1=gate1,
                index=index,
                eps=EPS,
            )

    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("Benchmark: qwen_image layernorm + scale_shift_gate_select01")
    print(f"{'=' * 80}\n")
    bench_layernorm_scale_shift_gate_select01.run(print_data=True)

    print(f"\n{'=' * 80}")
    print("Benchmark: qwen_image residual + layernorm + scale_shift_gate_select01")
    print(f"{'=' * 80}\n")
    bench_residual_layernorm_scale_shift_gate_select01.run(print_data=True)

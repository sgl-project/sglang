import itertools

import sgl_kernel
import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import run_benchmark
from sglang.jit_kernel.cutedsl_dual_gemm import cutedsl_dual_gemm
from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8
from sglang.srt.compilation.fusion.ops.triton_ops.dual_gemm import dual_gemm

DTYPE = torch.bfloat16
DEVICE = "cuda"
M_LIST = [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024]
K_LIST = [1024, 2048, 4096]
N_LIST = [1024, 2048, 4096, 8192]

configs = list(itertools.product(M_LIST, K_LIST, N_LIST))


def _is_sm90():
    return torch.cuda.get_device_capability()[0] >= 9


def cutedsl_fn(x_fp8, w_fp8, out_fp8, x_scale, w_scale, o_scale):
    cutedsl_dual_gemm(x_fp8, w_fp8, out_fp8, x_scale, w_scale, o_scale)


def triton_fn(x_fp8, w_fp8, x_scale, w_scale, o_scale):
    return dual_gemm(x_fp8, w_fp8, x_scale, w_scale, o_scale)


def reference_fp8_fn(
    x_fp8, w_fp8_col_major, out_bf16, out_fp8, x_scale, w_scale, o_scale
):
    # Step 1: FP8 scaled matmul -> BF16
    # w_fp8_col_major is (K, 2*N) column-major for _scaled_mm
    mm_result = torch._scaled_mm(
        x_fp8,
        w_fp8_col_major,
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=torch.bfloat16,
    )
    # Step 2: silu_and_mul
    sgl_kernel.silu_and_mul(mm_result, out_bf16)
    # Step 3: per-tensor FP8 quantize
    per_tensor_quant_fp8(out_bf16, out_fp8, o_scale, is_static=True)


def warmup():
    if not _is_sm90():
        return
    for K, N in set((K, N) for _, K, N in configs):
        M = M_LIST[0]
        x = torch.randn((M, K), dtype=DTYPE, device=DEVICE)
        x_fp8 = x.to(torch.float8_e4m3fn)
        w_gate_fp8 = torch.randn((K, N), dtype=DTYPE, device=DEVICE).to(
            torch.float8_e4m3fn
        )
        w_up_fp8 = torch.randn((K, N), dtype=DTYPE, device=DEVICE).to(
            torch.float8_e4m3fn
        )
        out_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=DEVICE)
        x_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
        w_scale = torch.ones((2 * N), dtype=torch.float32, device=DEVICE)
        o_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
        w_fp8 = torch.cat([w_gate_fp8, w_up_fp8], dim=1)
        cutedsl_dual_gemm(x_fp8, w_fp8, out_fp8, x_scale, w_scale, o_scale)
        torch.cuda.synchronize()


LINE_VALS = ["cutedsl", "triton", "reference"]
LINE_NAMES = [
    "CuteDSL Dual GEMM FP8",
    "Triton Dual GEMM FP8",
    "Reference (scaled_mm + silu_and_mul + quant_fp8)",
]
STYLES = [("blue", "-"), ("orange", "--"), ("red", ":")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "K", "N"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dual-gemm-fp8-performance",
        args={},
    )
)
def benchmark(M: int, K: int, N: int, provider: str):
    if provider == "cutedsl" and not _is_sm90():
        return 0.0, 0.0, 0.0

    x_bf16 = torch.randn((M, K), dtype=DTYPE, device=DEVICE)
    x_fp8 = x_bf16.to(torch.float8_e4m3fn)
    w_gate_fp8 = torch.randn((K, N), dtype=DTYPE, device=DEVICE).to(torch.float8_e4m3fn)
    w_up_fp8 = torch.randn((K, N), dtype=DTYPE, device=DEVICE).to(torch.float8_e4m3fn)
    out_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=DEVICE)
    x_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    w_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    o_scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)

    # For reference: _scaled_mm expects b as column-major (2*N, K).t()
    w_cat = torch.cat([w_gate_fp8, w_up_fp8], dim=1)  # (K, 2*N)
    w_fp8_t = w_cat.t().contiguous().t()  # (K, 2*N) column-major
    out_bf16 = torch.empty((M, N), dtype=DTYPE, device=DEVICE)

    FN_MAP = {
        "cutedsl": lambda: cutedsl_fn(x_fp8, w_cat, out_fp8, x_scale, w_scale, o_scale),
        "triton": lambda: triton_fn(x_fp8, w_cat, x_scale, w_scale, o_scale),
        "reference": lambda: reference_fp8_fn(
            x_fp8, w_fp8_t, out_bf16, out_fp8, x_scale, w_scale, o_scale
        ),
    }
    return run_benchmark(FN_MAP[provider])


if __name__ == "__main__":
    warmup()
    benchmark.run(print_data=True)

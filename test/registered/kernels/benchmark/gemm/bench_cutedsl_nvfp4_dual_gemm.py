"""Benchmark SM100 CuTe DSL NVFP4 dual GEMM + SwiGLU against PyTorch _scaled_mm."""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm import nvfp4_dual_gemm_swiglu
from sglang.kernels.ops.gemm.cutedsl_nvfp4_dual_gemm import (
    _ceil_div,
    permute_nvfp4_scale,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="nightly", runner_config="4-gpu-b200")

_BENCHMARK_SHAPES = [
    (256, 4096, 7168),
    (512, 4096, 7168),
    (256, 3072, 4096),
    (512, 3072, 7168),
]


def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    if rows != n_row_blocks * 128 or cols != n_col_blocks * 4:
        padded = torch.zeros(
            (n_row_blocks * 128, n_col_blocks * 4),
            dtype=input_matrix.dtype,
            device=input_matrix.device,
        )
        padded[:rows, :cols] = input_matrix
        input_matrix = padded
    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _torch_nvfp4_dual_gemm(a, b1, b2, sa, sb1, sb2, out):
    res1 = torch._scaled_mm(
        a[:, :, 0],
        b1[:, :, 0].transpose(0, 1),
        sa,
        sb1,
        bias=None,
        out_dtype=torch.float32,
    )
    res2 = torch._scaled_mm(
        a[:, :, 0],
        b2[:, :, 0].transpose(0, 1),
        sa,
        sb2,
        bias=None,
        out_dtype=torch.float32,
    )
    out[:, :, 0] = (F.silu(res1) * res2).to(torch.float16)
    return out


@marker.parametrize("m,n,k", _BENCHMARK_SHAPES)
@marker.benchmark("impl", ["cutedsl", "torch"], unit="us")
def benchmark_nvfp4_dual_gemm(m: int, n: int, k: int, impl: str):
    torch.manual_seed(1111)
    l = 1
    sf_k = _ceil_div(k, 16)

    def create_fp4(mn: int) -> torch.Tensor:
        raw = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda")
        raw = raw & 0b1011_1011
        return raw.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)

    a = create_fp4(m)
    b1 = create_fp4(n)
    b2 = create_fp4(n)
    sfa_ref = (
        torch.rand((l, m, sf_k), dtype=torch.float32, device="cuda")
        .to(torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    sfb1_ref = (
        torch.rand((l, n, sf_k), dtype=torch.float32, device="cuda")
        .to(torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    sfb2_ref = (
        torch.rand((l, n, sf_k), dtype=torch.float32, device="cuda")
        .to(torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    out = torch.empty((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    if impl == "cutedsl":
        sfa_perm = permute_nvfp4_scale(sfa_ref)
        sfb1_perm = permute_nvfp4_scale(sfb1_ref)
        sfb2_perm = permute_nvfp4_scale(sfb2_ref)
        return marker.do_bench(
            nvfp4_dual_gemm_swiglu,
            input_args=(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, out),
        )
    else:
        sa = _to_blocked(sfa_ref[:, :, 0])
        sb1 = _to_blocked(sfb1_ref[:, :, 0])
        sb2 = _to_blocked(sfb2_ref[:, :, 0])
        return marker.do_bench(
            _torch_nvfp4_dual_gemm,
            input_args=(a, b1, b2, sa, sb1, sb2, out),
        )


if __name__ == "__main__":
    benchmark_nvfp4_dual_gemm.run()

from __future__ import annotations

import torch
from sgl_kernel.gemm import fp8_scaled_mm as fp8_scaled_mm_aot

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm.fp8_scaled_mm import fp8_scaled_mm as fp8_scaled_mm_jit
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _torch_reference(a, b, scales_a, scales_b, out_dtype):
    return ((a.float() @ b.float()) * scales_a.reshape(-1, 1) * scales_b).to(out_dtype)


FN_MAP = {
    "jit": fp8_scaled_mm_jit,
    "aot": fp8_scaled_mm_aot,
    "torch": _torch_reference,
}


@marker.parametrize(
    "m,n,k",
    [
        (1, 4096, 4096),
        (17, 9216, 2048),
        (128, 4096, 4096),
        (189, 4608, 8192),
        (3330, 256, 8192),
    ],
    [(1, 4096, 4096), (189, 4608, 8192)],
)
@marker.parametrize("out_dtype", [torch.bfloat16, torch.float16], [torch.bfloat16])
@marker.benchmark("provider", ["jit", "aot", "torch"])
def benchmark(m, n, k, out_dtype, provider):
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).t()
    scales_a = torch.rand(m, device="cuda", dtype=torch.float32) * 0.05
    scales_b = torch.rand(n, device="cuda", dtype=torch.float32) * 0.05
    return marker.do_bench(
        FN_MAP[provider],
        input_args=(a, b, scales_a, scales_b, out_dtype),
    )


if __name__ == "__main__":
    arch = get_jit_cuda_arch()
    if is_hip_runtime() or not (
        (arch.major, arch.minor) == (8, 9) or arch.major in (9, 10, 12)
    ):
        print("fp8_scaled_mm requires SM89, SM90, SM100/103, or SM120")
    else:
        benchmark.run()

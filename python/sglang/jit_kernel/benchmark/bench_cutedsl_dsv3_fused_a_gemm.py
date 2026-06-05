# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark the CuTe DSL dsv3 fused-A GEMM (JIT) against the sgl_kernel AOT kernel.

Both are timed with flashinfer.testing.bench_gpu_time_with_cupti (CUPTI HW tracing,
cold L2). The AOT kernel's PDL device intrinsics don't survive a CUDA-graph timing
harness, so the CUPTI path is used for a fair, capture-free comparison.
"""

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-benchmark-1-gpu-large")

GEMM_M = 2112
GEMM_K = 7168

try:
    from sgl_kernel import dsv3_fused_a_gemm as aot_fn

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

try:
    from sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm import dsv3_fused_a_gemm as jit_fn

    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False


def _median_us(fn) -> float:
    from flashinfer.testing import bench_gpu_time_with_cupti

    times = bench_gpu_time_with_cupti(fn, use_cuda_graph=False, cold_l2_cache=True)
    return torch.tensor(times, dtype=torch.float64).median().item() * 1e3


def benchmark():
    torch.manual_seed(0)
    weight = torch.randn(GEMM_M, GEMM_K, dtype=torch.bfloat16, device="cuda")
    mat_b = weight.t()  # [GEMM_K, GEMM_M] column-major (the weight)
    num_tokens = [1] if is_in_ci() else [1, 2, 4, 8, 12, 16]

    print(f"dsv3 fused-A GEMM  K={GEMM_K} N={GEMM_M}  (CUPTI cold-L2, us)")
    print(f"{'M':>4} {'aot':>9} {'jit':>9} {'aot/jit':>9}")
    for m in num_tokens:
        a = torch.randn(m, GEMM_K, dtype=torch.bfloat16, device="cuda")
        aot_us = _median_us(lambda: aot_fn(a, mat_b)) if AOT_AVAILABLE else float("nan")
        jit_us = _median_us(lambda: jit_fn(a, mat_b)) if JIT_AVAILABLE else float("nan")
        ratio = aot_us / jit_us if (AOT_AVAILABLE and JIT_AVAILABLE) else float("nan")
        print(f"{m:>4} {aot_us:>9.2f} {jit_us:>9.2f} {ratio:>9.2f}")


if __name__ == "__main__":
    benchmark()

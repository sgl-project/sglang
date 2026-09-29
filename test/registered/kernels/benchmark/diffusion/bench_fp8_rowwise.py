# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fp8_rowwise
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("rows", [32, 340, 2720, 3173], [3173])
@marker.parametrize("hidden", [3072, 9216], [3072])
@marker.benchmark("impl", ["eager", "fused"], unit="us")
def benchmark(rows: int, hidden: int, impl: str):
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device="cuda")

    def eager():
        values = F.pad(x, (0, 0, 0, -rows % 16)).float()
        scale = (values.abs().amax(1) / 448.0).clamp(min=1e-12)
        return (values / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn), scale

    fn = eager if impl == "eager" else lambda: fp8_rowwise(x, 16)
    return marker.do_bench(fn, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()

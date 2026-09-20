from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=6, stage="jit-kernel-benchmark", runner_config="amd")

EPS = 1e-6


def reference(
    x: torch.Tensor,
    prev: torch.Tensor,
    ew: torch.Tensor,
    hw: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    xf = x.float()
    pf = prev.float()
    x_var = xf.pow(2).mean(dim=-1, keepdim=True)
    p_var = pf.pow(2).mean(dim=-1, keepdim=True)
    return torch.cat(
        (
            (xf * torch.rsqrt(x_var + eps) * ew.float()).to(x.dtype),
            (pf * torch.rsqrt(p_var + eps) * hw.float()).to(prev.dtype),
        ),
        dim=-1,
    )


FN_MAP = {
    "jit": fused_eh_norm,
    "torch": reference,
}


@marker.parametrize("dtype", [torch.bfloat16, torch.float16], [torch.bfloat16])
@marker.parametrize("hidden_size", [6144, 7168], [7168])
@marker.parametrize("num_tokens", [1, 4, 6, 8, 16, 32, 128, 512], [1, 6])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(num_tokens: int, hidden_size: int, dtype: torch.dtype, impl: str):
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    prev = torch.randn_like(x)
    ew = torch.randn(hidden_size, device="cuda", dtype=dtype)
    hw = torch.randn(hidden_size, device="cuda", dtype=dtype)

    expected = reference(x, prev, ew, hw, EPS)
    actual = fused_eh_norm(x, prev, ew, hw, EPS)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)

    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, prev, ew, hw, EPS),
        memory_args=(x, prev, ew, hw),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()

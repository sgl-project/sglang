from dataclasses import dataclass

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fused_interleaved_rope_fp64
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    seq_len: int
    num_heads: int
    head_dim: int


CASES = {
    case.name: case
    for case in (
        Case("sana_video_480p", 2, 7800, 20, 112),
        Case("sana_video_small", 2, 1920, 20, 112),
    )
}


def eager_interleaved_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def apply(hidden_states: torch.Tensor) -> torch.Tensor:
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        output = torch.empty_like(hidden_states)
        output[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
        output[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]
        return output

    return apply(q), apply(k)


FN_MAP = {
    "eager": eager_interleaved_rope,
    "jit": fused_interleaved_rope_fp64,
}


@marker.parametrize("case_name", list(CASES), ci_vals=["sana_video_480p"])
@marker.benchmark("impl", ["eager", "jit"], unit="ms")
def benchmark(case_name: str, impl: str) -> marker.BenchResult:
    case = CASES[case_name]
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(
        case.batch,
        case.seq_len,
        case.num_heads,
        case.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn_like(q)
    cos = torch.randn(
        1,
        case.seq_len,
        1,
        case.head_dim,
        dtype=torch.float64,
        device="cuda",
        generator=generator,
    )
    sin = torch.randn_like(cos)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(q, k, cos, sin),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()

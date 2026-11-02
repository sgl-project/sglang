import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.l2norm import fused_l2norm_qk, l2norm_fwd
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=8, stage="jit-kernel-benchmark", runner_config="amd")


def _run_fused(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_l2norm_qk(q, k)


def _run_separate(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return l2norm_fwd(q), l2norm_fwd(k)


FN_MAP = {
    "fused": _run_fused,
    "separate": _run_separate,
}


@marker.parametrize("tokens", [15, 16, 17, 257, 1024], [257])
@marker.parametrize(
    "local_heads,head_dim",
    [
        (2, 128),
        (4, 128),
        (8, 128),
        (16, 128),
        (8, 256),
    ],
)
@marker.benchmark("impl", ["fused", "separate"])
def benchmark(tokens: int, local_heads: int, head_dim: int, impl: str):
    q = torch.randn(tokens, local_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(tokens, local_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(q, k),
        warmup_iters=80,
        replay_iters=1200,
        graph_clone_args=(0, 1),
        memory_args=(q, k),
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()

"""``vdn_delta_factors`` (fused inverse + products) vs the eager Cholesky chain it replaces."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import vdn_delta_factors
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

D_ = 128


def eager_delta_factors(A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor):
    eye = torch.eye(D_, device=A.device, dtype=torch.float32).expand_as(A)
    chol = torch.linalg.cholesky(A + eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    inv = linv.transpose(-1, -2) @ linv
    return alpha.unsqueeze(-1) * inv, B @ inv


FN_MAP = {"jit": vdn_delta_factors, "eager": eager_delta_factors}


def _inputs(num: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    k = torch.nn.functional.normalize(
        torch.randn(num, 1008, D_, device="cuda", generator=g), dim=-1
    )
    v = torch.randn(num, 1008, D_, device="cuda", generator=g)
    beta = torch.sigmoid(torch.randn(num, 1008, device="cuda", generator=g))
    A = (k * beta.unsqueeze(-1)).transpose(-1, -2) @ k
    A = 0.5 * (A + A.transpose(-1, -2))
    B = (v * beta.unsqueeze(-1)).transpose(-1, -2) @ k
    alpha = torch.rand(num, D_, device="cuda", generator=g)
    return A.contiguous(), B.contiguous(), alpha.contiguous()


# 707 = 101 frames x 7 heads: the paper workload per rank (8 x B200, Ulysses 8)
@marker.parametrize("num_matrices", [64, 707], [64])
@marker.benchmark("impl", ["jit", "eager"], unit="us")
def benchmark(num_matrices: int, impl: str):
    A, B, alpha = _inputs(num_matrices)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(A, B, alpha),
        graph_clone_args=(),
        use_cuda_graph=impl == "jit",
    )


if __name__ == "__main__":
    benchmark.run()

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import wan_rmsnorm_silu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="standalone benchmark",
)

DEVICE = "cuda"


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, int, int, int, int]
    x_dtype: torch.dtype
    affine_dtype: torch.dtype
    atol: float
    rtol: float


CASES = [
    Case(
        "fastwan21_c96_t4_h480_w832",
        (1, 96, 4, 480, 832),
        torch.bfloat16,
        torch.float32,
        1.5e-1,
        3e-2,
    ),
    Case(
        "fastwan22_c256_t4_h384_w576",
        (1, 256, 4, 384, 576),
        torch.float32,
        torch.float32,
        1e-5,
        1e-5,
    ),
]
CASE_BY_NAME = {case.name: case for case in CASES}
CASE_NAMES = list(CASE_BY_NAME)


@torch.no_grad()
def native_wan_rmsnorm_silu(
    x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return F.silu(F.normalize(x, dim=1) * x.shape[1] ** 0.5 * gamma + bias)


@torch.no_grad()
def sglang_wan_rmsnorm_silu(
    x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return wan_rmsnorm_silu(x, gamma, bias)


def make_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(case.shape[1] * 1009 + case.shape[-1])
    x = torch.randn(
        case.shape,
        device=DEVICE,
        dtype=case.x_dtype,
        generator=generator,
    ).contiguous(memory_format=torch.channels_last_3d)
    gamma = torch.randn(
        (case.shape[1], 1, 1, 1),
        device=DEVICE,
        dtype=case.affine_dtype,
        generator=generator,
    )
    bias = torch.randn_like(gamma)
    return x, gamma, bias


@marker.parametrize("case_name", CASE_NAMES)
@marker.benchmark("provider", ["torch", "sglang"])
def benchmark(case_name: str, provider: str) -> marker.BenchResult:
    case = CASE_BY_NAME[case_name]
    x, gamma, bias = make_inputs(case)
    expected = native_wan_rmsnorm_silu(x, gamma, bias)
    actual = sglang_wan_rmsnorm_silu(x, gamma, bias)
    torch.testing.assert_close(actual, expected, atol=case.atol, rtol=case.rtol)
    assert actual.stride() == x.stride()

    fn = native_wan_rmsnorm_silu if provider == "torch" else sglang_wan_rmsnorm_silu
    return marker.do_bench(
        fn,
        input_args=(x, gamma, bias),
        use_cuda_graph=False,
        replay_iters=100,
        memory_args=(x, gamma, bias),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()

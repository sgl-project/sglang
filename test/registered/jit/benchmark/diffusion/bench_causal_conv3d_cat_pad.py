from dataclasses import dataclass

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.kernels.ops.diffusion.causal_conv3d_cat_pad import (
    fused_causal_conv3d_cat_pad_cuda,
)
from sglang.kernels.ops.diffusion.triton.causal_conv3d_pad import (
    fused_causal_conv3d_cat_pad as fused_causal_conv3d_cat_pad_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="standalone benchmark",
)
register_amd_ci(est_time=20, stage="jit-kernel-benchmark", runner_config="amd")

DEVICE = "cuda"
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class Case:
    name: str
    channels: int
    t_size: int
    h_size: int
    w_size: int
    cache_t: int
    trace_count: int


CASES = [
    Case("c1024_t1_h30_w52_cache1", 1024, 1, 30, 52, 1, 8),
    Case("c1024_t1_h30_w52_cache2", 1024, 1, 30, 52, 2, 8),
    Case("c1024_t2_h60_w104_cache1", 1024, 2, 60, 104, 1, 5),
    Case("c1024_t2_h60_w104_cache2", 1024, 2, 60, 104, 2, 5),
    Case("c512_t4_h120_w208_cache1", 512, 4, 120, 208, 1, 5),
    Case("c512_t4_h120_w208_cache2", 512, 4, 120, 208, 2, 5),
    Case("c256_t4_h240_w416_cache1", 256, 4, 240, 416, 1, 6),
    Case("c256_t4_h240_w416_cache2", 256, 4, 240, 416, 2, 6),
]
CASE_BY_NAME = {case.name: case for case in CASES}
CASE_NAMES = [case.name for case in CASES]


def make_inputs(case: Case) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(case.channels * 1009 + case.t_size * 251 + case.cache_t)
    x = torch.randn(
        (1, case.channels, case.t_size, case.h_size, case.w_size),
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    cache_x = torch.randn(
        (1, case.channels, case.cache_t, case.h_size, case.w_size),
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    padding = (1, 1, 1, 1, case.cache_t, 0)
    return x, cache_x, padding


@marker.parametrize("case_name", CASE_NAMES, ci_vals=CASE_NAMES[:2])
@marker.benchmark("provider", ["triton", "cuda"])
def benchmark(case_name: str, provider: str) -> marker.BenchResult:
    case = CASE_BY_NAME[case_name]
    x, cache_x, padding = make_inputs(case)
    fn = (
        fused_causal_conv3d_cat_pad_triton
        if provider == "triton"
        else fused_causal_conv3d_cat_pad_cuda
    )
    actual = fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
    expected = fused_causal_conv3d_cat_pad_triton(x, cache_x, padding)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return marker.do_bench(
        fn,
        input_args=(x, cache_x, padding),
        use_cuda_graph=False,
        replay_iters=200,
        graph_clone_args=(0, 1),
        memory_args=(x, cache_x),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()

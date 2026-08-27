"""Benchmark: M=1 skinny fp8 PTPC GEMV (JIT HIP, gfx950) vs the aiter
bpreshuffle GEMM it replaces.

Both consume a [1, K] fp8 activation and an (16,16)-preshuffled fp8 weight
with per-token / per-channel fp32 scales and produce a [1, N] bf16 output.
The shapes are the M3 qkv projections the dispatch gate covers.

The kernel is weight-read bound; warm (cache-resident) numbers here
understate the cold production gap.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=8, stage="jit-kernel-benchmark", runner_config="amd")

SHAPES = [(1280, 6144), (1536, 6144), (2304, 6144), (2560, 6144)]


def _aiter(aq, w_shuf, x_scale, w_scale):
    import aiter

    return aiter.gemm_a8w8_bpreshuffle(
        aq, w_shuf, x_scale, w_scale, None, torch.bfloat16
    )


def _jit(aq, w_shuf, x_scale, w_scale):
    from sglang.kernels.ops.gemm.skinny_ptpc_gemv import skinny_ptpc_gemv

    return skinny_ptpc_gemv(aq, w_shuf, x_scale, w_scale)


FN_MAP = {"jit": _jit, "aiter_bpreshuffle": _aiter}


@marker.parametrize("shape", SHAPES, [SHAPES[0], SHAPES[1]])
@marker.benchmark("impl", ["jit", "aiter_bpreshuffle"])
def benchmark(shape, impl: str):
    from aiter.ops.shuffle import shuffle_weight

    n, k = shape
    torch.manual_seed(0)
    wq = (
        (torch.randn(n, k, device="cuda") * 0.02)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    w_shuf = shuffle_weight(wq, (16, 16))
    w_scale = torch.rand(n, 1, device="cuda").float()
    aq = torch.randn(1, k, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
    x_scale = torch.rand(1, 1, device="cuda").float()
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(aq, w_shuf, x_scale, w_scale),
        graph_clone_args=(0, 1, 2, 3),  # all read-only inputs
        memory_args=(w_shuf,),
    )


if __name__ == "__main__":
    benchmark.run()

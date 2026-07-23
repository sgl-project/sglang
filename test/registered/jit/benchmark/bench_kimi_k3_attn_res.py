"""Benchmark the production K3 SM100 attention-residual TMA kernel."""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_empty, create_random
from sglang.jit_kernel.kimi_k3.attn_res import attn_res_fused_tma
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

_H = 7168
_NB = 8
_EPS = 1e-6


def _run_tma(prefix, bank, cw, ow, out, nvb):
    attn_res_fused_tma(prefix, bank, cw, ow, out, nvb, _EPS)


@marker.parametrize("nvb", list(range(1, 9)), [8])
@marker.parametrize("num_tokens", [2**x for x in range(14)], [1, 64])
@marker.benchmark("impl", ["tma"])
def benchmark(num_tokens: int, nvb: int, impl: str):
    if torch.cuda.get_device_capability()[0] < 10:
        marker.skip("attn_res tma impl requires SM100a+")

    prefix = create_random(num_tokens, _H)
    bank = create_random(num_tokens, _NB, _H)
    cw = (create_random(_H) * _H**-0.5).contiguous()
    ow = create_random(_H)
    out = create_empty(num_tokens, _H)
    args = (prefix, bank, cw, ow, out, nvb)

    return marker.do_bench(
        _run_tma,
        input_args=args,
        graph_clone_args=(0, 1),  # prefix / bank are the read inputs
        memory_args=(prefix, bank[:, :nvb]),
        memory_output=(out,),
    )


if __name__ == "__main__":
    benchmark.run()

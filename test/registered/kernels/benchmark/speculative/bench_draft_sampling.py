"""Benchmark the Gumbel-max draft sample: one-pass div_argmax vs clamp_min_ + division + argmax."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.sampling import argmax as sampling_argmax
from sglang.kernels.ops.speculative.row_argmax import div_argmax
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

_TINY = torch.finfo(torch.float32).tiny


def torch_sample(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1.0)
    q.clamp_min_(_TINY)
    return sampling_argmax(probs / q)


def fused_sample(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1.0)
    return div_argmax(probs, q, divisor_min=_TINY, ignore_nan=True)


FN_MAP = {"fused": fused_sample, "torch": torch_sample}


@marker.parametrize("vocab_size", [32000, 129280, 151936, 248320], [151936])
@marker.parametrize("batch_size", [1, 4, 16, 64, 128, 256], [1, 64])
@marker.benchmark("impl", ["fused", "torch"])
def benchmark(vocab_size: int, batch_size: int, impl: str):
    probs = torch.softmax(
        torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.float32),
        dim=-1,
    )
    return marker.do_bench(FN_MAP[impl], input_args=(probs,))


if __name__ == "__main__":
    benchmark.run()

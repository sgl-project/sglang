"""Compare SynthID sampling through the logits transform and a stock sampler
against the fused sample kernel; both providers then commit the drawn tokens."""

from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.sampling.watermarking.synthid.algorithm import (
    REFERENCE_SYNTHID_KEYS,
    SynthIDConfig,
)
from sglang.srt.sampling.watermarking.synthid.sampler import SynthIDSampler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
CI_BATCH_SIZES = [1, 16, 64]
VOCAB_SIZE = 248_320


@marker.parametrize("batch_size", BATCH_SIZES, CI_BATCH_SIZES)
@marker.benchmark("provider", ["transform", "fused"])
def benchmark_stochastic(batch_size: int, provider: str):
    config = SynthIDConfig(keys=REFERENCE_SYNTHID_KEYS, top_k=1024)
    sampler = SynthIDSampler(config, max(BATCH_SIZES), "cuda")
    slots = torch.arange(batch_size, device="cuda")
    sampler.state.seed(slots, [[] for _ in range(batch_size)])
    logits = torch.randn(batch_size, VOCAB_SIZE, dtype=torch.float32, device="cuda")
    temperatures = torch.ones((batch_size, 1), dtype=torch.float32, device="cuda")

    if provider == "transform":

        def fn():
            watermarked = logits.clone()
            sampler.apply(watermarked, temperatures, slots)
            probs = torch.softmax(watermarked / temperatures, dim=-1)
            noise = torch.empty_like(probs).exponential_(1.0)
            token_ids = (probs / noise).argmax(dim=-1)
            sampler.state.commit(slots, token_ids)

    elif provider == "fused":

        def fn():
            token_ids = sampler.sample(logits, temperatures, slots)
            sampler.state.commit(slots, token_ids)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return marker.do_bench(
        fn,
        use_cuda_graph=False,
        memory_args=(logits,),
        memory_output=None,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark_stochastic.run()

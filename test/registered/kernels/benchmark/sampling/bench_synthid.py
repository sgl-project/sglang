"""Compare fused and unfused SynthID stochastic sampling."""

from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.watermarking.synthid import (
    REFERENCE_SYNTHID_KEYS,
    SynthIDConfig,
    SynthIDState,
    SynthIDWatermarkProcessor,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
CI_BATCH_SIZES = [1, 16, 64]
VOCAB_SIZE = 248_320


@marker.parametrize("batch_size", BATCH_SIZES, CI_BATCH_SIZES)
@marker.benchmark("provider", ["sparse", "fused"])
def benchmark_stochastic(batch_size: int, provider: str):
    keys = tuple(REFERENCE_SYNTHID_KEYS)
    config = SynthIDConfig(
        keys=keys,
        top_k=1024,
    )
    processor = SynthIDWatermarkProcessor(
        config,
        device="cuda",
        num_slots=max(BATCH_SIZES),
    )
    state = SynthIDState.from_output_ids(
        [[] for _ in range(batch_size)],
        config=config,
        device="cuda",
        slot_indices=torch.arange(batch_size, device="cuda"),
        generations=torch.ones(batch_size, dtype=torch.int64, device="cuda"),
    )
    logits = torch.randn(batch_size, VOCAB_SIZE, dtype=torch.float32, device="cuda")
    temperatures = torch.ones((batch_size, 1), dtype=torch.float32, device="cuda")
    noise = torch.empty(
        (batch_size, config.top_k), dtype=torch.float32, device="cuda"
    ).exponential_(1.0)

    if provider == "sparse":

        def fn():
            output = processor.apply(
                logits,
                temperatures,
                state,
                scatter_to_logits=False,
            )
            sampled_indices = (torch.softmax(output.scores, dim=-1) / noise).argmax(
                dim=-1, keepdim=True
            )
            token_ids = output.token_ids.gather(1, sampled_indices).view(-1)
            processor.commit(state, token_ids)

    elif provider == "fused":

        def fn():
            processor.sample_stochastic_and_commit(
                logits,
                temperatures,
                state,
                exponential_noise=noise,
            )

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

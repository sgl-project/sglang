"""Benchmark CUDA topk=1 speculative decoding helpers."""

from __future__ import annotations

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=30, stage="jit-kernel-benchmark", runner_config="amd")


BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    ci_range=[1, 16, 256, 2048],
)
VOCAB_SIZES = {
    "dsv4": 129280,
    "glm5_2": 151552,
}
VOCAB_SIZE_RANGE = get_benchmark_range(
    full_range=list(VOCAB_SIZES.values()),
    ci_range=list(VOCAB_SIZES.values()),
)
NUM_STEPS = 3


def make_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(
        (batch_size, vocab_size), dtype=torch.float32, device=DEFAULT_DEVICE
    )
    max_index = (
        torch.arange(batch_size, dtype=torch.long, device=DEFAULT_DEVICE) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, max_index[:, None], 1000.0)
    return logits


def make_draft_case(batch_size: int, vocab_size: int):
    logits = make_logits(batch_size, vocab_size)
    positions = torch.zeros(batch_size, dtype=torch.long, device=DEFAULT_DEVICE)
    return logits, positions


def make_chain_case(batch_size: int, vocab_size: int):
    seed_topk_index = torch.randint(
        0, vocab_size, (batch_size, 1), dtype=torch.long, device=DEFAULT_DEVICE
    )
    logits = [make_logits(batch_size, vocab_size) for _ in range(NUM_STEPS - 1)]
    positions = torch.zeros(batch_size, dtype=torch.long, device=DEFAULT_DEVICE)
    return seed_topk_index, logits, positions


def eager_draft_topk1_postprocess(logits: torch.Tensor, positions: torch.Tensor):
    topk_index = torch.argmax(logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    positions.add_(1)
    return topk_p, topk_index


def fused_draft_topk1_postprocess(logits: torch.Tensor, positions: torch.Tensor):
    return draft_topk1_postprocess(logits, positions)


def eager_chain_materialize(
    seed_topk_index: torch.Tensor,
    logits: list[torch.Tensor],
    positions: torch.Tensor,
):
    token_list = [seed_topk_index]
    for step_logits in logits:
        _, topk_index = eager_draft_topk1_postprocess(step_logits, positions)
        token_list.append(topk_index)
    return torch.cat(token_list, dim=1)


def fused_chain_materialize(
    seed_topk_index: torch.Tensor,
    logits: list[torch.Tensor],
    positions: torch.Tensor,
):
    draft_tokens = torch.empty(
        (seed_topk_index.shape[0], NUM_STEPS),
        dtype=torch.long,
        device=DEFAULT_DEVICE,
    )
    draft_tokens[:, :1].copy_(seed_topk_index)
    for i, step_logits in enumerate(logits, start=1):
        draft_topk1_postprocess(
            step_logits,
            positions,
            draft_tokens,
            draft_token_column=i,
        )
    return draft_tokens


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size"],
        x_vals=[(bs, vocab) for bs in BATCH_SIZE_RANGE for vocab in VOCAB_SIZE_RANGE],
        line_arg="provider",
        line_vals=["fused", "eager"],
        line_names=["Fused Triton", "Eager torch"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="spec-topk1-draft-postprocess",
        args={},
    )
)
def benchmark_draft_postprocess(
    batch_size: int, vocab_size: int, provider: str
) -> tuple[float, float, float]:
    logits, positions = make_draft_case(batch_size, vocab_size)
    if provider == "fused":
        fn = lambda: fused_draft_topk1_postprocess(logits, positions)
    elif provider == "eager":
        fn = lambda: eager_draft_topk1_postprocess(logits, positions)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    fn()
    torch.cuda.synchronize()
    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size"],
        x_vals=[(bs, vocab) for bs in BATCH_SIZE_RANGE for vocab in VOCAB_SIZE_RANGE],
        line_arg="provider",
        line_vals=["fused", "eager"],
        line_names=["Fused Triton", "Eager argmax + cat"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="spec-topk1-chain-materialize",
        args={},
    )
)
def benchmark_chain_materialize(
    batch_size: int, vocab_size: int, provider: str
) -> tuple[float, float, float]:
    seed_topk_index, logits, positions = make_chain_case(batch_size, vocab_size)
    if provider == "fused":
        fn = lambda: fused_chain_materialize(seed_topk_index, logits, positions)
    elif provider == "eager":
        fn = lambda: eager_chain_materialize(seed_topk_index, logits, positions)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    fn()
    torch.cuda.synchronize()
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark_draft_postprocess.run(print_data=True)
    benchmark_chain_materialize.run(print_data=True)

"""Benchmark CUDA topk=1 speculative decoding helpers."""

from __future__ import annotations

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.kernels.ops.speculative.topk1 import (
    draft_extend_topk1_postprocess,
    draft_topk1_postprocess,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


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
DRAFT_EXTEND_BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128],
    ci_range=[1, 16, 64],
)
DRAFT_EXTEND_TREE_WIDTH = 6
DRAFT_EXTEND_VOCAB_SIZE = 154880
DRAFT_EXTEND_HIDDEN_SIZE = 6144
DRAFT_EXTEND_DSA_TOPK = 2048


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


def make_draft_extend_case(batch_size: int):
    num_rows = batch_size * DRAFT_EXTEND_TREE_WIDTH
    logits = make_logits(num_rows, DRAFT_EXTEND_VOCAB_SIZE)
    hidden_states = torch.empty(
        (num_rows, DRAFT_EXTEND_HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=DEFAULT_DEVICE,
    )
    dsa_topk_indices = torch.empty(
        (max(720, num_rows), DRAFT_EXTEND_DSA_TOPK),
        dtype=torch.int32,
        device=DEFAULT_DEVICE,
    )
    row_indices = (
        torch.arange(batch_size, dtype=torch.long, device=DEFAULT_DEVICE)
        * DRAFT_EXTEND_TREE_WIDTH
        + DRAFT_EXTEND_TREE_WIDTH
        - 1
    )
    return logits, row_indices, hidden_states, dsa_topk_indices


def eager_draft_topk1_postprocess(logits: torch.Tensor, positions: torch.Tensor):
    topk_index = torch.argmax(logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    positions.add_(1)
    return topk_p, topk_index


def fused_draft_topk1_postprocess(logits: torch.Tensor, positions: torch.Tensor):
    return draft_topk1_postprocess(logits, positions)


def eager_draft_extend_topk1_postprocess(
    logits: torch.Tensor,
    row_indices: torch.Tensor,
    hidden_states: torch.Tensor,
    dsa_topk_indices: torch.Tensor,
):
    selected_dsa = dsa_topk_indices[row_indices]
    selected_logits = logits[row_indices]
    selected_hidden = hidden_states[row_indices]
    topk_index = torch.argmax(selected_logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    return topk_p, topk_index, selected_hidden, selected_dsa


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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=DRAFT_EXTEND_BATCH_SIZE_RANGE,
        line_arg="provider",
        line_vals=["fused", "eager"],
        line_names=["Fused indexed Triton", "Eager index + argmax"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="spec-topk1-draft-extend-postprocess",
        args={},
    )
)
def benchmark_draft_extend_postprocess(
    batch_size: int, provider: str
) -> tuple[float, float, float]:
    logits, row_indices, hidden_states, dsa_topk_indices = make_draft_extend_case(
        batch_size
    )
    if provider == "fused":
        fn = lambda: draft_extend_topk1_postprocess(
            logits,
            row_indices,
            hidden_states=hidden_states,
            dsa_topk_indices=dsa_topk_indices,
        )
    elif provider == "eager":
        fn = lambda: eager_draft_extend_topk1_postprocess(
            logits, row_indices, hidden_states, dsa_topk_indices
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    fn()
    torch.cuda.synchronize()
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark_draft_postprocess.run(print_data=True)
    benchmark_chain_materialize.run(print_data=True)
    benchmark_draft_extend_postprocess.run(print_data=True)

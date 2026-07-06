"""Benchmark CUDA topk=1 speculative decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.srt.speculative.triton_ops.topk1 import (
    draft_topk1_postprocess,
    select_top_k_tokens_topk1_later,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    ci_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
)
VOCAB_SIZES = {
    "glm5_2": 151552,
    "dsv4": 129280,
}
VOCAB_SIZE_RANGE = get_benchmark_range(
    full_range=list(VOCAB_SIZES.values()),
    ci_range=list(VOCAB_SIZES.values()),
)
NUM_STEPS = 3


@dataclass
class DraftPostprocessCase:
    logits: torch.Tensor
    positions: torch.Tensor


@dataclass
class SelectMaterializeCase:
    hidden_states: torch.Tensor
    topk_ps: list[torch.Tensor]
    topk_indices: list[torch.Tensor]


def make_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(
        (batch_size, vocab_size), dtype=torch.float32, device=DEFAULT_DEVICE
    )
    max_index = (
        torch.arange(batch_size, dtype=torch.long, device=DEFAULT_DEVICE) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, max_index[:, None], 1000.0)
    return logits


def make_draft_case(batch_size: int, vocab_size: int) -> DraftPostprocessCase:
    return DraftPostprocessCase(
        logits=make_logits(batch_size, vocab_size),
        positions=torch.zeros(batch_size, dtype=torch.long, device=DEFAULT_DEVICE),
    )


def make_select_case(batch_size: int) -> SelectMaterializeCase:
    topk_ps = [
        torch.rand((batch_size, 1), dtype=torch.float32, device=DEFAULT_DEVICE)
        for _ in range(NUM_STEPS)
    ]
    first = torch.randint(
        0, 151552, (batch_size, 1), dtype=torch.long, device=DEFAULT_DEVICE
    )
    topk_indices = [first, first + 17, first + 37]
    hidden_states = torch.randn(
        (batch_size, 128), dtype=torch.float32, device=DEFAULT_DEVICE
    )
    return SelectMaterializeCase(
        hidden_states=hidden_states,
        topk_ps=topk_ps,
        topk_indices=topk_indices,
    )


def eager_draft_topk1_postprocess(case: DraftPostprocessCase):
    topk_index = torch.argmax(case.logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    case.positions.add_(1)
    return topk_p, topk_index


def fused_draft_topk1_postprocess(case: DraftPostprocessCase):
    return draft_topk1_postprocess(case.logits, case.positions)


def eager_select_materialize(case: SelectMaterializeCase):
    scores = case.topk_ps[0]
    token_list = [case.topk_indices[0]]
    last_parents = None
    for i in range(1, NUM_STEPS):
        scores = scores * case.topk_ps[i]
        last_parents = torch.full_like(case.topk_indices[i], i)
        token_list.append(case.topk_indices[i])
    draft_tokens = torch.cat(token_list, dim=1)
    return scores, draft_tokens, last_parents


def fused_select_materialize(case: SelectMaterializeCase):
    scores = case.topk_ps[0]
    draft_tokens = torch.empty(
        (case.topk_indices[0].shape[0], NUM_STEPS),
        dtype=torch.long,
        device=DEFAULT_DEVICE,
    )
    last_tree_info = None
    for i in range(1, NUM_STEPS):
        _, _, scores, last_tree_info = select_top_k_tokens_topk1_later(
            i,
            case.topk_ps[i],
            case.topk_indices[i],
            case.hidden_states,
            scores,
            draft_tokens,
            case.topk_indices[0] if i == 1 else None,
        )
    return scores, draft_tokens, last_tree_info[2]


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
    case = make_draft_case(batch_size, vocab_size)
    if provider == "fused":
        fn = lambda: fused_draft_topk1_postprocess(case)
    elif provider == "eager":
        fn = lambda: eager_draft_topk1_postprocess(case)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    fn()
    torch.cuda.synchronize()
    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZE_RANGE,
        line_arg="provider",
        line_vals=["fused", "eager"],
        line_names=["Fused Triton", "Eager torch + cat"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="spec-topk1-select-materialize",
        args={},
    )
)
def benchmark_select_materialize(
    batch_size: int, provider: str
) -> tuple[float, float, float]:
    case = make_select_case(batch_size)
    if provider == "fused":
        fn = lambda: fused_select_materialize(case)
    elif provider == "eager":
        fn = lambda: eager_select_materialize(case)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    fn()
    torch.cuda.synchronize()
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark_draft_postprocess.run(print_data=True)
    benchmark_select_materialize.run(print_data=True)

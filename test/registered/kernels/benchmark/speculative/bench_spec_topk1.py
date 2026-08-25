"""Benchmark accelerated topk=1 speculative decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.srt.utils.common import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=30, stage="jit-kernel-benchmark", runner_config="amd")


BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    ci_range=[1, 16, 256, 2048],
)
REPRESENTATIVE_VOCAB_SIZES = [129280, 151552, 248320]
VOCAB_SIZE_RANGE = get_benchmark_range(
    full_range=REPRESENTATIVE_VOCAB_SIZES,
    ci_range=REPRESENTATIVE_VOCAB_SIZES[:2],
)
LARGE_VOCAB_SIZE_RANGE = get_benchmark_range(
    full_range=REPRESENTATIVE_VOCAB_SIZES,
    ci_range=REPRESENTATIVE_VOCAB_SIZES[-1:],
)
NUM_STEPS = 3
SERVING_BATCH_SIZE_RANGE = [4, 64, 256]
AMD_GPU_MODEL_BY_ARCH = {
    "gfx924": "MI300",
    "gfx950": "MI355",
    "gfx1250": "MI450",
}


@dataclass(frozen=True)
class BenchmarkEnvironment:
    backend: str
    provider: str
    gpu_model: str
    gpu_arch: str


def detect_benchmark_environment(provider: str | None = None) -> BenchmarkEnvironment:
    """Resolve benchmark dispatch and descriptive hardware metadata."""
    backend = "rocm" if is_hip() else "cuda"
    resolved_provider = provider or "triton"
    if resolved_provider not in {"aiter", "triton"}:
        raise ValueError(f"Unsupported accelerated provider: {resolved_provider}")
    if backend != "rocm" and resolved_provider == "aiter":
        raise ValueError("The AITER provider requires the ROCm backend")

    properties = torch.cuda.get_device_properties(DEFAULT_DEVICE)
    if backend == "rocm":
        gpu_arch = getattr(properties, "gcnArchName", "unknown").split(":", 1)[0]
        gpu_model = AMD_GPU_MODEL_BY_ARCH.get(gpu_arch, properties.name)
    else:
        major, minor = torch.cuda.get_device_capability(DEFAULT_DEVICE)
        gpu_arch = f"sm{major}{minor}"
        gpu_model = properties.name

    return BenchmarkEnvironment(
        backend=backend,
        provider=resolved_provider,
        gpu_model=gpu_model,
        gpu_arch=gpu_arch,
    )


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


def aiter_draft_topk1_postprocess(
    logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Old production path retained only for local AITER/Triton comparisons."""
    from aiter import greedy_sample

    batch_size = logits.shape[0]
    topk_index_i32 = torch.empty(batch_size, dtype=torch.int32, device=logits.device)
    greedy_sample(topk_index_i32, logits)
    topk_index = topk_index_i32.to(dtype=torch.long).view(batch_size, 1)
    topk_p = torch.ones((batch_size, 1), dtype=torch.float32, device=logits.device)
    positions.add_(1)
    if draft_tokens is not None:
        draft_tokens[:, draft_token_column].copy_(topk_index[:, 0])
    return topk_p, topk_index


def accelerated_draft_topk1_postprocess(
    logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
    provider: str | None = None,
):
    resolved_provider = provider or detect_benchmark_environment().provider
    if resolved_provider == "aiter":
        return aiter_draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column
        )
    if resolved_provider == "triton":
        return draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column
        )
    raise ValueError(f"Unsupported accelerated provider: {resolved_provider}")


def softmax_max_draft_topk1_postprocess(logits: torch.Tensor, positions: torch.Tensor):
    topk_p, topk_index = torch.max(torch.softmax(logits, dim=-1), dim=-1, keepdim=True)
    positions.add_(1)
    return topk_p, topk_index


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


def accelerated_chain_materialize(
    seed_topk_index: torch.Tensor,
    logits: list[torch.Tensor],
    positions: torch.Tensor,
    provider: str,
):
    draft_tokens = torch.empty(
        (seed_topk_index.shape[0], NUM_STEPS),
        dtype=torch.long,
        device=DEFAULT_DEVICE,
    )
    draft_tokens[:, :1].copy_(seed_topk_index)
    for i, step_logits in enumerate(logits, start=1):
        accelerated_draft_topk1_postprocess(
            step_logits,
            positions,
            draft_tokens,
            draft_token_column=i,
            provider=provider,
        )
    return draft_tokens


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size"],
        x_vals=[(bs, vocab) for bs in BATCH_SIZE_RANGE for vocab in VOCAB_SIZE_RANGE],
        line_arg="provider",
        line_vals=["accelerated", "eager"],
        line_names=["Triton", "Eager torch"],
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
    if provider == "accelerated":
        environment = detect_benchmark_environment()
        fn = lambda: accelerated_draft_topk1_postprocess(
            logits, positions, provider=environment.provider
        )
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
        line_vals=["accelerated", "eager"],
        line_names=["AITER / Triton", "Eager argmax + cat"],
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
    if provider == "accelerated":
        environment = detect_benchmark_environment()
        fn = lambda: accelerated_chain_materialize(
            seed_topk_index, logits, positions, provider=environment.provider
        )
    elif provider == "eager":
        fn = lambda: eager_chain_materialize(seed_topk_index, logits, positions)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    fn()
    torch.cuda.synchronize()
    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "vocab_size"],
        x_vals=[
            (batch_size, vocab_size)
            for batch_size in SERVING_BATCH_SIZE_RANGE
            for vocab_size in LARGE_VOCAB_SIZE_RANGE
        ],
        line_arg="provider",
        line_vals=["raw_argmax", "softmax_max"],
        line_names=["Raw-logits Triton", "Softmax + torch.max"],
        styles=[("blue", "-"), ("orange", "--")],
        ylabel="us",
        plot_name="large-vocab-draft-topk1",
        args={},
    )
)
def benchmark_large_vocab_draft_topk1(
    batch_size: int, vocab_size: int, provider: str
) -> tuple[float, float, float]:
    logits = make_logits(batch_size, vocab_size)
    positions = torch.zeros(batch_size, dtype=torch.long, device=DEFAULT_DEVICE)
    if provider == "raw_argmax":
        environment = detect_benchmark_environment()
        fn = lambda: accelerated_draft_topk1_postprocess(
            logits, positions, provider=environment.provider
        )
    elif provider == "softmax_max":
        fn = lambda: softmax_max_draft_topk1_postprocess(logits, positions)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    fn()
    torch.cuda.synchronize()
    return run_benchmark(fn)


if __name__ == "__main__":
    environment = detect_benchmark_environment()
    print(
        "benchmark_environment:"
        f" backend={environment.backend}"
        f" provider={environment.provider}"
        f" gpu_model={environment.gpu_model}"
        f" gpu_arch={environment.gpu_arch}"
    )
    benchmark_draft_postprocess.run(print_data=True)
    benchmark_chain_materialize.run(print_data=True)
    benchmark_large_vocab_draft_topk1.run(print_data=True)

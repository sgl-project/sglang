"""SGLANG_BENCH_DRAFTER_TEMPERATURE: a benchmark-only fixed draft sampling temperature.

With simulated acceptance, greedy drafts can collapse into short repeating loops.
A positive temperature samples from the draft head's candidates while the target
continues to verify greedily.
"""

from __future__ import annotations

import functools

import torch

from sglang.srt.environ import envs

ENV_NAME = "SGLANG_BENCH_DRAFTER_TEMPERATURE"
GREEDY_REQUEST_ERROR = f"{ENV_NAME} requires greedy requests (temperature 0 / top_k 1)."
SUPPORTED_ALGORITHMS = ("DFLASH", "DSPARK")

# Offset for callers with a second step counter (an eager fallback next to a CUDA
# graph's in-graph counter), so the two never reuse each other's noise.
EAGER_STEP_OFFSET = 1 << 31


@functools.cache
def bench_drafter_temperature() -> float:
    """SGLANG_BENCH_DRAFTER_TEMPERATURE, validated once per process; 0.0 when unset."""
    value = float(envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.get())
    # All draft samplers stage temperatures in float32; GPU kernels can flush
    # subnormal values to zero, which would turn sampling back into argmax.
    limits = torch.finfo(torch.float32)
    if value != 0.0 and not limits.tiny <= value <= limits.max:
        raise ValueError(
            f"{ENV_NAME} must be 0 (disabled) or a positive normal float32 "
            f"temperature, got {value!r}."
        )
    return value


def bench_drafter_enabled() -> bool:
    return bench_drafter_temperature() > 0.0


def check_algorithm(algorithm) -> None:
    """Server-args gate: refuse algorithms whose drafts would ignore the override."""
    if bench_drafter_enabled() and algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"{ENV_NAME} supports --speculative-algorithm "
            f"{' / '.join(SUPPORTED_ALGORITHMS)}, not {algorithm}."
        )


def require_greedy_requests(sampling_info) -> None:
    """A sampled request would be accepted against drafts drawn at a temperature it
    did not ask for; the benchmark keeps every request greedy."""
    if sampling_info is not None and not sampling_info.is_all_greedy:
        raise ValueError(GREEDY_REQUEST_ERROR)


def override_draft_sampling(
    temperatures: torch.Tensor, greedy_mask: torch.Tensor
) -> None:
    """In place, when enabled: the given draft-only temperature / greedy-mask rows sample
    at the benchmark temperature. Callers pass the ``[:bs]`` rows of buffers that only the
    draft proposal reads, never ones the target verify reads."""
    temperature = bench_drafter_temperature()
    if temperature <= 0.0:
        return
    temperatures.fill_(temperature)
    greedy_mask.fill_(False)


def gumbel_draft_scores(
    logits: torch.Tensor,
    *,
    col_start: int,
    row_start: int,
    step: torch.Tensor | int,
) -> torch.Tensor:
    """Score each global token with independent Gumbel noise at the draft temperature.

    Every TP rank uses the same (step, global row, global token) counters, so taking
    shard maxima then a global max agrees with sampling the full vocabulary.
    """
    temperature = bench_drafter_temperature()
    if temperature <= 0.0:
        return logits

    from sglang.kernels.ops.speculative.bench_drafter import draft_scores

    return draft_scores(logits, temperature, step, row_start, col_start)

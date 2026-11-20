"""SGLANG_BENCH_DRAFTER_TEMPERATURE: a benchmark-only fixed draft sampling temperature.

With simulated acceptance (SGLANG_SIMULATE_ACC_LEN) a greedy drafter's committed drafts
collapse into short repeating loops, which keeps MoE routing on a small repeated set and
makes the measured step time optimistic. A positive value makes the drafters that support
it (DSpark and the DFLASH family) sample every draft row at this temperature, e.g. 1000
(close to uniform), while the target still verifies greedily.
"""

from __future__ import annotations

import functools
import math
from typing import Callable, NamedTuple, Optional

import torch

from sglang.srt.environ import envs

ENV_NAME = "SGLANG_BENCH_DRAFTER_TEMPERATURE"
GREEDY_REQUEST_ERROR = f"{ENV_NAME} requires greedy requests (temperature 0 / top_k 1)."
SUPPORTED_ALGORITHMS = ("DFLASH", "DSPARK")
# The request clamp: below it the draft kernels' value / temperature overflows.
MIN_TEMPERATURE = 1e-5

# Philox domain of the DFLASH argmax-head noise (DSpark keeps its own draft key).
_HEAD_NOISE_DOMAIN = 0x6B6E6F6973652D31
# Counter words separating the head's two noise streams (draft graph, eager fallback).
GRAPH_STEP, EAGER_STEP = 0, 1


@functools.cache
def bench_drafter_temperature() -> float:
    """SGLANG_BENCH_DRAFTER_TEMPERATURE, validated once per process; 0.0 when unset."""
    value = float(envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.get())
    if value != 0.0 and not (
        MIN_TEMPERATURE <= value <= torch.finfo(torch.float32).max
    ):
        raise ValueError(
            f"{ENV_NAME} must be 0 (disabled) or a float32 temperature of at least "
            f"{MIN_TEMPERATURE}, got {value!r}."
        )
    return value


def bench_drafter_enabled() -> bool:
    return bench_drafter_temperature() > 0.0


def validate_server_args(algorithm: Optional[str]) -> None:
    """Server-args gate, run on every launch before any weights load: a malformed
    value, an algorithm whose drafts would ignore the override, or a benchmark
    setting the drafters cannot honor fails here instead of mid-run."""
    if not bench_drafter_enabled():  # also validates the value on every server
        return
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"{ENV_NAME} needs --speculative-algorithm "
            f"{' / '.join(SUPPORTED_ALGORITHMS)}, got {algorithm}."
        )
    simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
    if not 0.0 < simulate_acc_len < math.inf:
        raise ValueError(
            f"{ENV_NAME} requires a finite SGLANG_SIMULATE_ACC_LEN > 0: real "
            "acceptance would reject the drafts it samples."
        )
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
        raise ValueError(f"{ENV_NAME} requires SGLANG_RAGGED_VERIFY_MODE=static.")
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    platform_worker = SpeculativeAlgorithm.from_string(
        algorithm
    ).get_platform_worker_cls()
    if platform_worker is not None:
        raise ValueError(
            f"{ENV_NAME} is applied by the built-in {algorithm} worker only; this "
            f"platform serves {algorithm} with {platform_worker.__name__}."
        )


def require_greedy_requests(sampling_info) -> None:
    """A sampled request would be accepted against drafts drawn at a temperature it
    did not ask for; the benchmark keeps every request greedy (the scheduler rejects
    others at admission)."""
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


def bench_noise_key(key: int) -> int:
    """Fold the attention-DP rank into a Philox key: the TP ranks of one DP group agree
    on every draw, while DP groups, whose batch rows hold unrelated requests, draw
    independently instead of drafting (and committing) the same tokens."""
    from sglang.srt.runtime_context import get_parallel

    dp_rank = int(get_parallel().attn_dp_rank)
    return (int(key) ^ (dp_rank * 0x9E3779B97F4A7C15)) & 0x7FFFFFFFFFFFFFFF


class BenchHeadNoise(NamedTuple):
    """Gumbel-max sampling for an argmax draft head over the target's vocabulary."""

    key: int
    # The benchmark temperature over the target's transformed logits, folded onto the
    # head's raw logits: m * x at T is x at T / m, and softcap_c(m * x) is
    # m * softcap_{c / m}(x).
    temperature: float
    softcap: Optional[float]


def bench_head_noise(
    *,
    random_seed: int,
    output_multiplier: float = 1.0,
    final_logit_softcapping: Optional[float] = None,
) -> Optional[BenchHeadNoise]:
    temperature = bench_drafter_temperature()
    if temperature <= 0.0:
        return None
    return BenchHeadNoise(
        key=bench_noise_key(int(random_seed) ^ _HEAD_NOISE_DOMAIN),
        temperature=temperature / output_multiplier,
        softcap=(
            None
            if final_logit_softcapping is None
            else final_logit_softcapping / output_multiplier
        ),
    )


def gumbel_argmax(
    logits: torch.Tensor,
    *,
    noise: BenchHeadNoise,
    temperatures: torch.Tensor,
    offset: torch.Tensor,
    step: int,
    vocab_start: int,
    vocab_limit: int,
    gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Global token ids sampled from ``softmax(logits / T)`` with DSpark's fused
    Gumbel-max (``MarkovSampleStep``): per-block (max, id) candidates, never a
    full-vocab score tensor. ``logits`` is the rank's padded vocab shard starting at
    ``vocab_start``; ``vocab_limit`` masks padding ids; ``gather`` exchanges the
    candidates across TP ranks. The noise is keyed by (row, ``step``, ``offset``,
    global token id), so a shard draws exactly its slice of the full-vocab draw.
    ``temperatures`` holds ``noise.temperature`` per row; ``offset`` is a 1-element
    int64 device counter the caller advances once per draft call."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    sample = MarkovSampleStep.triton if logits.is_cuda else MarkovSampleStep.torch
    return sample(
        base_logits=logits,
        bias=None,
        temperatures=temperatures,
        key=noise.key,
        offset=offset,
        step=step,
        step_dtype=torch.float32,
        vocab_start=vocab_start,
        gather=gather,
        has_sampled_rows=True,
        softcap=noise.softcap,
        vocab_limit=vocab_limit,
    )

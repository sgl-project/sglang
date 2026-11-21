"""Draft temperature for DSpark and DFLASH-family drafters under simulated acceptance
(SGLANG_SIMULATE_ACC_LEN).

Forced acceptance commits whatever the drafter proposes. Greedy drafts collapse into
short repeating loops (DFLASH's default token mode even commits one fixed id), so the
model runs on a few repeated token ids, unlike real traffic, which can skew expert
routing and cache locality. Whenever simulated acceptance commits drafts, greedy draft
rows therefore sample at 1000 (close to uniform) by default, while the target still
verifies greedily. SGLANG_BENCH_DRAFTER_TEMPERATURE sets the temperature; 0 keeps
greedy drafts. Rows that already sample their drafts keep their own temperature;
plain DFLASH's argmax head drafts every row greedily, so all of its rows sample.
"""

from __future__ import annotations

import functools
import math
from typing import Callable, NamedTuple, Optional

import torch

from sglang.srt.environ import envs

ENV_NAME = "SGLANG_BENCH_DRAFTER_TEMPERATURE"
SUPPORTED_ALGORITHMS = ("DFLASH", "DSPARK")
# The draft temperature under simulated acceptance when the variable is unset.
DEFAULT_TEMPERATURE = 1000.0
# The request clamp: below it the draft kernels' value / temperature overflows.
MIN_TEMPERATURE = 1e-5

# Philox domain of the DFLASH argmax-head noise (DSpark keeps its own draft key).
_HEAD_NOISE_DOMAIN = 0x6B6E6F6973652D31
# Counter words separating the head's two noise streams (draft graph, eager fallback).
GRAPH_STEP, EAGER_STEP = 0, 1


def requested_temperature() -> Optional[float]:
    """SGLANG_BENCH_DRAFTER_TEMPERATURE as set, validated; None when unset."""
    value = envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.get()
    if value is None:
        return None
    value = float(value)
    if value != 0.0 and not (
        MIN_TEMPERATURE <= value <= torch.finfo(torch.float32).max
    ):
        raise ValueError(
            f"{ENV_NAME} must be 0 (greedy drafts) or a float32 temperature of at "
            f"least {MIN_TEMPERATURE}, got {value!r}."
        )
    return value


@functools.cache
def bench_drafter_temperature() -> float:
    """The draft temperature of greedy draft rows, read by the DSpark and DFLASH-family
    draft paths only: the variable when set, else DEFAULT_TEMPERATURE whenever
    simulated acceptance commits drafts (SGLANG_SIMULATE_ACC_LEN > 1), else 0."""
    requested = requested_temperature()
    if requested is not None:
        return requested
    if float(envs.SGLANG_SIMULATE_ACC_LEN.get()) > 1.0:
        return DEFAULT_TEMPERATURE
    return 0.0


def bench_drafter_enabled() -> bool:
    return bench_drafter_temperature() > 0.0


def opt_out_hint() -> str:
    """Appended to refusals of the default, which the user did not ask for."""
    if requested_temperature() is not None:
        return ""
    return (
        f" ({ENV_NAME} defaults to {DEFAULT_TEMPERATURE:g} under "
        f"SGLANG_SIMULATE_ACC_LEN; set {ENV_NAME}=0 to keep greedy drafts)"
    )


def validate_server_args(algorithm: Optional[str]) -> None:
    """Server-args gate, run on every launch before any weights load: an out-of-range
    value, an explicit value on an algorithm whose drafts would ignore it, or one
    without forced acceptance fails here instead of mid-run. (A value that is not a
    float warns and falls back to the default, like any SGLang env var.)"""
    bench_drafter_temperature.cache_clear()  # this launch's variables, not a cached one
    requested = requested_temperature()  # validates the value on every server
    if algorithm not in SUPPORTED_ALGORITHMS:
        if requested:
            raise ValueError(
                f"{ENV_NAME} applies to --speculative-algorithm "
                f"{' / '.join(SUPPORTED_ALGORITHMS)}, not {algorithm}."
            )
        return
    if not bench_drafter_enabled():
        return
    # The default needs SGLANG_SIMULATE_ACC_LEN > 1; so does an explicit value: real
    # acceptance would reject the sampled drafts, and a length of 1 commits none.
    # (The DSpark and DFLASH workers refuse a verify schedule that is not verify-all.)
    simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
    if not 1.0 < simulate_acc_len < math.inf:
        raise ValueError(
            f"{ENV_NAME} requires forced acceptance (a finite SGLANG_SIMULATE_ACC_LEN "
            f"> 1), got SGLANG_SIMULATE_ACC_LEN={simulate_acc_len:g}."
        )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    platform_worker = SpeculativeAlgorithm.from_string(
        algorithm
    ).get_platform_worker_cls()
    if platform_worker is not None:
        raise ValueError(
            f"{ENV_NAME} is applied by the built-in {algorithm} worker only; this "
            f"platform serves {algorithm} with {platform_worker.__name__}"
            f"{opt_out_hint()}."
        )


def override_draft_sampling(
    temperatures: torch.Tensor, greedy_mask: torch.Tensor
) -> None:
    """In place, when enabled: the given draft-only rows that would draft greedily
    sample at the benchmark temperature; sampled rows keep their own temperature.
    Callers pass the resolved ``[:bs]`` rows of buffers that only the draft proposal
    reads, never ones the target verify reads."""
    temperature = bench_drafter_temperature()
    if temperature <= 0.0:
        return
    temperatures.masked_fill_(greedy_mask, temperature)
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
    # Ids from here on are padding rows, which the target masks as well.
    vocab_limit: int


def bench_head_noise(
    *,
    random_seed: int,
    vocab_limit: int,
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
        vocab_limit=int(vocab_limit),
    )


def gumbel_argmax(
    logits: torch.Tensor,
    *,
    noise: BenchHeadNoise,
    temperatures: torch.Tensor,
    offset: torch.Tensor,
    step: int,
    vocab_start: int,
    gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Global token ids sampled from ``softmax(logits / T)`` with DSpark's fused
    Gumbel-max (``MarkovSampleStep``): per-block (max, id) candidates, never a
    full-vocab score tensor. ``logits`` is the rank's padded vocab shard starting at
    ``vocab_start``; ids from ``noise.vocab_limit`` on are masked; ``gather``
    exchanges the candidates across TP ranks. The noise is keyed by (row, ``step``,
    ``offset``, global token id), so a shard draws exactly its slice of the
    full-vocab draw. ``temperatures`` holds ``noise.temperature`` per row;
    ``offset`` is a 1-element int64 device counter the caller advances once per
    draft call."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    return MarkovSampleStep.execute(
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
        vocab_limit=noise.vocab_limit,
    )

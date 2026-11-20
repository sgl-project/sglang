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

import torch

from sglang.srt.environ import envs

ENV_NAME = "SGLANG_BENCH_DRAFTER_TEMPERATURE"
SUPPORTED_ALGORITHMS = ("DFLASH", "DSPARK")

_MASK32 = 0xFFFFFFFF
# Domain separation from the other draft/verify noise streams.
_NOISE_SALT = 0x5BD1E995
# Offset for callers with a second step counter (an eager fallback next to a CUDA
# graph's in-graph counter), so the two never reuse each other's noise.
EAGER_STEP_OFFSET = 1 << 31


@functools.cache
def bench_drafter_temperature() -> float:
    """SGLANG_BENCH_DRAFTER_TEMPERATURE, validated once per process; 0.0 when unset."""
    value = float(envs.SGLANG_BENCH_DRAFTER_TEMPERATURE.get())
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{ENV_NAME} must be a finite non-negative float (0 disables it), "
            f"got {value!r}."
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
        raise ValueError(
            f"{ENV_NAME} requires greedy requests (temperature 0 / top_k 1)."
        )


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


def _mul32(x: torch.Tensor, constant: int) -> torch.Tensor:
    """(x * constant) mod 2**32 for int64 ``x`` in [0, 2**32), without int64 overflow."""
    lo, hi = constant & 0xFFFF, constant >> 16
    return (x * lo + (((x * hi) & 0xFFFF) << 16)) & _MASK32


def _fmix32(h: torch.Tensor) -> torch.Tensor:
    """MurmurHash3's 32-bit finalizer on int64 tensors holding uint32 values."""
    h = h ^ (h >> 16)
    h = _mul32(h, 0x85EBCA6B)
    h = h ^ (h >> 13)
    h = _mul32(h, 0xC2B2AE35)
    return h ^ (h >> 16)


def gumbel_draft_scores(
    logits: torch.Tensor,
    *,
    col_start: int,
    row_start: int,
    step: torch.Tensor,
) -> torch.Tensor:
    """``logits / T`` plus Gumbel noise, as float32; an argmax over the result samples
    ``softmax(logits / T)`` (Gumbel-max). Identity when the override is off.

    The noise is a counter-based hash of (``step``, global row, global token id), so a
    TP vocab shard (``col_start`` = its first token id) draws independent noise from the
    other shards while every rank agrees on each token's draw, and a per-shard max then
    all-gather max stays an exact sample. ``step`` is a 0-d int64 tensor (an in-graph
    counter) or an int that the caller advances once per draft call; plain torch ops
    keep this CUDA-graph capturable.
    """
    temperature = bench_drafter_temperature()
    if temperature <= 0.0:
        return logits
    rows, cols = logits.shape
    device = logits.device
    row_ids = torch.arange(
        row_start, row_start + rows, dtype=torch.int64, device=device
    )
    col_ids = torch.arange(
        col_start, col_start + cols, dtype=torch.int64, device=device
    )
    step_key = _fmix32((step + _NOISE_SALT) & _MASK32)
    row_key = _fmix32((_mul32(row_ids, 0x9E3779B1) ^ step_key) & _MASK32)
    h = _fmix32(
        (row_key.view(-1, 1) ^ _mul32(col_ids, 0x85EBCA77).view(1, -1)) & _MASK32
    )
    h = _fmix32((h + row_key.view(-1, 1)) & _MASK32)
    # The top 24 bits are exact in float32, so u stays strictly inside (0, 1);
    # -log(u) ~ Exp(1) and -log(-log(u)) ~ Gumbel(0, 1).
    u = (h >> 8).to(torch.float32).add_(0.5).div_(2.0**24)
    gumbel = u.log_().neg_().log_().neg_()
    return gumbel.add_(logits.float() / temperature)

"""Timing-only same-service A/B of the decode graph padding (SGLANG_TRTL17_PAD_AB, default off).

SGLANG_TRTL17_PAD_AB is a comma list of labels, one per block of Q35MF_AB_BLOCK (default 200)
target-verify rounds, in turn: "pow2" pads every decode-phase graph runner's batch to the
smallest captured power of two, any other label keeps the smallest captured bucket. Every rank
counts the same rounds, so all ranks replay the same graphs. The round timer
(Q35MF_ROUNDTIMER_DIR) tags each round with current() / round_index().
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from sglang.srt.environ import envs

_LABELS = tuple(envs.SGLANG_TRTL17_PAD_AB.get())
# the bench harness's block (ll16_bench_seed.py reads the same variable from the server env)
_BLOCK = int(os.getenv("Q35MF_AB_BLOCK") or 200)
ACTIVE = bool(_LABELS)
_round = 0


def enabled() -> bool:
    return ACTIVE


def round_index() -> int:
    return _round


def current() -> str:
    return _LABELS[(_round // _BLOCK) % len(_LABELS)]


def tick() -> None:
    """One target-verify round starts (every rank)."""
    global _round
    _round += 1


def pad(raw_size: int, buckets: Sequence[int]) -> Optional[int]:
    """The bucket of a "pow2" round, or None for the default; only decode batch-size
    lists (which hold 1 and 2), not token-count buckets."""
    if current() != "pow2" or buckets[0] != 1 or 2 not in buckets:
        return None
    return next(
        (b for b in buckets if b >= raw_size and b & (b - 1) == 0), None
    )

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fail-stop deadlines (RFC #43949 section 7), the host half: every wait on another workgroup or GPU gives up after
half of ``--watchdog-timeout``, at most 60 s, and a deadline under 5 s is refused at startup. The device half
(``device.guard``) turns a missed deadline into a stop of every rank.

Each rank owns GUARD_WORDS int32 words in its peer-symmetric buffer:
  [STOP]  set to 1 by any rank that stopped (written into every rank's words)
  [ERROR] sticky reason bits of this rank's own stop (1: deadline, 2: a peer stopped)
  [SITE]  the wait that stopped first on this rank (its region's site id)
  [ZERO]  always 0: the address the stopping store goes to
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Optional

GUARD_WORDS = 16
GUARD_BYTES = 4 * GUARD_WORDS
STOP, ERROR, SITE, ZERO = 0, 1, 2, 3
REASON_DEADLINE, REASON_PEER = 1, 2

MAX_DEADLINE_S = 60.0
MIN_DEADLINE_S = 5.0
TICKS_PER_S = 100_000_000  # s_memrealtime


def deadline_s(watchdog_timeout: Optional[float]) -> float:
    if not watchdog_timeout or watchdog_timeout <= 0:
        return MAX_DEADLINE_S
    return min(float(watchdog_timeout) / 2, MAX_DEADLINE_S)


def deadline_ticks(watchdog_timeout: Optional[float]) -> int:
    return int(deadline_s(watchdog_timeout) * TICKS_PER_S)


def deadline_refusals(watchdog_timeout: Optional[float]) -> list[str]:
    s = deadline_s(watchdog_timeout)
    if s < MIN_DEADLINE_S:
        return [
            f"--watchdog-timeout {watchdog_timeout} gives a {s:g} s wait deadline; monokernels need at least "
            f"{MIN_DEADLINE_S:g} s (a watchdog of {2 * MIN_DEADLINE_S:g} s or more)"
        ]
    return []


def site_id(region: str) -> int:
    """A wait site's id in the SITE word: its region name's crc, never 0."""
    return (zlib.crc32(region.encode()) & 0x7FFF) | 0x8000


FAULT_ENV = "SGLANG_TEST_MONO_FAULT_INJECT"

# the SGLang names this module reads (pinned by test_mono_contract)
SGLANG_READS = ("sglang.srt.environ.envs",)


@dataclass(frozen=True)
class Fault:
    """``SGLANG_TEST_MONO_FAULT_INJECT=rank:site[:after]`` (tests only): on ``rank``, the waits on region ``site``
    never see their data from launch pair ``after`` on (a pair = one layer's launches, counted from startup)."""

    rank: int
    site: str
    after: int = 0


def fault_from_env() -> Optional[Fault]:
    from sglang.srt.environ import envs

    spec = envs.SGLANG_TEST_MONO_FAULT_INJECT.get() or ""
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not parts[1]:
        raise ValueError(f"{FAULT_ENV}={spec!r}: want rank:site[:after]")
    return Fault(int(parts[0]), parts[1], int(parts[2]) if len(parts) == 3 else 0)


def fault_for_rank(rank: int) -> tuple[str, int]:
    """(site, after) for this rank's builds: ("", 0) when nothing is injected here."""
    f = fault_from_env()
    if f is None or f.rank != rank:
        return "", 0
    return f.site, f.after

# SPDX-License-Identifier: Apache-2.0
"""Sizing of the HiCache host Mamba pool: ``--hicache-mamba-size-gb``.

Pure arithmetic, importable without torch, so the split rule and its boot-time
coverage report can be unit-tested on any machine.

Background. A hybrid (attention + Mamba/KDA) model backs up two things per
cached prefix into host memory: the attention KV rows of every token, and one
Mamba state checkpoint per ``--chunked-prefill-size`` tokens plus one at the
end of each request. ``build_hybrid_mamba_stack`` splits the fixed
``--hicache-size`` budget between the two host pools in proportion to their
*device* byte sizes (``_split_hicache_size``). The device Mamba pool is small
(a few hundred slots, sized for running requests), so the host Mamba pool ends
up with far fewer checkpoint slots than the KV host tier needs anchors, and
long prefixes whose KV is still resident on host cannot be restored because
their last checkpoint was evicted.

The knob has three forms:

* ``None`` (default): today's byte-proportional split, unchanged.
* a number of gigabytes: that much host memory for Mamba checkpoints, the rest
  of ``--hicache-size`` for KV.
* ``"auto"``: enough slots for one checkpoint per chunk of the KV host tier
  plus ``FINISH_SLOTS_PER_RUNNING_REQUEST`` per running request, solved as a
  fixed point since the KV share shrinks as the Mamba share grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

AUTO = "auto"

# Finish checkpoints (and the states a request holds while it runs) that must
# stay allocatable on top of the one-per-chunk demand of the KV host tier.
FINISH_SLOTS_PER_RUNNING_REQUEST = 4

# Kept aligned with the pool constructors: MambaPoolHost and HostKVCache both
# turn a GB figure into slots with ``int(host_size * 1e9 // size_per_token)``.
_BYTES_PER_GB = 1e9

MambaHostSizeKnob = Union[None, float, str]


def parse_mamba_host_size(value: object) -> MambaHostSizeKnob:
    """Parse the ``--hicache-mamba-size-gb`` value.

    Returns ``None`` (unset), a positive float (gigabytes) or ``"auto"``.
    Raises ``ValueError`` for anything else.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(_bad_value(value))
    if isinstance(value, (int, float)):
        gb = float(value)
    else:
        text = str(value).strip().lower()
        if text == AUTO:
            return AUTO
        try:
            gb = float(text)
        except ValueError:
            raise ValueError(_bad_value(value)) from None
    if not math.isfinite(gb) or gb <= 0:
        raise ValueError(_bad_value(value))
    return gb


def _bad_value(value: object) -> str:
    return (
        "--hicache-mamba-size-gb must be a positive number of gigabytes or "
        f"'auto', got {value!r}"
    )


def validate_mamba_host_size_knob(
    value: object, hicache_size_gb: float
) -> MambaHostSizeKnob:
    """Server-argument rule: parse the knob and check it against --hicache-size.

    The knob carves the Mamba share out of the fixed ``--hicache-size`` budget,
    so it needs that budget (ratio-sized host pools have no fixed budget to
    split) and an explicit figure must leave the KV host pool a positive share.
    """
    knob = parse_mamba_host_size(value)
    if knob is None:
        return None
    if hicache_size_gb is None or hicache_size_gb <= 0:
        raise ValueError(
            "--hicache-mamba-size-gb requires --hicache-size > 0: it sets the "
            "Mamba share of that fixed host budget. With --hicache-ratio the "
            "host pools are sized from the device pools instead."
        )
    if knob != AUTO and knob >= hicache_size_gb:
        raise ValueError(
            f"--hicache-mamba-size-gb {knob:g} must be smaller than "
            f"--hicache-size {hicache_size_gb:g} so the KV host pool keeps a "
            "positive share of the budget."
        )
    return knob


def proportional_split(
    hicache_size_gb: float, device_pool_bytes: Sequence[int]
) -> tuple[float, ...]:
    """Byte-proportional split, the same arithmetic as ``_split_hicache_size``."""
    total = sum(device_pool_bytes)
    return tuple(
        hicache_size_gb * size_bytes / total for size_bytes in device_pool_bytes
    )


def host_slots(size_gb: float, bytes_per_slot: float) -> int:
    """Slots a host pool gets for ``size_gb`` (the pool constructors' formula)."""
    return int(size_gb * _BYTES_PER_GB // bytes_per_slot)


def all_sizes_or_none(values: Sequence[object]) -> bool:
    """True when every value is an ``int`` or ``None``: the inputs the coverage
    report can take.

    Pool assembly also runs under tests that pass MagicMock params and patch the
    host pool classes; ``checkpoint_demand`` would then compare a mock with 0
    and raise. The
    coverage boot line is informational and must never break pool assembly, so
    ``_log_mamba_host_coverage`` skips it unless this predicate holds.
    """
    return all(value is None or isinstance(value, int) for value in values)


def checkpoint_demand(
    kv_tokens: int,
    chunked_prefill_size: Optional[int],
    max_running_requests: Optional[int],
) -> int:
    """Host Mamba slots needed to anchor every token of a KV host tier.

    One checkpoint per prefill chunk of the tier, plus
    ``FINISH_SLOTS_PER_RUNNING_REQUEST`` per running request for finish
    checkpoints and in-flight states. Without chunked prefill only the
    per-request term is known.
    """
    per_chunk = 0
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        per_chunk = math.ceil(kv_tokens / chunked_prefill_size)
    running = max(int(max_running_requests or 0), 0)
    return per_chunk + FINISH_SLOTS_PER_RUNNING_REQUEST * running


@dataclass(frozen=True)
class MambaHostCoverage:
    """How far a host Mamba pool covers the checkpoint demand of a KV host tier."""

    kv_tokens: int
    slots: int
    chunked_prefill_size: Optional[int]
    max_running_requests: int
    demanded_slots: int
    coverage: float
    # Tokens anchorable at one checkpoint per chunk once the per-request
    # finish slots are set aside; 0 when chunked prefill is off.
    covered_tokens: int

    @property
    def is_full(self) -> bool:
        return self.coverage >= 1.0

    def boot_line(self) -> str:
        if self.chunked_prefill_size is None or self.chunked_prefill_size <= 0:
            return (
                f"host mamba slots {self.slots} (chunked prefill off: one "
                f"finish checkpoint per request, {self.demanded_slots} slots "
                f"reserved for {self.max_running_requests} running requests); "
                f"KV host tier {self.kv_tokens} tokens "
                f"(coverage {self.coverage * 100:.0f}%)"
            )
        return (
            f"host mamba slots {self.slots} cover {self.covered_tokens} tokens "
            f"at one checkpoint per {self.chunked_prefill_size}-token chunk; "
            f"KV host tier {self.kv_tokens} tokens "
            f"(coverage {self.coverage * 100:.0f}%)"
        )


def mamba_host_coverage(
    kv_tokens: int,
    slots: int,
    chunked_prefill_size: Optional[int],
    max_running_requests: Optional[int],
) -> MambaHostCoverage:
    running = max(int(max_running_requests or 0), 0)
    demanded = checkpoint_demand(kv_tokens, chunked_prefill_size, running)
    coverage = slots / demanded if demanded > 0 else 1.0
    covered_tokens = 0
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        spare = max(slots - FINISH_SLOTS_PER_RUNNING_REQUEST * running, 0)
        covered_tokens = spare * chunked_prefill_size
    return MambaHostCoverage(
        kv_tokens=int(kv_tokens),
        slots=int(slots),
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=running,
        demanded_slots=demanded,
        coverage=coverage,
        covered_tokens=covered_tokens,
    )


@dataclass(frozen=True)
class MambaHostSplit:
    """The resolved split of ``--hicache-size`` between the KV and Mamba host pools."""

    mode: str  # "proportional" | "explicit" | "auto"
    hicache_size_gb: float
    kv_gb: float
    mamba_gb: float
    slots: int
    kv_tokens: int
    report: MambaHostCoverage

    @property
    def coverage(self) -> float:
        return self.report.coverage

    @property
    def demanded_slots(self) -> int:
        return self.report.demanded_slots

    @property
    def needs_warning(self) -> bool:
        return not self.report.is_full

    def boot_line(self) -> str:
        return (
            f"HiCache host split ({self.mode}, --hicache-size "
            f"{self.hicache_size_gb:g} GB): KV {self.kv_gb:.2f} GB, mamba "
            f"{self.mamba_gb:.2f} GB; {self.report.boot_line()}"
        )


def _solve_auto_split(
    hicache_size_gb: float,
    kv_bytes_per_token: float,
    mamba_bytes_per_slot: float,
    chunked_prefill_size: Optional[int],
    max_running_requests: int,
) -> tuple[float, float, int]:
    """Fixed point of ``slots = demand(kv_tokens(H - slots * bytes_per_slot))``.

    Pass 1 solves the continuous relaxation in closed form and rounds the
    demand up; pass 2 re-derives the KV share from what the rounded Mamba
    share leaves and keeps the larger demand, so coverage is never below 100%.
    """
    if chunked_prefill_size is None or chunked_prefill_size <= 0:
        raise ValueError(
            "--hicache-mamba-size-gb auto needs chunked prefill "
            "(--chunked-prefill-size > 0) to know the checkpoint cadence; pass "
            "an explicit number of gigabytes instead."
        )
    budget = hicache_size_gb * _BYTES_PER_GB
    finish_bytes = (
        FINISH_SLOTS_PER_RUNNING_REQUEST * max_running_requests * mamba_bytes_per_slot
    )
    if budget - finish_bytes <= 0:
        raise ValueError(
            f"--hicache-size {hicache_size_gb:g} GB cannot hold the "
            f"{FINISH_SLOTS_PER_RUNNING_REQUEST} x {max_running_requests} finish "
            f"checkpoints ({finish_bytes / _BYTES_PER_GB:.2f} GB) that "
            "--hicache-mamba-size-gb auto reserves; raise --hicache-size or "
            "pass an explicit number of gigabytes."
        )
    # Continuous relaxation: kv_bytes + slots(kv_bytes) * bps = budget with
    # slots = kv_bytes / (bpt * chunk) + finish_slots.
    per_token_overhead = mamba_bytes_per_slot / (
        kv_bytes_per_token * chunked_prefill_size
    )
    kv_bytes = (budget - finish_bytes) / (1.0 + per_token_overhead)
    slots = checkpoint_demand(
        int(kv_bytes // kv_bytes_per_token), chunked_prefill_size, max_running_requests
    )
    # Second pass on the discrete remainder.
    kv_tokens_after = int((budget - slots * mamba_bytes_per_slot) // kv_bytes_per_token)
    slots = max(
        slots,
        checkpoint_demand(kv_tokens_after, chunked_prefill_size, max_running_requests),
    )
    # +1 byte so the pool constructor's floor division cannot lose a slot to
    # floating-point rounding of the GB figure.
    mamba_gb = (slots * mamba_bytes_per_slot + 1) / _BYTES_PER_GB
    kv_gb = hicache_size_gb - mamba_gb
    if kv_gb <= 0:
        raise ValueError(
            f"--hicache-mamba-size-gb auto leaves no KV share of --hicache-size "
            f"{hicache_size_gb:g} GB ({slots} checkpoint slots need "
            f"{mamba_gb:.2f} GB); raise --hicache-size."
        )
    return kv_gb, mamba_gb, slots


def resolve_mamba_host_split(
    hicache_size_gb: float,
    knob: MambaHostSizeKnob,
    kv_bytes_per_token: float,
    mamba_bytes_per_slot: float,
    device_kv_bytes: int,
    device_mamba_bytes: int,
    chunked_prefill_size: Optional[int],
    max_running_requests: Optional[int],
) -> MambaHostSplit:
    """Split ``--hicache-size`` between the KV and Mamba host pools.

    ``knob`` is the parsed ``--hicache-mamba-size-gb``: ``None`` reproduces
    the byte-proportional split exactly, a float reserves that many GB for
    Mamba, ``"auto"`` sizes Mamba to the KV tier's checkpoint demand.
    ``kv_bytes_per_token`` covers every non-Mamba host pool together (full
    attention plus SWA when present); ``device_*_bytes`` feed the proportional
    default only.
    """
    if hicache_size_gb <= 0:
        raise ValueError("resolve_mamba_host_split needs --hicache-size > 0")
    if kv_bytes_per_token <= 0 or mamba_bytes_per_slot <= 0:
        raise ValueError("bytes per token / per slot must be positive")
    running = max(int(max_running_requests or 0), 0)

    if knob is None:
        mode = "proportional"
        kv_gb, mamba_gb = proportional_split(
            hicache_size_gb, (device_kv_bytes, device_mamba_bytes)
        )
    elif knob == AUTO:
        mode = AUTO
        kv_gb, mamba_gb, _ = _solve_auto_split(
            hicache_size_gb,
            kv_bytes_per_token,
            mamba_bytes_per_slot,
            chunked_prefill_size,
            running,
        )
    else:
        mode = "explicit"
        mamba_gb = float(knob)
        if mamba_gb >= hicache_size_gb:
            raise ValueError(
                f"--hicache-mamba-size-gb {mamba_gb:g} must be smaller than "
                f"--hicache-size {hicache_size_gb:g}"
            )
        kv_gb = hicache_size_gb - mamba_gb

    slots = host_slots(mamba_gb, mamba_bytes_per_slot)
    kv_tokens = host_slots(kv_gb, kv_bytes_per_token)
    report = mamba_host_coverage(kv_tokens, slots, chunked_prefill_size, running)
    return MambaHostSplit(
        mode=mode,
        hicache_size_gb=float(hicache_size_gb),
        kv_gb=kv_gb,
        mamba_gb=mamba_gb,
        slots=slots,
        kv_tokens=kv_tokens,
        report=report,
    )

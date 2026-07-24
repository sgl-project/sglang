"""Kernel value/index validity checks — the kernel-check model.

Each check has two layers:

  * data layer (sanitize / containment): unconditional, branchless, ~free.
    Belongs to correctness / memory-safety; NOT gated. It may live in `recover`
    here, or inside the kernel itself (then declare `recover=None`, signal-only).
  * signal layer (detect + log/crash): the costly reduction + report. Gated by
    SGLANG_KERNEL_CHECK (off / warn / strict).

A `Bucket` classifies an invariant by blast radius and recoverability; the
(bucket x level) matrix decides whether a hit crashes, logs, or is silent.
Invariants are declared once as module-level `Invariant` objects (registering
themselves for the CI coverage meta-test) and passed to `expect` at each site.

Detection stays async (no GPU-CPU sync): violations surface via torch's async
assert at the next sync point, and counts via a pinned-memory readback.
"""

from __future__ import annotations

import enum
import logging
from typing import Callable, Optional

import torch

from sglang.srt.environ import KernelCheckLevel, envs

logger = logging.getLogger(__name__)


class Bucket(enum.Enum):
    """Invariant classification (see RFC decision tree)."""

    SOFTEN = "soften"  # recoverable + legitimate event (never a bug)
    GUARD = "guard"  # recoverable + is-a-bug / root cause unknown
    FATAL_CONTAINABLE = "fatal_containable"  # no correct fallback; cheap containment
    FATAL_UNCONTAINABLE = "fatal_uncontainable"  # no containment; global corruption


def resolve_level() -> KernelCheckLevel:
    """Current signal level, bridging the legacy ASYNC_ASSERT flag.

    Until every callsite migrates, an unset SGLANG_KERNEL_CHECK inherits
    SGLANG_ENABLE_ASYNC_ASSERT=true as STRICT so CI keeps failing loud.
    """
    if envs.SGLANG_KERNEL_CHECK.is_set():
        return KernelCheckLevel(envs.SGLANG_KERNEL_CHECK.get())
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return KernelCheckLevel.STRICT
    return KernelCheckLevel.OFF


class Property:
    """A per-element validity predicate over a tensor. Detection only."""

    name: str

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        """Elementwise bool, True where the value is fine (no reduction here)."""
        raise NotImplementedError


class NotNaN(Property):
    """Not NaN, tolerating +-Inf (e.g. legitimately masked -inf logits)."""

    name = "not_nan"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return ~torch.isnan(value)


class NotInf(Property):
    """Not +-Inf (e.g. fp16 overflow), tolerating NaN."""

    name = "not_inf"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return ~torch.isinf(value)


class Finite(Property):
    """Neither NaN nor Inf — use when inf is also a bug for this tensor."""

    name = "finite"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(value)


class InRange(Property):
    """Half-open [lo, hi) — unifies oob / range / index-domain checks."""

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.name = f"in_range[{lo},{hi})"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return (value >= self.lo) & (value < self.hi)


class PageAligned(Property):
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.name = f"page_aligned[{page_size}]"

    def ok(self, value: torch.Tensor) -> torch.Tensor:
        return value % self.page_size == 0


# name -> Invariant; enumerated by the CI meta-test to enforce injection coverage.
_REGISTRY: dict[str, Invariant] = {}


class Invariant:
    """A declared kernel-check invariant. Constructing one registers it.

    `recover` is the optional python-side data layer (branchless, idempotent on
    clean input). Leave it None when the data layer lives inside the kernel;
    the call is then signal-only. FATAL_UNCONTAINABLE must not have a recover.
    """

    def __init__(
        self,
        name: str,
        bucket: Bucket,
        prop: Property,
        *,
        recover: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if bucket is Bucket.FATAL_UNCONTAINABLE and recover is not None:
            raise ValueError(f"uncontainable FATAL {name!r} cannot have a recover")
        if name in _REGISTRY:
            raise ValueError(f"duplicate kernel-check invariant {name!r}")
        self.name = name
        self.bucket = bucket
        self.prop = prop
        self.recover = recover
        _REGISTRY[name] = self


def registered_invariants() -> dict[str, Invariant]:
    return dict(_REGISTRY)


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


# Per-(rank, name) checks between aggregated log flushes.
_FLUSH_EVERY = 512


class _CheckReporter:
    """Sync-free per-(rank, name) hit counter that logs on a call cadence.

    Detection is accumulated on the device and mirrored to pinned host memory
    with a non-blocking copy; the host reads the (slightly stale) count on a
    later call. First hit logs immediately; subsequent hits are batched.
    """

    def __init__(self):
        self._dev: dict[str, torch.Tensor] = {}
        self._host: dict[str, torch.Tensor] = {}
        self._calls: dict[str, int] = {}
        self._logged_total: dict[str, int] = {}

    def record(self, key: str, bucket: Bucket, hit_count: torch.Tensor, msg: str):
        dev = self._dev.get(key)
        if dev is None:
            dev = torch.zeros(1, dtype=torch.int64, device=hit_count.device)
            self._dev[key] = dev
            self._host[key] = torch.zeros(
                1, dtype=torch.int64, pin_memory=hit_count.is_cuda
            )
            self._calls[key] = 0
            self._logged_total[key] = 0

        dev.add_(hit_count.reshape(1).to(torch.int64))
        self._host[key].copy_(dev, non_blocking=True)  # async, no sync
        self._calls[key] += 1

        total = int(self._host[key][0])  # stale host read, no sync
        last = self._logged_total[key]
        first_hit = last == 0 and total > 0
        cadence = self._calls[key] % _FLUSH_EVERY == 0
        if total > last and (first_hit or cadence):
            level = logging.INFO if bucket is Bucket.SOFTEN else logging.WARNING
            logger.log(
                level,
                "kernel-check [%s]: +%d hit(s), %d total. %s",
                key,
                total - last,
                total,
                msg,
            )
            self._logged_total[key] = total


_reporter = _CheckReporter()


def _crashes(bucket: Bucket, level: KernelCheckLevel) -> bool:
    """The (bucket x level) crash decision — pure, matches the RFC matrix."""
    if bucket is Bucket.SOFTEN:
        return False
    if bucket is Bucket.FATAL_UNCONTAINABLE:
        return level >= KernelCheckLevel.WARN
    return level == KernelCheckLevel.STRICT  # GUARD, FATAL_CONTAINABLE


def _signal(ok: torch.Tensor, *, inv: Invariant, level: KernelCheckLevel, msg: str):
    if _crashes(inv.bucket, level):
        # Loud: async assert surfaces at the next sync point (no CPU sync).
        torch._assert_async(ok.all(), f"kernel-check FAILED [{inv.name}]: {msg}")
        return
    _reporter.record(f"{inv.name}@rank{_get_rank()}", inv.bucket, (~ok).sum(), msg)


def expect(
    inv: Invariant,
    value: Optional[torch.Tensor],
    *,
    msg: str = "",
) -> Optional[torch.Tensor]:
    """Assert `inv` holds over `value`, per the (bucket x level) matrix.

    The data layer (`inv.recover`) is applied unconditionally; only detection +
    reporting is gated by SGLANG_KERNEL_CHECK. Returns the recovered value.
    """
    level = resolve_level()
    if level >= KernelCheckLevel.WARN and value is not None and value.numel() > 0:
        detail = f"{inv.prop.name}: {msg}" if msg else inv.prop.name
        _signal(inv.prop.ok(value), inv=inv, level=level, msg=detail)

    if inv.recover is not None and value is not None:
        value = inv.recover(value)
    return value

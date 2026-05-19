from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class CanaryMode(str, enum.Enum):
    OFF = "off"
    LOG = "log"
    RAISE = "raise"

    @classmethod
    def parse(cls, value: str | "CanaryMode" | None) -> "CanaryMode":
        if value is None:
            return cls.OFF
        if isinstance(value, cls):
            return value
        return cls(value)


class RealKvHashMode(str, enum.Enum):
    """``--kv-cache-canary-real-data`` modes.

    Controls whether the canary kernel reads a portion of the real KV pool
    on each write and stores a splitmix64 fingerprint of it into the
    extra ``real_kv_hash`` slot field, then re-reads + re-hashes on
    verify and compares. Catches corruption that is invisible to the
    pure-canary path because both the canary read and write paths
    operate correctly but the **real KV** pool got written wrong
    (e.g. attn-kernel idle-logic misconfiguration; PD transfer bit-rot).

    Modes:

    - ``OFF`` (default): the feature is disabled; the kernel stores 0
      in the ``real_kv_hash`` field and skips the comparison.
    - ``BIT``: read 16 bytes of the real-KV slot at the configured
      pool layer and fold through splitmix64. Cheap default for
      always-on production.
    - ``ALL``: read the full real-KV slot stride and fold through
      splitmix64. Higher overhead, used when stronger evidence is
      needed (e.g. a confirmed corruption investigation).
    """

    OFF = "off"
    BIT = "bit"
    ALL = "all"

    @classmethod
    def parse(cls, value: str | "RealKvHashMode" | None) -> "RealKvHashMode":
        if value is None:
            return cls.OFF
        if isinstance(value, cls):
            return value
        return cls(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryConfig:
    mode: CanaryMode
    seed: int = 0xC0FFEE1234567890
    violation_ring_capacity: int = 256
    health_print_every_n_forwards: int = 1024
    counter_zero_warmup_forwards: int = 64
    perturb_req_to_token_prob: float = 0.0
    perturb_req_to_token_seed: int = 0
    # Sliding-window-attention window length, in tokens. Non-None ONLY for
    # canary runners attached to a SWA pool — SWA's req_to_token mapping
    # only addresses the most recent ``swa_window_size`` slots of a req
    # (older positions get evicted / overwritten), so the verify range
    # must be clipped to ``[K_req - swa_window_size, K_req)``. Reading
    # outside that window lands on slots that belong to other reqs and
    # would trip a (spurious) violation.
    swa_window_size: Optional[int] = None
    real_kv_hash_mode: RealKvHashMode = RealKvHashMode.OFF

    @classmethod
    def from_server_args(
        cls,
        mode: str | CanaryMode | None,
        real_kv_hash_mode: str | RealKvHashMode | None = None,
    ) -> "CanaryConfig":
        parsed = CanaryMode.parse(mode)
        raw_prob = envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB.get()
        perturb_prob = max(0.0, min(1.0, raw_prob))
        if perturb_prob != raw_prob:
            logger.warning(
                "kv-canary: SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB %f "
                "out of [0,1]; clamped to %f",
                raw_prob,
                perturb_prob,
            )
        return cls(
            mode=parsed,
            perturb_req_to_token_prob=perturb_prob,
            perturb_req_to_token_seed=envs.SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_SEED.get(),
            real_kv_hash_mode=RealKvHashMode.parse(real_kv_hash_mode),
        )

    @property
    def enabled(self) -> bool:
        return self.mode is not CanaryMode.OFF


def real_kv_hash_mode_to_int(mode: RealKvHashMode) -> int:
    """Map :class:`RealKvHashMode` to the kernel's int constants.

    Mirrors ``REAL_KV_HASH_MODE_*`` in ``jit_kernel/kv_cache_canary.py``
    and ``kRealKvHashMode*`` in ``canary.cuh``. Kept centralised here so
    the int wire format is in one place.
    """
    if mode is RealKvHashMode.OFF:
        return 0
    if mode is RealKvHashMode.BIT:
        return 1
    if mode is RealKvHashMode.ALL:
        return 2
    raise ValueError(f"kv-canary: unknown RealKvHashMode {mode!r}")


def real_kv_hash_read_bytes(
    mode: RealKvHashMode, real_kv_slot_stride_bytes: int
) -> int:
    """Return how many real-KV bytes the kernel reads per slot for ``mode``.

    - ``OFF`` -> 0 (kernel takes the early-out path).
    - ``BIT`` -> 16 bytes (a fixed cheap prefix; matches
      ``REAL_KV_HASH_BIT_BYTES``).
    - ``ALL`` -> ``real_kv_slot_stride_bytes`` (the full slot).
    """
    if mode is RealKvHashMode.OFF:
        return 0
    if mode is RealKvHashMode.BIT:
        return min(16, int(real_kv_slot_stride_bytes))
    if mode is RealKvHashMode.ALL:
        return int(real_kv_slot_stride_bytes)
    raise ValueError(f"kv-canary: unknown RealKvHashMode {mode!r}")

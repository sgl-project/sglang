"""Bounded, no-eviction shape cache for whole-role graphs."""

from __future__ import annotations

from enum import Enum
from typing import Any

import msgspec

from sglang.srt.model_executor.cuda_graph_config import pad_to_capture_size

from .config import AFDConfig
from .contracts import (
    AFDError,
    AFDPairedShape,
    AFDReason,
    AFDShapeIdentity,
)


class AFDBucketPhase(str, Enum):
    ARMING = "arming"
    INSTALLED = "installed"
    TERMINAL_EAGER = "terminal_eager"
    CLOSED = "closed"


class AFDBucketUsage(msgspec.Struct, kw_only=True):
    arms: int = 0
    installs: int = 0
    replays: int = 0
    rollbacks: int = 0
    eager: dict[str, int] = {}
    retained_hbm_bytes: int = 0


class AFDCacheSelection(msgspec.Struct, kw_only=True):
    bucket: Any | None
    reason: AFDReason | None
    replay: bool
    arming: bool


class AFDBucket:
    """One retained shape bucket. Buckets are never evicted or retried."""

    def __init__(self, *, shape: AFDShapeIdentity) -> None:
        self.shape = shape
        self.phase = AFDBucketPhase.ARMING
        # ARMING and INSTALLED share one owner; phase controls replay eligibility.
        self.program: Any | None = None
        self.usage = AFDBucketUsage()
        self.retained_backing_hbm_bytes = 0
        self.terminal_reason: AFDReason | None = None

    def record_eager(self, reason: AFDReason) -> None:
        key = reason.value
        self.usage.eager[key] = self.usage.eager.get(key, 0) + 1

    def release_resources(self) -> None:
        """Release this shape's program; the runtime owns the shared pool."""
        program, self.program = self.program, None
        try:
            if program is not None:
                program.close()
        finally:
            self.retained_backing_hbm_bytes = 0


def make_shape(
    *,
    lane: int,
    lane_rows: tuple[tuple[int, ...], ...],
    hidden_size: int,
    dtype: str,
    config: AFDConfig,
    capture_sizes: tuple[int, ...],
    tokens_per_request: int = 1,
) -> AFDPairedShape:
    if type(tokens_per_request) is not int or tokens_per_request != 1:
        raise AFDError("AFD_TOKENS_PER_REQUEST_INVALID")
    if (
        len(lane_rows) != config.attention_lane_count
        or lane < 0
        or lane >= max(config.attention_lane_count, config.lanes)
        or any(len(vector) != config.stages for vector in lane_rows)
        or hidden_size < 1
        or not dtype
    ):
        raise AFDError(
            "AFD_GRAPH_SHAPE_INVALID",
            f"lane={lane} rows={lane_rows!r} hidden={hidden_size} dtype={dtype!r}",
        )
    # One planned width across lanes/stages avoids a combinatorial graph cache.
    # Native sizes are per-lane request counts, mapped to ceil(bs / S2) at startup.
    max_rows = max(max(vector) for vector in lane_rows)
    width = (
        pad_to_capture_size(max_rows, capture_sizes)
        if capture_sizes and max_rows <= capture_sizes[-1]
        else None
    )
    return AFDPairedShape(
        lane=lane,
        lane_stage_rows=lane_rows,
        lane_bucket_rows=(
            ((width, width),) * len(lane_rows) if width is not None else lane_rows
        ),
        hidden_size=hidden_size,
        dtype=dtype,
        ffn_lanes=config.lanes,
        tokens_per_request=tokens_per_request,
    )


class AFDShapeCache:
    """Retain only configured shapes, without observation slots or eviction."""

    def __init__(
        self,
        *,
        config: AFDConfig,
        capture_sizes: tuple[int, ...],
        base_hbm_bytes: int = 0,
    ) -> None:
        if base_hbm_bytes < 0 or base_hbm_bytes > config.max_hbm_bytes:
            raise AFDError(
                "AFD_GRAPH_BASE_HBM_LIMIT",
                f"base={base_hbm_bytes} limit={config.max_hbm_bytes}",
            )
        self._config = config
        self._capture_sizes = capture_sizes
        self._buckets: dict[str, AFDBucket] = {}
        self._base_hbm_bytes = base_hbm_bytes
        self._retained_hbm_bytes = base_hbm_bytes
        self._capture_hbm_bytes = 0
        self._closed = False
        self._overflow_eager = 0
        self._sealed = False

    @property
    def capture_sizes(self) -> tuple[int, ...]:
        return self._capture_sizes

    def finish_capture(self) -> None:
        if self._sealed or self._closed:
            raise AFDError("AFD_GRAPH_STARTUP_ALREADY_FINISHED")
        if tuple(
            sorted(bucket.shape.lane_bucket_rows[0][0] for bucket in self.buckets)
        ) != self._capture_sizes or any(
            bucket.phase != AFDBucketPhase.INSTALLED for bucket in self.buckets
        ):
            raise AFDError("AFD_GRAPH_STARTUP_INCOMPLETE")
        self._sealed = True

    @property
    def retained_hbm_bytes(self) -> int:
        return self._retained_hbm_bytes

    @property
    def buckets(self) -> tuple[AFDBucket, ...]:
        return tuple(self._buckets.values())

    def select(
        self,
        *,
        shape: AFDShapeIdentity,
        estimated_hbm_bytes: int,
        capture: bool = False,
    ) -> AFDCacheSelection:
        if self._closed:
            return AFDCacheSelection(
                bucket=None,
                reason=AFDReason.CLOSED,
                replay=False,
                arming=False,
            )
        bucket = self._buckets.get(shape.digest)
        if capture and self._sealed:
            raise AFDError("AFD_GRAPH_RUNTIME_CAPTURE_FORBIDDEN")
        if bucket is None:
            width = shape.lane_bucket_rows[0][0]
            if (
                width not in self._capture_sizes
                or any(rows != (width, width) for rows in shape.lane_bucket_rows)
                or (capture and len(self._buckets) == len(self._capture_sizes))
            ):
                self._overflow_eager += 1
                return AFDCacheSelection(
                    bucket=None,
                    reason=AFDReason.BUCKET_LIMIT,
                    replay=False,
                    arming=False,
                )
            if not capture:
                if self._sealed:
                    raise AFDError("AFD_GRAPH_STARTUP_SHAPE_MISSING", shape.digest)
                return AFDCacheSelection(
                    bucket=None,
                    reason=AFDReason.FORWARD_MODE,
                    replay=False,
                    arming=False,
                )
            bucket = AFDBucket(shape=shape)
            self._buckets[shape.digest] = bucket
        elif bucket.phase == AFDBucketPhase.ARMING:
            raise AFDError("AFD_GRAPH_CONCURRENT_ARM_UNSUPPORTED", shape.digest)
        if bucket.phase == AFDBucketPhase.INSTALLED:
            return AFDCacheSelection(
                bucket=bucket,
                reason=None,
                replay=True,
                arming=False,
            )
        if bucket.phase == AFDBucketPhase.TERMINAL_EAGER:
            reason = bucket.terminal_reason or AFDReason.CAPTURE_FAILED
            bucket.record_eager(reason)
            return AFDCacheSelection(
                bucket=bucket,
                reason=reason,
                replay=False,
                arming=False,
            )
        if (
            estimated_hbm_bytes < 1
            or self._retained_hbm_bytes + estimated_hbm_bytes
            > self._config.max_hbm_bytes
        ):
            bucket.phase = AFDBucketPhase.TERMINAL_EAGER
            bucket.terminal_reason = AFDReason.HBM_LIMIT
            bucket.record_eager(AFDReason.HBM_LIMIT)
            return AFDCacheSelection(
                bucket=bucket,
                reason=AFDReason.HBM_LIMIT,
                replay=False,
                arming=False,
            )
        bucket.phase = AFDBucketPhase.ARMING
        bucket.usage.arms += 1
        bucket.usage.retained_hbm_bytes = estimated_hbm_bytes
        self._retained_hbm_bytes += estimated_hbm_bytes
        return AFDCacheSelection(
            bucket=bucket,
            reason=AFDReason.ARMING,
            replay=False,
            arming=True,
        )

    def install(
        self,
        *,
        bucket: AFDBucket,
    ) -> None:
        if bucket.phase != AFDBucketPhase.ARMING:
            raise AFDError(
                "AFD_GRAPH_INSTALL_PHASE_INVALID",
                f"phase={bucket.phase.value}",
            )
        if bucket.program is None:
            raise AFDError("AFD_GRAPH_PARTIAL_CAPTURE", "whole-step program missing")
        bucket.phase = AFDBucketPhase.INSTALLED
        bucket.usage.installs += 1

    def ensure_reservation(
        self,
        *,
        bucket: AFDBucket,
        retained_hbm_bytes: int,
    ) -> bool:
        if bucket.phase != AFDBucketPhase.ARMING:
            raise AFDError(
                "AFD_GRAPH_HBM_PHASE_INVALID",
                f"phase={bucket.phase.value}",
            )
        if retained_hbm_bytes <= bucket.usage.retained_hbm_bytes:
            return True
        growth = retained_hbm_bytes - bucket.usage.retained_hbm_bytes
        if self._retained_hbm_bytes + growth > self._config.max_hbm_bytes:
            return False
        bucket.usage.retained_hbm_bytes = retained_hbm_bytes
        self._retained_hbm_bytes += growth
        return True

    def account_capture(
        self,
        *,
        bucket: AFDBucket,
        retained_hbm_bytes: int,
        capture_growth_bytes: int,
    ) -> bool:
        if bucket.phase != AFDBucketPhase.ARMING:
            raise AFDError(
                "AFD_GRAPH_HBM_PHASE_INVALID",
                f"phase={bucket.phase.value}",
            )
        # Allocator retention belongs to the common capture owner, even after
        # a single shape is released. Count only each capture's new growth,
        # including its static inputs, never the shared pool total per bucket.
        if retained_hbm_bytes < 0 or capture_growth_bytes < 0:
            raise AFDError("AFD_GRAPH_HBM_NEGATIVE")
        self._capture_hbm_bytes += capture_growth_bytes
        self._retained_hbm_bytes += (
            capture_growth_bytes + retained_hbm_bytes - bucket.usage.retained_hbm_bytes
        )
        bucket.usage.retained_hbm_bytes = retained_hbm_bytes
        return self._retained_hbm_bytes <= self._config.max_hbm_bytes

    def rollback(
        self,
        *,
        bucket: AFDBucket,
        reason: AFDReason,
    ) -> None:
        if bucket.phase != AFDBucketPhase.ARMING:
            return
        try:
            bucket.release_resources()
        finally:
            self._retained_hbm_bytes -= bucket.usage.retained_hbm_bytes
            bucket.usage.retained_hbm_bytes = 0
            bucket.usage.rollbacks += 1
            bucket.phase = AFDBucketPhase.TERMINAL_EAGER
            bucket.terminal_reason = reason
            bucket.record_eager(reason)

    def invalidate(
        self,
        *,
        bucket: AFDBucket,
        reason: AFDReason,
    ) -> None:
        if bucket.phase == AFDBucketPhase.ARMING:
            self.rollback(bucket=bucket, reason=reason)
            return
        if bucket.phase != AFDBucketPhase.INSTALLED:
            return
        try:
            bucket.release_resources()
        finally:
            self._retained_hbm_bytes -= bucket.usage.retained_hbm_bytes
            bucket.usage.retained_hbm_bytes = 0
            bucket.usage.rollbacks += 1
            bucket.phase = AFDBucketPhase.TERMINAL_EAGER
            bucket.terminal_reason = reason
            bucket.record_eager(reason)

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.usage(status="CLOSED")
        self._closed = True
        first_error = None
        for bucket in self._buckets.values():
            try:
                bucket.release_resources()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                bucket.usage.retained_hbm_bytes = 0
                bucket.phase = AFDBucketPhase.CLOSED
        self._retained_hbm_bytes = 0
        self._capture_hbm_bytes = 0
        if first_error is not None:
            raise first_error
        return self.usage(status="CLOSED")

    def usage(self, *, status: str) -> dict[str, Any]:
        return {
            "schema": "afd-graph-usage-v2",
            "status": status,
            "max_buckets": len(self._capture_sizes),
            "max_hbm_bytes": self._config.max_hbm_bytes,
            "base_hbm_bytes": self._base_hbm_bytes,
            "capture_hbm_bytes": self._capture_hbm_bytes,
            "bucket_count": len(self._buckets),
            "overflow_eager": self._overflow_eager,
            "retained_hbm_bytes": self._retained_hbm_bytes,
            "buckets": {
                bucket.shape.digest: {
                    "phase": bucket.phase.value,
                    "shape": {
                        "stage_rows": bucket.shape.stage_rows,
                        "bucket_rows": bucket.shape.bucket_rows,
                        "capture_width": bucket.shape.lane_bucket_rows[0][0],
                        "hidden_size": bucket.shape.hidden_size,
                        "dtype": bucket.shape.dtype,
                    },
                    "usage": msgspec.to_builtins(bucket.usage),
                    "terminal_reason": (
                        bucket.terminal_reason.value
                        if bucket.terminal_reason is not None
                        else None
                    ),
                }
                for bucket in self._buckets.values()
            },
        }

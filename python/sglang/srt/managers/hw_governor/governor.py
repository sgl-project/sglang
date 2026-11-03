"""The hardware governor: bridges device telemetry to the scheduler's existing knobs."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from sglang.srt.managers.hw_governor.policy import (
    GovernorDecision,
    GovernorPolicy,
    resolve_policy_cls,
)
from sglang.srt.managers.hw_governor.telemetry import (
    DeviceTelemetry,
    TelemetrySource,
    resolve_telemetry_source_cls,
)
from sglang.srt.managers.io_struct import HwGovernorAdviceReq
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

logger = logging.getLogger(__name__)


class HardwareGovernor:
    """Resolves scheduler knobs under the latest hardware-governor advice.

    Only the leader rank reads telemetry: a background thread samples the
    source and runs the policy, so the scheduler loop never waits on the
    hardware. The leader sends each new decision as a ``HwGovernorAdviceReq``
    through the scheduler's request broadcast, and every rank applies it at the
    same point in its request stream, so all ranks resolve knobs identically.
    """

    def __init__(
        self,
        *,
        page_size: int,
        source: Optional[TelemetrySource],
        policy: Optional[GovernorPolicy],
        interval_s: float,
        metrics_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """``source`` and ``policy`` are set on the leader rank only."""
        self._page_size = page_size
        self._applied = GovernorDecision()
        # Written by the sampler thread; a single reference swap of a frozen struct.
        self._desired = self._applied
        self._source = source
        self._policy = policy
        self._interval_s = interval_s
        self._metrics = (
            _GovernorMetrics(metrics_labels)
            if metrics_labels is not None and source is not None
            else None
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if source is not None:
            self._thread = threading.Thread(
                target=self._run, name="hw-governor-sampler", daemon=True
            )
            self._thread.start()

    # ----- leader: sampling and advice -----

    def poll_advice(self) -> List[HwGovernorAdviceReq]:
        # Only the leader originates advice; a follower's desired decision never moves.
        if self._source is None:
            return []
        # Resent until applied: a skipped receive drops local requests,
        # and applying the same advice twice is a no-op.
        desired = self._desired
        if desired == self._applied:
            return []
        return [HwGovernorAdviceReq(decision=desired)]

    def sample_once(self) -> None:
        try:
            samples = self._source.sample()
        except Exception:
            logger.exception("Hardware governor: telemetry read failed")
            samples = []
        try:
            self._desired = self._policy.update(samples, now=time.monotonic())
        except Exception:
            logger.exception("Hardware governor: policy update failed")
        if self._metrics is not None:
            self._metrics.observe_samples(samples)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.sample_once()

    # ----- every rank: apply advice, resolve knobs -----

    def apply_advice(self, req: HwGovernorAdviceReq) -> None:
        if req.decision.level != self._applied.level:
            logger.info(
                "Hardware governor: level %d -> %d (%s)",
                self._applied.level,
                req.decision.level,
                req.decision.overrides.describe(),
            )
        self._applied = req.decision
        if self._metrics is not None:
            self._metrics.observe_level(req.decision.level)

    @property
    def level(self) -> int:
        return self._applied.level

    def chunked_prefill_size(self, configured: Optional[int]) -> Optional[int]:
        return self._applied.overrides.resolve_chunked_prefill_size(
            configured, page_size=self._page_size
        )

    def max_prefill_tokens(self, configured: int) -> int:
        return self._applied.overrides.resolve_max_prefill_tokens(
            configured, page_size=self._page_size
        )

    def max_running_requests(self, configured: int) -> int:
        return self._applied.overrides.resolve_max_running_requests(configured)

    def prefill_decode_interval(self, configured: int) -> int:
        return self._applied.overrides.resolve_prefill_decode_interval(configured)

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 1.0)
        if self._source is not None:
            self._source.close()


def create_hw_governor(
    *,
    page_size: int,
    device_id: int,
    tp_group: GroupCoordinator,
    source_spec: str,
    policy_spec: str,
    policy_config: Optional[Dict],
    interval_ms: float,
    metrics_collector: Optional[SchedulerMetricsCollector],
) -> HardwareGovernor:
    """Build the governor on every rank; the leader also gets the source and policy.

    Collective over ``tp_group``: every rank shares its device UUID so the
    leader can watch the whole group's GPUs.
    """
    device_uuid = current_platform.get_device_uuid(device_id)
    group_device_uuids = (
        tp_group.all_gather_object(device_uuid)
        if tp_group.world_size > 1
        else [device_uuid]
    )
    source = policy = None
    # Must match the rank that adds local requests in Scheduler.ingest_requests.
    parallel = get_parallel()
    if (
        parallel.pp_rank == 0
        and parallel.attn_tp_rank == 0
        and parallel.attn_cp_rank == 0
    ):
        policy = resolve_policy_cls(policy_spec)(policy_config)
        source = resolve_telemetry_source_cls(source_spec)(
            device_uuids=group_device_uuids
        )
        logger.info(
            "Hardware governor: source=%s policy=%s interval=%.0fms devices=%s",
            source_spec,
            policy_spec,
            interval_ms,
            list(source.device_uuids),
        )
    return HardwareGovernor(
        page_size=page_size,
        source=source,
        policy=policy,
        interval_s=interval_ms / 1000.0,
        metrics_labels=(
            metrics_collector.labels if metrics_collector is not None else None
        ),
    )


def validate_hw_governor_args(
    *, source_spec: str, policy_spec: str, policy_config: Optional[Dict]
) -> None:
    """Fail at startup, in the launching process, on an unknown source or a bad policy config."""
    resolve_telemetry_source_cls(source_spec)
    resolve_policy_cls(policy_spec)(policy_config)


class _GovernorMetrics:
    def __init__(self, labels: Dict[str, str]) -> None:
        # Imported late: prometheus_client must load after PROMETHEUS_MULTIPROC_DIR is set.
        from prometheus_client import Gauge

        self._labels = labels
        device_labels = [*labels.keys(), "gpu_uuid"]

        def gauge(name: str, documentation: str, labelnames) -> Gauge:
            return Gauge(
                name=name,
                documentation=documentation,
                labelnames=labelnames,
                multiprocess_mode="mostrecent",
            )

        self._temperature = gauge(
            "sglang:hw_gpu_temperature_celsius",
            "GPU temperature read by the hardware governor (in C).",
            device_labels,
        )
        self._power = gauge(
            "sglang:hw_gpu_power_watts",
            "GPU power draw read by the hardware governor (in W).",
            device_labels,
        )
        self._sm_clock = gauge(
            "sglang:hw_gpu_sm_clock_mhz",
            "GPU SM clock read by the hardware governor (in MHz).",
            device_labels,
        )
        self._clock_event_reasons = gauge(
            "sglang:hw_gpu_clock_event_reasons",
            "Bitmask of active clock event reasons (NVML nvmlClocksEventReason*).",
            device_labels,
        )
        self._level = gauge(
            "sglang:hw_governor_level",
            "Throttle level applied by the hardware governor (0 = no overrides).",
            list(labels.keys()),
        )

    def observe_samples(self, samples: Sequence[DeviceTelemetry]) -> None:
        for sample in samples:
            labels = {**self._labels, "gpu_uuid": sample.device_uuid}
            for gauge, value in (
                (self._temperature, sample.gpu_temp_c),
                (self._power, sample.power_w),
                (self._sm_clock, sample.sm_clock_mhz),
                (self._clock_event_reasons, sample.clock_event_reasons),
            ):
                if value is not None:
                    gauge.labels(**labels).set(value)

    def observe_level(self, level: int) -> None:
        self._level.labels(**self._labels).set(level)

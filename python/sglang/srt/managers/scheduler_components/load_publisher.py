"""Per-scheduler load reporting for load-aware routing.

Independent of KV-cache events: each scheduler publishes a periodic
[`LoadStat`] gauge on its own ZMQ PUB socket (a dedicated port range,
distinct from the KV-event publisher's) so load-aware routers — e.g. the
experimental sgl-router `cache_aware_zmq` policy — can route on the
engine's true queue depth / KV occupancy instead of inferring load from a
router-side in-flight counter.

The only thing borrowed is the generic ZMQ PUB transport
(`ZmqEventPublisher`) from `sglang.srt.utils.event_publisher`; the load wire
format and publishing cadence live here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec

from sglang.srt.utils.event_publisher import (
    KVEventsConfig,
    NullEventPublisher,
    ZmqEventPublisher,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState

logger = logging.getLogger(__name__)

# ZMQ topic the load publisher tags its frames with. The load socket carries
# only load, so subscribers can subscribe-all; the topic is cosmetic/self-
# documenting.
LOAD_TOPIC = "load"

# Publish a load snapshot at most once every this many `publish_load_stat`
# calls, unless `force=True` (extend/prefill batches, where load changes
# most). Load is a gauge consumed for routing, so per-decode-step publishing
# is wasteful.
LOAD_PUBLISH_INTERVAL = 5

# Re-warn about publish failures every this many consecutive failures, so a
# permanent failure (e.g. a renamed field) keeps a live breadcrumb instead of
# going silent after the first warning, without flooding the log.
LOAD_PUBLISH_FAIL_WARN_EVERY = 60


class LoadStat(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,  # type: ignore[call-arg]
):
    """Per-scheduler runtime load snapshot.

    Wire shape (tag + array_like): ``["LoadStat", num_running_reqs,
    num_waiting_reqs, num_tokens, max_total_num_tokens, attn_dp_rank?]`` —
    the router decoder reads the four counts and ignores the rest.
    `attn_dp_rank` exists so the snapshot can be published directly through
    `ZmqEventPublisher.publish` (which stamps it); the router keys load by
    the subscriber's socket rank, not this field.
    """

    num_running_reqs: int
    num_waiting_reqs: int
    # KV tokens currently in use, from the engine's KV pool.
    num_tokens: int
    # KV-cache token capacity; 0 when unknown.
    max_total_num_tokens: int
    attn_dp_rank: Optional[int] = None


@dataclass(kw_only=True, slots=True)
class SchedulerLoadPublisher:
    """Owns one scheduler's dedicated load PUB socket and the throttled,
    best-effort `publish_load_stat` path.

    Enabled on the same condition as KV-event publishing (a `kv_events_config`
    on the attn-TP/CP-rank-0 scheduler), and binds the load port range packed
    immediately after the KV-event range (`kv_base + dp_size`). Stays a no-op
    (a `NullEventPublisher`) when disabled or when the KV config has no usable
    ZMQ endpoint.
    """

    kv_events_config: Optional[str]
    ps: ParallelState
    # Number of attention-DP ranks (= the KV port range width); the load port
    # range starts at kv_base + dp_size.
    dp_size: int
    enable: bool = False
    publisher: Any = None
    _publish_counter: int = 0
    # Consecutive publish failures, reset on success (drives the periodic warn).
    _fail_count: int = 0

    def __post_init__(self) -> None:
        self.publisher = NullEventPublisher()
        self.enable = bool(
            self.kv_events_config
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        if not self.enable:
            return
        try:
            cfg = KVEventsConfig.from_cli(self.kv_events_config)
        except Exception:
            # Malformed config — the KV publisher init would have failed too;
            # stay a no-op rather than raising at scheduler startup.
            return
        if cfg.publisher == "null" or not cfg.endpoint:
            return
        load_endpoint = ZmqEventPublisher.offset_endpoint_port(
            cfg.endpoint, self.dp_size
        )
        if load_endpoint is None:
            return
        # Dedicated load socket: own port, replay disabled, unbuffered (load is
        # a gauge, not a replayable delta).
        self.publisher = ZmqEventPublisher(
            self.ps.attn_dp_rank,
            endpoint=load_endpoint,
            replay_endpoint=None,
            buffer_steps=0,
            topic=LOAD_TOPIC,
        )

    def publish_load_stat(self, load_provider: Callable, force: bool = False) -> None:
        """Publish a load snapshot, throttled to [`LOAD_PUBLISH_INTERVAL`]
        calls unless `force`.

        `load_provider` returns a live load snapshot (a `GetLoadsReqOutput`)
        read directly from scheduler state — used instead of metrics stats,
        whose values are only populated under `--enable-metrics`. Invoked only
        after the throttle passes, so the snapshot is computed only when
        actually publishing.

        Best-effort: a failure here must never crash the scheduler loop —
        routers fall back to their own in-flight counter. Failures re-warn
        every [`LOAD_PUBLISH_FAIL_WARN_EVERY`] consecutive failures.
        """
        if not self.enable:
            return

        self._publish_counter += 1
        if not force and self._publish_counter < LOAD_PUBLISH_INTERVAL:
            return
        self._publish_counter = 0

        try:
            load = load_provider()
            self.publisher.publish(
                LoadStat(
                    num_running_reqs=load.num_running_reqs,
                    num_waiting_reqs=load.num_waiting_reqs,
                    num_tokens=load.num_used_tokens,
                    max_total_num_tokens=load.max_total_num_tokens,
                )
            )
            self._fail_count = 0
        except Exception:
            if self._fail_count % LOAD_PUBLISH_FAIL_WARN_EVERY == 0:
                logger.warning(
                    "load-publisher: publish_load_stat failed (%d consecutive); "
                    "load-aware routers fall back to their in-flight load signal",
                    self._fail_count + 1,
                    exc_info=True,
                )
            self._fail_count += 1

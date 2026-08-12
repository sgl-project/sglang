"""Per-scheduler load reporting for load-aware routers.

Independent of KV-cache events: each scheduler publishes a periodic
[`LoadStat`] gauge on its own ZMQ PUB socket (a dedicated port range,
packed after the KV-event publisher's) so load-aware routers —
e.g. the experimental sgl-router `cache_aware_zmq` policy — can route on
the engine's true queue depth / KV occupancy instead of inferring load
from a router-side in-flight counter.

This is the *external* counterpart of `sglang.srt.managers.load_snapshot`:
that module fans per-rank `LoadSnapshot`s into shared memory (or a zmq
PUSH to node 0) for consumers inside the deployment — the
DataParallelController's dispatch and `/v1/loads`. Neither transport is
subscribable by an out-of-process router that only knows the worker's URL,
so load-aware routers get their own per-rank PUB socket instead, on a port
advertised via `/server_info` (see
`ServerArgs.describe_kv_events_publisher`). The payload is a compact,
tagged subset of the snapshot rather than the full `LoadSnapshot`, so the
router-facing wire contract stays fixed while the internal snapshot keeps
growing fields.

The transport (`ZmqEventPublisher`), config parsing, and rank/port
derivation are borrowed from `sglang.srt.disaggregation.kv_events`; the
load wire format and publishing cadence live here. The port itself comes
from `derive_load_port_base` — the same function `/server_info` advertises
with — which packs the load range after the KV-event range (skipping the
replay ROUTER range when one is configured). Operators must leave that
derived range free on the host: in particular, two workers sharing a host
need a `2 * dp_size` gap between their KV-event port bases, or one
worker's load socket lands on the other's KV-event port.

Other prior art, considered and not reused: `/v1/loads` serves the same
snapshot over HTTP, but polling it routes through the HTTP-plane event
loop, which lags exactly when the server is overloaded — the moment the
signal matters; `KvMetrics` (kv_events_publisher.py) pushes similar
counts, but over a point-to-point IPC socket gated on --enable-metrics
with a Dynamo-pinned schema; forward-pass metrics publish per-iteration
on a host-local, operator-configured endpoint. None is per-worker
subscribable via /server_info.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

import msgspec

from sglang.srt.disaggregation.kv_events import (
    LOAD_TOPIC,
    KVEventsConfig,
    ZmqEventPublisher,
    derive_load_port_base,
    is_kv_publisher_rank,
    select_kv_publisher_dp_rank,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.load_snapshot import LoadSnapshot

logger = logging.getLogger(__name__)

# Publish a load snapshot at most once every this many `publish_load_stat`
# calls, unless `force=True` (extend/prefill batches, where load changes
# most). Load is a gauge consumed for routing, so per-decode-step publishing
# is wasteful.
LOAD_PUBLISH_INTERVAL = 5

# Wall-clock floor between publishes, applied even to forced ones. Bounds
# the cost of hot force paths — per-batch publishes under prefill-heavy
# load, and the idle loop (`on_idle` forces a publish every iteration,
# which busy-spins when --sleep-on-idle is off) — while staying far fresher
# than any router's staleness window.
LOAD_PUBLISH_MIN_INTERVAL_S = 0.05

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
    num_waiting_reqs, num_tokens, max_total_num_tokens, attn_dp_rank]`` —
    the router decoder reads the four counts and ignores the rest. The
    trailing field is always emitted (null when unset); array_like structs
    do not trim it. `attn_dp_rank` exists so the snapshot can be published
    directly through `ZmqEventPublisher.publish` (which stamps it); the
    router keys load by the subscriber's socket rank, not this field.
    """

    num_running_reqs: int
    num_waiting_reqs: int
    # KV tokens currently in use, from the engine's KV pool.
    num_tokens: int
    # KV-cache token capacity; 0 when unknown.
    max_total_num_tokens: int
    attn_dp_rank: Optional[int] = None


class SchedulerLoadPublisher:
    """Owns one scheduler's dedicated load PUB socket and the throttled,
    best-effort `publish_load_stat` path.

    Enabled on the same condition as KV-event publishing
    (`is_kv_publisher_rank`), on the port range `derive_load_port_base`
    packs after the KV-event range; within the range, the per-rank offset
    follows `select_kv_publisher_dp_rank`, exactly like the KV-event
    publisher, so pure-DP replicas don't collide on one port. Stays a
    no-op (`publisher is None`) when disabled or when no load port is
    derivable from the KV config.
    """

    def __init__(
        self,
        *,
        kv_events_config: Optional[str],
        ps: ParallelState,
        dp_size: int,
    ) -> None:
        # `publisher is None` means "no PUB socket is bound"; every early
        # return below leaves it None so `publish_load_stat` skips the
        # snapshot computation entirely.
        self.publisher: Optional[ZmqEventPublisher] = None
        self._publish_counter = 0
        self._last_publish_ts = 0.0
        # Consecutive publish failures, reset on success (drives the
        # periodic warn).
        self._fail_count = 0
        if not is_kv_publisher_rank(kv_events_config, ps):
            return
        try:
            cfg = KVEventsConfig.from_cli(kv_events_config)
        except Exception:
            # Malformed config — the KV publisher init would have failed too;
            # stay a no-op rather than raising at scheduler startup.
            return
        if cfg.publisher == "null" or not cfg.endpoint:
            return
        # Shared with describe_kv_events_publisher, which advertises the
        # same derivation (or omits the key) on /server_info — so a router
        # never subscribes to a port this constructor declined. (A runtime
        # bind failure below is the one case where the advertisement can
        # outlive the socket; the router then sees silence and falls back.)
        load_base = derive_load_port_base(cfg.endpoint, cfg.replay_endpoint, dp_size)
        if load_base is None:
            logger.info(
                "load-publisher disabled: no load port range is derivable "
                "from --kv-events-config endpoint %r (needs a tcp:// "
                "endpoint with an integer port whose load range fits in "
                "the u16 port space)",
                cfg.endpoint,
            )
            return
        load_endpoint = f"{cfg.endpoint[:cfg.endpoint.rfind(':')]}:{load_base}"
        # Dedicated load socket: own port, replay disabled, unbuffered (load
        # is a gauge, not a replayable delta). Best-effort like everything
        # else here: an unforeseen bind failure must not take down scheduler
        # startup over a routing hint.
        try:
            self.publisher = ZmqEventPublisher(
                select_kv_publisher_dp_rank(
                    ps.attn_dp_size, ps.attn_dp_rank, ps.dp_rank
                ),
                endpoint=load_endpoint,
                replay_endpoint=None,
                buffer_steps=0,
                topic=LOAD_TOPIC,
            )
        except Exception:
            logger.warning(
                "load-publisher disabled: failed to bind the load socket at %r",
                load_endpoint,
                exc_info=True,
            )
            self.publisher = None

    @property
    def enable(self) -> bool:
        """True when a real load PUB socket is bound."""
        return self.publisher is not None

    def publish_load_stat(
        self, load_provider: Callable[[], LoadSnapshot], force: bool = False
    ) -> None:
        """Publish a load snapshot, throttled to [`LOAD_PUBLISH_INTERVAL`]
        calls unless `force`, and to [`LOAD_PUBLISH_MIN_INTERVAL_S`] seconds
        always.

        `load_provider` returns a live [`LoadSnapshot`] read directly from
        scheduler state (`SchedulerLoadInquirer.get_loads`) — used instead of
        metrics stats, whose values are only populated under
        `--enable-metrics`. Invoked only after the throttles pass, so the
        snapshot is computed only when actually publishing.

        Best-effort: a failure here must never crash the scheduler loop —
        routers fall back to their own in-flight counter. Failures re-warn
        every [`LOAD_PUBLISH_FAIL_WARN_EVERY`] consecutive failures.
        """
        if self.publisher is None:
            return

        self._publish_counter += 1
        if not force and self._publish_counter < LOAD_PUBLISH_INTERVAL:
            return
        now = time.monotonic()
        if now - self._last_publish_ts < LOAD_PUBLISH_MIN_INTERVAL_S:
            return
        self._publish_counter = 0
        self._last_publish_ts = now

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

"""Per-scheduler load reporting for load-aware routers.

Independent of KV-cache events: each scheduler publishes a periodic
[`LoadStat`] gauge on its own ZMQ PUB socket (a dedicated port range,
packed immediately after the KV-event publisher's) so load-aware routers —
e.g. the experimental sgl-router `cache_aware_zmq` policy — can route on
the engine's true queue depth / KV occupancy instead of inferring load
from a router-side in-flight counter.

This is the *external* counterpart of `sglang.srt.managers.load_snapshot`:
that module fans per-rank `LoadSnapshot`s into shared memory (or a zmq
PUSH to node 0) for consumers inside the deployment — the
DataParallelController's dispatch and `/v1/loads`. Neither transport is
subscribable by an out-of-process router that only knows the worker's URL,
so load-aware routers get their own per-rank PUB socket instead, on a port
derivable from `/server_info` (see
`ServerArgs.describe_kv_events_publisher`). The payload is a compact,
tagged subset of the snapshot rather than the full `LoadSnapshot`, so the
router-facing wire contract stays fixed while the internal snapshot keeps
growing fields.

The transport (`ZmqEventPublisher`), config parsing, and rank/port
derivation are borrowed from `sglang.srt.disaggregation.kv_events`; the
load wire format and publishing cadence live here.

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
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec

from sglang.srt.disaggregation.kv_events import (
    KVEventsConfig,
    NullEventPublisher,
    ZmqEventPublisher,
    select_kv_publisher_dp_rank,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.load_snapshot import LoadSnapshot

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


def _tcp_port(endpoint: Optional[str]) -> Optional[int]:
    """Port of a tcp:// endpoint, or None when there is none to compare."""
    if not endpoint or not endpoint.startswith("tcp://"):
        return None
    _, _, tail = endpoint.rpartition(":")
    try:
        return int(tail)
    except ValueError:
        return None


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

    Enabled on the same condition as KV-event publishing (a
    `kv_events_config` on the pp/attn-TP/CP-rank-0 scheduler), and binds the
    load port range packed immediately after the KV-event range
    (`kv_base + dp_size`); within the range, the per-rank offset follows
    `select_kv_publisher_dp_rank`, exactly like the KV-event publisher, so
    pure-DP replicas don't collide on one port. Stays a no-op (a
    `NullEventPublisher`) when disabled or when the KV config has no usable
    ZMQ endpoint.
    """

    def __init__(
        self,
        *,
        kv_events_config: Optional[str],
        ps: ParallelState,
        dp_size: int,
    ) -> None:
        self.publisher: Any = NullEventPublisher()
        # `enable` means "a real PUB socket is bound": every early return
        # below leaves it False so `publish_load_stat` skips the snapshot
        # computation entirely instead of feeding the null publisher.
        self.enable = False
        self._publish_counter = 0
        # Consecutive publish failures, reset on success (drives the
        # periodic warn).
        self._fail_count = 0
        eligible = bool(
            kv_events_config
            and ps.pp_rank == 0
            and ps.attn_tp_rank == 0
            and ps.attn_cp_rank == 0
        )
        if not eligible:
            return
        try:
            cfg = KVEventsConfig.from_cli(kv_events_config)
        except Exception:
            # Malformed config — the KV publisher init would have failed too;
            # stay a no-op rather than raising at scheduler startup.
            return
        if cfg.publisher == "null" or not cfg.endpoint:
            return
        # Only tcp:// endpoints carry a port to offset. inproc:// and ipc://
        # are valid KV-event endpoints, but deriving a distinct load endpoint
        # from them is not defined, and offset_endpoint_port raises on ipc://
        # — so decline rather than take down scheduler startup over a load
        # socket.
        if not cfg.endpoint.startswith("tcp://"):
            logger.info(
                "load-publisher disabled: --kv-events-config endpoint %r is not "
                "tcp://, so no separate load port can be derived",
                cfg.endpoint,
            )
            return
        # `dp_size` is the KV port range width; the load range starts right
        # after it.
        load_endpoint = ZmqEventPublisher.offset_endpoint_port(cfg.endpoint, dp_size)
        if load_endpoint is None:
            return
        # Decline ports the range cannot legally bind, mirroring the
        # conditions under which describe_kv_events_publisher omits
        # load_endpoint_port_base from /server_info — so a router never
        # subscribes to a port this constructor declined: the load range
        # [kv_port + dp_size, kv_port + 2*dp_size) must fit in u16 and stay
        # clear of the replay ROUTER range (replay_port + rank), which by the
        # inherited convention (endpoint 5557 / replay 5558) sits exactly
        # where this socket would otherwise land.
        load_base = _tcp_port(load_endpoint)
        if load_base is None or load_base + dp_size - 1 > 65535:
            logger.info(
                "load-publisher disabled: load port range at %r would exceed "
                "the u16 port space",
                load_endpoint,
            )
            return
        replay_base = _tcp_port(cfg.replay_endpoint) if cfg.replay_endpoint else None
        if (
            replay_base is not None
            and load_base < replay_base + dp_size
            and replay_base < load_base + dp_size
        ):
            logger.info(
                "load-publisher disabled: load port range %d..%d overlaps the "
                "replay_endpoint range %d..%d",
                load_base,
                load_base + dp_size - 1,
                replay_base,
                replay_base + dp_size - 1,
            )
            return
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
            self.publisher = NullEventPublisher()
            return
        self.enable = True

    def publish_load_stat(
        self, load_provider: Callable[[], LoadSnapshot], force: bool = False
    ) -> None:
        """Publish a load snapshot, throttled to [`LOAD_PUBLISH_INTERVAL`]
        calls unless `force`.

        `load_provider` returns a live [`LoadSnapshot`] read directly from
        scheduler state (`SchedulerLoadInquirer.get_loads`) — used instead of
        metrics stats, whose values are only populated under
        `--enable-metrics`. Invoked only after the throttle passes, so the
        snapshot is computed only when actually publishing.

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

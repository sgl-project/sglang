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

Wire framing — three frames, matching the KV-event socket's layout so one
subscriber loop handles both: ``[topic b"load", big-endian i64 seq,
msgpack LoadStat]``. The transport is a plain synchronous PUB socket owned
here: a PUB send is an enqueue to ZMQ's IO thread, so unlike the KV-event
path there is no background thread or replay buffer — a gauge needs
neither, and a small send HWM sheds backlog toward a stalled subscriber
instead of queueing stale readings.

The port comes from `resolve_load_pub_range` in
`sglang.srt.disaggregation.kv_events` — the same function `/server_info`
advertises with — which packs the load range after the KV-event range,
bumping past the replay ROUTER range when the two overlap (with the
conventional replay = kv + 1 they always do); `--load-publish-endpoint`
moves the range outright. Operators must leave the resolved range free on
the host: with the default packing a worker's ZMQ footprint spans
`[kv_base, load_base + dp_size)` — `2 * dp_size` ports without a replay
endpoint, `2 * dp_size + 1` with the conventional adjacent one — so
co-hosted workers need their KV port bases spaced at least that far
apart, or one worker's load socket lands on the other's KV-event port.

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

import atexit
import logging
import time
from itertools import count
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    LOAD_TOPIC,
    KVEventsConfig,
    is_kv_publisher_rank,
    resolve_load_pub_range,
    select_kv_publisher_dp_rank,
)
from sglang.srt.utils.network import NetworkAddress, is_zmq_endpoint_ipv6

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.load_snapshot import LoadSnapshot

logger = logging.getLogger(__name__)

# Publish a load snapshot at most once every this many `publish_load_stat`
# calls, unless `force=True` (extend/prefill batches, where load changes
# most). Load is a gauge consumed for routing, so per-decode-step publishing
# is wasteful.
LOAD_PUBLISH_INTERVAL = 5

# Re-send an *unchanged* stat at most once per this many seconds. A changed
# stat always publishes immediately — busy->idle and idle->busy transitions
# are never delayed — while the heartbeat keeps the gauge fresh for
# subscribers with staleness windows and bounds the send rate on paths that
# force-publish an unchanged gauge in a tight loop (`on_idle` fires every
# event-loop iteration, which busy-spins when --sleep-on-idle is off).
LOAD_PUBLISH_HEARTBEAT_S = 1.0

# After the first report, re-warn about publish failures at most this often.
# Wall-clock rather than a count: a permanently broken socket fails once per
# publish attempt and attempts are driven by the scheduler loop, so a
# count-based bound would still emit at a rate proportional to that loop.
FAIL_WARN_PERIOD_S = 60.0

# Send high-water mark. Small on purpose: load is a gauge, so once a
# subscriber's pipe is full, shedding sends (PUB drops the newest message)
# loses readings that the next heartbeat supersedes anyway — better than
# queueing a backlog of stale ones for it to drain later.
LOAD_PUB_HWM = 8

_encoder = msgspec.msgpack.Encoder()


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
    trailing field is always emitted (array_like structs do not trim it;
    the publisher stamps its rank, and a decoder must also tolerate null
    there). The router keys load by the subscriber's socket rank, not this
    field — it is informational.
    """

    num_running_reqs: int
    num_waiting_reqs: int
    # KV tokens currently in use, from the engine's KV pool.
    num_tokens: int
    # KV-cache token capacity; 0 when unknown.
    max_total_num_tokens: int
    # Stamped with select_kv_publisher_dp_rank(...): the attention-DP rank
    # under DP attention, but the plain data-parallel rank in pure DP. The
    # name follows the EventBatch.attn_dp_rank precedent on the KV-event
    # socket; either way it is informational only.
    attn_dp_rank: Optional[int] = None


def _open_pub_socket(endpoint: str) -> zmq.Socket:
    """Bind the load PUB socket. Module-level so tests can stub the one side
    effect while exercising the real gating and port derivation."""
    sock = zmq.Context.instance().socket(zmq.PUB)
    try:
        sock.set_hwm(LOAD_PUB_HWM)
        sock.setsockopt(zmq.LINGER, 0)
        if is_zmq_endpoint_ipv6(endpoint):
            sock.setsockopt(zmq.IPV6, 1)
        sock.bind(endpoint)
    except Exception:
        # Release the handle on the shared context rather than leaking it.
        sock.close()
        raise
    return sock


class SchedulerLoadPublisher:
    """Owns one scheduler's dedicated load PUB socket and the throttled,
    best-effort `publish_load_stat` path.

    Enabled on the same condition as KV-event publishing
    (`is_kv_publisher_rank`), on the port range `resolve_load_pub_range`
    derives; within the range, the per-rank offset follows
    `select_kv_publisher_dp_rank`, exactly like the KV-event publisher, so
    pure-DP replicas don't collide on one port. Stays a no-op
    (`_socket is None`) when disabled or when no load range is resolvable.
    """

    def __init__(
        self,
        *,
        kv_events_config: Optional[str],
        ps: ParallelState,
        dp_size: int,
        load_publish_endpoint: Optional[str] = None,
    ) -> None:
        # `_socket is None` means "no PUB socket is bound"; every early
        # return below leaves it None so `publish_load_stat` skips the
        # snapshot computation entirely.
        self._socket: Optional[zmq.Socket] = None
        self._rank = 0
        self._seq = count()
        self._publish_counter = 0
        # The four counts of the last stat actually sent, and when — drives
        # the changed-immediately / unchanged-on-heartbeat dedup.
        self._last_counts: Optional[tuple] = None
        self._last_publish_ts = 0.0
        # Consecutive publish failures and when they were last reported.
        self._fail_count = 0
        self._last_fail_warn_ts = float("-inf")
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
        # same resolution (or omits the key) on /server_info — so a router
        # never subscribes to a range this constructor declined. (A runtime
        # bind failure below is the one case where the advertisement can
        # outlive the socket; the router then sees silence and falls back.)
        resolved, reason = resolve_load_pub_range(
            kv_endpoint=cfg.endpoint,
            replay_endpoint=cfg.replay_endpoint,
            dp_size=dp_size,
            load_publish_endpoint=load_publish_endpoint,
        )
        if resolved is None:
            if reason:
                logger.warning("load-publisher disabled: %s", reason)
            return
        host, base = resolved
        self._rank = select_kv_publisher_dp_rank(
            ps.attn_dp_size, ps.attn_dp_rank, ps.dp_rank
        )
        endpoint = NetworkAddress(host, base + self._rank).to_tcp()
        # Best-effort: an unforeseen bind failure must not take down
        # scheduler startup over a routing hint.
        try:
            self._socket = _open_pub_socket(endpoint)
            # The scheduler has no shutdown hook to call close() from (the
            # KV-event publisher cleans up the same way); LINGER=0 makes
            # this safe even under a hard exit.
            atexit.register(self.close)
        except Exception:
            logger.warning(
                "load-publisher disabled: failed to bind the load socket at "
                "%r; /server_info advertises this range but nothing is "
                "listening on it",
                endpoint,
                exc_info=True,
            )

    @property
    def enable(self) -> bool:
        """True when a real load PUB socket is bound."""
        return self._socket is not None

    def publish_load_stat(
        self,
        load_provider: Callable[[], LoadSnapshot],
        force: bool = False,
        snapshot: Optional[LoadSnapshot] = None,
    ) -> None:
        """Publish a load snapshot, throttled to [`LOAD_PUBLISH_INTERVAL`]
        calls unless `force`; an unchanged stat is re-sent at most once per
        [`LOAD_PUBLISH_HEARTBEAT_S`] while a changed one always goes out
        immediately.

        `load_provider` returns a live [`LoadSnapshot`] read directly from
        scheduler state (`SchedulerLoadInquirer.get_loads`) — used instead of
        metrics stats, whose values are only populated under
        `--enable-metrics`. Invoked only after the counter throttle passes,
        and not at all when the caller passes `snapshot` — a [`LoadSnapshot`]
        it already computed for another sink this same cycle (the
        DP-balancing writer), so the queues are never walked twice and a
        disabled publisher costs the caller nothing but this call.

        Best-effort: a failure here must never crash the scheduler loop —
        routers fall back to their own in-flight counter. Failures re-warn
        at most once per [`FAIL_WARN_PERIOD_S`].
        """
        if self._socket is None:
            return

        self._publish_counter += 1
        if not force and self._publish_counter < LOAD_PUBLISH_INTERVAL:
            return
        # Reset where the throttle passes, not where a send happens: resetting
        # only on the send path lets one dedup hit (or provider failure) leave
        # the counter saturated, silently disengaging the throttle — every
        # subsequent decode step would then run the O(queue) provider.
        self._publish_counter = 0

        try:
            load = snapshot if snapshot is not None else load_provider()
            counts = (
                load.num_running_reqs,
                load.num_waiting_reqs,
                load.num_used_tokens,
                load.max_total_num_tokens,
            )
            now = time.monotonic()
            if (
                counts == self._last_counts
                and now - self._last_publish_ts < LOAD_PUBLISH_HEARTBEAT_S
            ):
                return
            payload = _encoder.encode(
                LoadStat(
                    num_running_reqs=counts[0],
                    num_waiting_reqs=counts[1],
                    num_tokens=counts[2],
                    max_total_num_tokens=counts[3],
                    attn_dp_rank=self._rank,
                )
            )
            seq = next(self._seq).to_bytes(8, "big")
            try:
                self._socket.send_multipart(
                    (LOAD_TOPIC.encode(), seq, payload), zmq.NOBLOCK
                )
            except zmq.Again:
                # A stalled pipe: drop this reading (blocking the scheduler
                # loop would be worse) and leave the dedup state untouched
                # so the next call retries it instead of deduping away a
                # send nobody got.
                return
            # Recorded only after a successful hand-off so a failed publish
            # retries instead of being deduped away.
            self._last_counts = counts
            self._last_publish_ts = now
            self._fail_count = 0
        except Exception:
            self._fail_count += 1
            now = time.monotonic()
            if (
                self._fail_count == 1
                or now - self._last_fail_warn_ts >= FAIL_WARN_PERIOD_S
            ):
                self._last_fail_warn_ts = now
                logger.warning(
                    "load-publisher: publish_load_stat failed (%d consecutive); "
                    "load-aware routers fall back to their in-flight load signal",
                    self._fail_count,
                    exc_info=True,
                )

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

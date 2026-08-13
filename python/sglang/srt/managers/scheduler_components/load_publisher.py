"""Per-scheduler load reporting for load-aware routers.

Each scheduler publishes a periodic `LoadStat` gauge on its own ZMQ PUB
socket so out-of-process load-aware routers (e.g. sgl-router's
`cache_aware_zmq` policy) can route on real queue depth instead of a
router-side in-flight counter. The in-deployment counterpart lives in
`sglang.srt.managers.load_snapshot` (SHM / PUSH to node 0), which a router
that only knows the worker URL cannot subscribe to; the port is instead
advertised via `/server_info` (`ServerArgs.describe_kv_events_publisher`).
The payload is a compact tagged subset of `LoadSnapshot` so the wire
contract stays fixed as the snapshot grows.

Framing is the KV-event socket's, so one subscriber loop handles both:
``[b"load", big-endian i64 seq, msgpack LoadStat]``. The transport is a
plain synchronous PUB socket (a send just enqueues to ZMQ's IO thread) —
no background thread or replay buffer, which a gauge does not need.

The port comes from `resolve_load_pub_range` (the same function
`/server_info` advertises with, so the two cannot drift). A worker's ZMQ
footprint is `2 * dp_size` ports after its KV base (`2 * dp_size + 1` with
the conventional adjacent replay), so co-hosted workers must space their
KV bases at least that far apart or one's load socket lands on another's
KV-event port; `--load-publish-endpoint` moves the range off a conflict.
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

# Publish at most once per this many calls unless force=True.
LOAD_PUBLISH_INTERVAL = 5

# Floor on how often the O(queue) provider runs. Caps compute on the stalled
# no-batch path (on_idle spins without sleeping and calls with force=False),
# where the call throttle alone would still fire at loop rate / INTERVAL.
LOAD_MIN_COMPUTE_INTERVAL_S = 0.05

# An unchanged stat is re-sent at most this often; a changed one always goes
# out immediately, so transitions are never delayed. Bounds the send rate on
# the idle spin loop (on_idle force-publishes every iteration).
LOAD_PUBLISH_HEARTBEAT_S = 1.0

# Re-warn about publish failures at most this often. Wall-clock, not a count:
# failures are driven by the scheduler loop, so a count bound would still
# flood at loop rate.
FAIL_WARN_PERIOD_S = 60.0

# Small HWM: load is a gauge, so shedding at a full pipe loses readings the
# next heartbeat supersedes. ZMQ_CONFLATE (true newest-wins) is unusable — it
# keeps a single frame, breaking the 3-frame framing — so a bounded backlog
# is the closest fit.
LOAD_PUB_HWM = 8

_encoder = msgspec.msgpack.Encoder()


class LoadStat(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    # No omit_defaults: it would trim trailing fields, shortening a wire
    # shape the router decodes positionally.
    gc=False,  # type: ignore[call-arg]
    tag=True,  # type: ignore[call-arg]
):
    """Per-scheduler runtime load snapshot.

    Wire shape (tag + array_like): ``["LoadStat", num_running_reqs,
    num_waiting_reqs, num_tokens, max_total_num_tokens, attn_dp_rank]``. The
    router reads the four counts; array_like always emits the trailing field
    (null when unset), so a decoder must tolerate it.
    """

    num_running_reqs: int
    num_waiting_reqs: int
    num_tokens: int  # KV tokens in use
    max_total_num_tokens: int  # KV capacity; 0 when unknown
    # attn_dp_rank under DP attention, else the plain dp_rank; informational
    # only (the router keys by socket rank). Name follows EventBatch's.
    attn_dp_rank: Optional[int] = None


def _open_pub_socket(endpoint: str) -> zmq.Socket:
    """Bind the load PUB socket. Module-level so tests can stub the one side
    effect while exercising the real gating and port derivation. Not
    get_zmq_socket: that sets SNDHWM=0, defeating LOAD_PUB_HWM."""
    sock = zmq.Context.instance().socket(zmq.PUB)
    try:
        sock.set_hwm(LOAD_PUB_HWM)
        sock.setsockopt(zmq.LINGER, 0)
        if is_zmq_endpoint_ipv6(endpoint):
            sock.setsockopt(zmq.IPV6, 1)
        sock.bind(endpoint)
    except Exception:
        sock.close()  # don't leak the handle on the shared context
        raise
    return sock


class SchedulerLoadPublisher:
    """Owns one scheduler's dedicated load PUB socket and the throttled,
    best-effort `publish_load_stat` path.

    Enabled on the same condition as KV-event publishing
    (`is_kv_publisher_rank`), keyed per rank like it
    (`select_kv_publisher_dp_rank`) so pure-DP replicas don't collide. Stays
    a no-op (`_socket is None`) when disabled or no range is resolvable.
    """

    def __init__(
        self,
        *,
        kv_events_config: Optional[str],
        ps: ParallelState,
        load_publish_endpoint: Optional[str] = None,
    ) -> None:
        # _socket is None == disabled: every early return below leaves it so,
        # and publish_load_stat then skips the snapshot entirely.
        self._socket: Optional[zmq.Socket] = None
        self._rank = 0
        self._seq = count()
        self._publish_counter = 0
        # Last sent counts + timestamp, driving the dedup/heartbeat.
        self._last_counts: Optional[tuple] = None
        self._last_publish_ts = 0.0
        self._last_compute_ts = float("-inf")
        self._fail_count = 0
        self._last_fail_warn_ts = float("-inf")
        if not is_kv_publisher_rank(kv_events_config, ps):
            return
        try:
            cfg = KVEventsConfig.from_cli(kv_events_config)
        except Exception:
            # Malformed config: the KV publisher would fail too; stay a no-op.
            return
        if cfg.publisher == "null" or not cfg.endpoint:
            return
        # Same resolver /server_info advertises with, so a router never
        # subscribes to a range this declines — except a runtime bind failure
        # below, which the advertisement can't retract (router sees silence).
        resolved, reason = resolve_load_pub_range(
            kv_endpoint=cfg.endpoint,
            replay_endpoint=cfg.replay_endpoint,
            dp_size=ps.dp_size,
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
        try:
            self._socket = _open_pub_socket(endpoint)
            # No scheduler shutdown hook to close() from; LINGER=0 keeps a
            # hard exit safe. (The KV-event publisher cleans up the same way.)
            atexit.register(self.close)
        except Exception:
            # Best-effort: a bind failure must not take down startup.
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
        [`LOAD_PUBLISH_HEARTBEAT_S`], a changed one always immediately.

        `load_provider` reads live scheduler state
        (`SchedulerLoadInquirer.get_loads`), used over metrics stats which
        are only populated under `--enable-metrics`. Skipped when the caller
        passes `snapshot` (one it already computed for the DP-balancing sink
        this cycle), and otherwise rate-limited to
        [`LOAD_MIN_COMPUTE_INTERVAL_S`].

        Best-effort: never crashes the loop (routers fall back to their own
        counter); failures re-warn at most once per [`FAIL_WARN_PERIOD_S`].
        """
        if self._socket is None:
            return

        self._publish_counter += 1
        if not force and self._publish_counter < LOAD_PUBLISH_INTERVAL:
            return
        # Reset where the throttle passes, not on send: a dedup hit or
        # provider failure would otherwise leave it saturated, silently
        # disengaging the throttle onto the O(queue) provider every step.
        self._publish_counter = 0

        now = time.monotonic()
        try:
            if snapshot is not None:
                load = snapshot
            else:
                # Floor the provider so a never-sleeping stall doesn't run it
                # at loop rate. A changed gauge still gets out within the
                # floor; only forced (fully-idle) calls bypass it.
                if (
                    not force
                    and now - self._last_compute_ts < LOAD_MIN_COMPUTE_INTERVAL_S
                ):
                    return
                self._last_compute_ts = now
                load = load_provider()
            counts = (
                load.num_running_reqs,
                load.num_waiting_reqs,
                load.num_used_tokens,
                load.max_total_num_tokens,
            )
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
                # Unreachable under PUB (it sheds at HWM, never blocks); kept
                # so a socket-type change still retries rather than deduping
                # away a send nobody got.
                return
            # Recorded on enqueue. A silent HWM drop leaves a changed reading
            # unseen until the next heartbeat; that is inherent to PUB.
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

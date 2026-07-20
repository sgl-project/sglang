# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NCCL RAS health collector — detection-only, supplements the forward watchdog."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from enum import IntEnum
from typing import Any, Optional

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# RAS STATUS timestamp format (local, second-granularity, no TZ).
_RAS_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

# Collective types observed in collective_counts (all always present, unused=0).
# Treated as an open set at parse time (unknown keys are tolerated), but used
# as the canonical iteration order for divergence tracking.
KNOWN_COLLECTIVES = (
    "AllReduce",
    "AllGather",
    "Broadcast",
    "Reduce",
    "ReduceScatter",
)


class RasState(IntEnum):
    # Ordered by severity so max() over comms yields the job-wide worst_state.
    # UNKNOWN (poll failed / stale timestamp) ranks above HEALTHY but below
    # SUSPECT so a flaky poll does not mask a real issue.
    HEALTHY = 0
    UNKNOWN = 1
    SUSPECT = 2
    STUCK = 3  # advisory: subgroup divergence sustained for stuck_polls
    ERROR = 4  # per-rank NCCL error state
    DEAD = 5  # a rank is considered_dead (cross-node)


# ---------------------------------------------------------------------------
# msgspec structures (frozen; field names align with the verified RAS schema)
# ---------------------------------------------------------------------------


class RasRank(msgspec.Struct, frozen=True, kw_only=True):
    rank: int
    host: str = ""
    pid: int = 0
    cuda_dev: int = 0
    nvml_dev: int = 0
    # Live-rank status fields. unresponsive/considered_dead are absent on live
    # ranks (present only under missing_ranks[].status).
    init_state: int = 0
    async_error: int = 0
    finalize_called: bool = False
    destroy_flag: bool = False
    abort_flag: bool = False
    unresponsive: bool = False
    considered_dead: bool = False
    collective_counts: dict[str, int] = {}  # type: ignore[assignment]


class RasCommunicator(msgspec.Struct, frozen=True, kw_only=True):
    comm_key: str  # f"{hash}:{secondary_hash}"
    hash: str
    secondary_hash: str
    size: int
    ranks_count: int
    missing_ranks_count: int
    ranks: list[RasRank]
    missing_ranks: list[RasRank]


class RasStatus(msgspec.Struct, frozen=True, kw_only=True):
    nccl_version: str = ""
    timestamp: str = ""
    communicators_count: int = 0
    communicators: list[RasCommunicator] = []
    # Retained for diagnostics / future support-bundle export.
    raw: dict[str, Any] = {}  # type: ignore[assignment]
    # Parsed epoch of `timestamp`; None if missing/unparseable (detector fails
    # closed to UNKNOWN). Cached at parse time so the detector never re-parses.
    timestamp_epoch: Optional[float] = None


class RasCommFinding(msgspec.Struct, frozen=True, kw_only=True):
    comm_key: str
    state: RasState
    missing: int
    unresponsive: int
    dead: int
    ranks_in_error: int
    # collective type -> cross-rank (max - min) spread among non-zero samples.
    divergence: dict[str, int] = {}  # type: ignore[assignment]
    stuck: bool = False
    # New considered_dead (absent/False -> True) transitions this poll.
    peer_dead_transitions: int = 0


class RasFindings(msgspec.Struct, frozen=True, kw_only=True):
    per_comm: dict[str, RasCommFinding] = {}  # type: ignore[assignment]
    total_missing: int = 0
    total_unresponsive: int = 0
    total_dead: int = 0
    total_ranks_in_error: int = 0
    divergence_comms: list[str] = []  # type: ignore[assignment]
    worst_state: RasState = RasState.HEALTHY
    poll_success: bool = True
    # None when poll failed or timestamp unparseable/stale (unset, not fresh).
    last_collection_age_sec: Optional[float] = None


# ---------------------------------------------------------------------------
# Parsing (pure function — fixture-testable, no socket)
# ---------------------------------------------------------------------------


def _parse_rank(raw: dict[str, Any]) -> RasRank:
    """Parse one rank entry (live or missing). External JSON → dict.get."""
    status = raw.get("status", {}) or {}
    counts = raw.get("collective_counts", {}) or {}
    return RasRank(
        rank=int(raw.get("rank", -1)),
        host=str(raw.get("host", "")),
        pid=int(raw.get("pid", 0)),
        cuda_dev=int(raw.get("cuda_dev", 0)),
        nvml_dev=int(raw.get("nvml_dev", 0)),
        init_state=int(status.get("init_state", 0)),
        async_error=int(status.get("async_error", 0)),
        finalize_called=bool(status.get("finalize_called", False)),
        destroy_flag=bool(status.get("destroy_flag", False)),
        abort_flag=bool(status.get("abort_flag", False)),
        unresponsive=bool(status.get("unresponsive", False)),
        considered_dead=bool(status.get("considered_dead", False)),
        collective_counts={str(k): int(v) for k, v in counts.items()},
    )


def _parse_comm(raw: dict[str, Any]) -> RasCommunicator:
    h = str(raw.get("hash", ""))
    sh = str(raw.get("secondary_hash", ""))
    ranks_raw = raw.get("ranks", []) or []
    missing_raw = raw.get("missing_ranks", []) or []
    return RasCommunicator(
        comm_key=f"{h}:{sh}",
        hash=h,
        secondary_hash=sh,
        size=int(raw.get("size", 0)),
        ranks_count=int(raw.get("ranks_count", len(ranks_raw))),
        missing_ranks_count=int(raw.get("missing_ranks_count", len(missing_raw))),
        ranks=[_parse_rank(r) for r in ranks_raw],
        missing_ranks=[_parse_rank(r) for r in missing_raw],
    )


def parse_status(raw: dict[str, Any]) -> RasStatus:
    """Pure parser: tolerate missing ``ras{}``/``collective_counts``/empty ranks.

    Caches the parsed ``timestamp`` epoch on the status (``None`` if missing or
    unparseable); the detector treats a present-but-stale timestamp and a
    missing/unparseable one identically — fail closed to UNKNOWN.
    """
    comms_raw = raw.get("communicators", []) or []
    comms = [_parse_comm(c) for c in comms_raw]
    timestamp = str(raw.get("timestamp", ""))
    timestamp_epoch: Optional[float] = None
    if timestamp:
        try:
            timestamp_epoch = float(
                time.mktime(time.strptime(timestamp, _RAS_TIMESTAMP_FMT))
            )
        except (ValueError, TypeError, OverflowError):
            timestamp_epoch = None
    return RasStatus(
        nccl_version=str(raw.get("nccl_version", "")),
        timestamp=timestamp,
        communicators_count=int(raw.get("communicators_count", len(comms))),
        communicators=comms,
        raw=raw,
        timestamp_epoch=timestamp_epoch,
    )


# ---------------------------------------------------------------------------
# Socket client
# ---------------------------------------------------------------------------


class RasSocketClient:
    """Short-connection pull client for the NCCL RAS line protocol."""

    def __init__(
        self,
        addr: str = "localhost",
        port: int = 28028,
        connect_timeout: float = 1.0,
        read_timeout: float = 1.0,
    ) -> None:
        self.addr = addr
        self.port = port
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    @classmethod
    def from_endpoint(cls, endpoint: str) -> "RasSocketClient":
        """Parse ``"host:port"`` (falls back to defaults on malformed input)."""
        host, port = "localhost", 28028
        if endpoint:
            # tolerate "host:port" as well as a bare host
            if ":" in endpoint:
                h, _, p = endpoint.rpartition(":")
                if h:
                    host = h
                try:
                    port = int(p)
                except ValueError:
                    pass
            else:
                host = endpoint
        return cls(addr=host, port=port)

    def poll_status(self) -> Optional[RasStatus]:
        """Send ``SET FORMAT json`` + ``STATUS`` and return parsed status.

        Returns ``None`` on connection refused / timeout / malformed payload —
        the detector treats ``None`` as ``poll_success=0`` (job/progress crash
        signal on a single node).
        """
        try:
            with socket.create_connection(
                (self.addr, self.port), timeout=self.connect_timeout
            ) as sock:
                sock.settimeout(self.read_timeout)
                sock.sendall(b"SET FORMAT json\n")
                sock.sendall(b"STATUS\n")
                chunks: list[bytes] = []
                total = 0
                while True:
                    try:
                        buf = sock.recv(4096)
                    except socket.timeout:
                        break
                    if not buf:
                        break
                    total += len(buf)
                    if total > _MAX_PAYLOAD_BYTES:
                        logger.warning(
                            "NCCL RAS payload exceeded %d bytes, discarding",
                            _MAX_PAYLOAD_BYTES,
                        )
                        return None
                    chunks.append(buf)
        except (ConnectionRefusedError, socket.timeout, OSError):
            return None
        data = b"".join(chunks)
        return self._parse_payload(data)

    @staticmethod
    def _parse_payload(data: bytes) -> Optional[RasStatus]:
        """Skip the ``OK\\n`` ack and any leading text to the first ``{``."""
        if not data:
            return None
        idx = data.find(b"{")
        if idx < 0:
            # TEXT_ONLY (< 2.28.7): no JSON to parse.
            return None
        try:
            raw = json.loads(data[idx:])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        return parse_status(raw)


# Upper bound on a RAS STATUS payload. Real snapshots are a few KB; this only
# guards against a misbehaving endpoint streaming forever.
_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Detector (pure state machine; holds cross-poll history)
# ---------------------------------------------------------------------------


# Staleness: a timestamp older than this means the RAS thread stopped
# updating it — fail closed to UNKNOWN.
_STALE_SEC = 300.0


def _rank_in_error(rank: RasRank) -> bool:
    """Per-rank NCCL error state (clean, noise-free signal)."""
    return (
        rank.async_error != 0
        or rank.abort_flag
        or rank.finalize_called
        or rank.destroy_flag
    )


def _is_all_zero_sample(comm: RasCommunicator) -> bool:
    """True if every live rank's collective_counts is all-zero this poll.

    About 1/3–1/2 of polls on a healthy busy job return all-zero counts (a
    cached "no fresh counts" snapshot). Such a sample carries no progress
    information and must not be compared against the prior non-zero sample.
    """
    if not comm.ranks:
        # No live ranks (all missing) — treat as no progress info either way;
        # the dead/missing path handles reporting, not stuck.
        return True
    for rank in comm.ranks:
        if not rank.collective_counts:
            return True
        if any(v != 0 for v in rank.collective_counts.values()):
            return False
    return True


class _CommTracker:
    """Per-communicator mutable history for the per-rank liveness check."""

    __slots__ = ("counts", "frozen_polls", "recent_advance")

    def __init__(self) -> None:
        # comm rank -> {collective type -> last non-zero count}
        self.counts: dict[int, dict[str, int]] = {}
        # (rank, collective type) -> consecutive non-zero polls without advance
        self.frozen_polls: dict[tuple[int, str], int] = {}
        # collective type -> polls-of-recent-advance remaining. Distinguishes a
        # subgroup lag (leader moving) from a symmetric hang (nobody moving).
        self.recent_advance: dict[str, int] = {}


class RasDetector:
    """State machine over consecutive RAS polls; owns per-comm history."""

    def __init__(self, stuck_polls: Optional[int] = None) -> None:
        # is_initializing is passed into evaluate() rather than stored so the
        # detector stays a pure function of (history, status, now, flags).
        self._stuck_polls = (
            int(stuck_polls)
            if stuck_polls is not None
            else int(envs.SGLANG_NCCL_RAS_STUCK_POLLS.get())
        )
        self._trackers: dict[str, _CommTracker] = {}
        # comm_key -> previous state, for transition logging.
        self._prev_state: dict[str, RasState] = {}
        # comm_key -> set of ranks previously considered_dead (to detect the
        # absent/False -> True transition and count it exactly once).
        self._prev_dead: dict[str, set[int]] = {}
        # Whether we have ever seen a non-None poll (cold-start bookkeeping).
        self._ever_polled = False

    def evaluate(
        self,
        status: Optional[RasStatus],
        now: float,
        is_initializing: bool = False,
    ) -> RasFindings:
        if status is None:
            self._ever_polled = True
            return RasFindings(
                worst_state=RasState.UNKNOWN,
                poll_success=False,
                last_collection_age_sec=None,
            )
        self._ever_polled = True

        # Staleness fail-closed: unparseable or stale timestamp → UNKNOWN.
        if status.timestamp_epoch is None:
            return self._finalize_unknown()
        age = now - status.timestamp_epoch
        if age > _STALE_SEC:
            return self._finalize_unknown()

        per_comm: dict[str, RasCommFinding] = {}
        total_missing = total_unresponsive = total_dead = total_err = 0
        total_dead_transitions = 0
        divergence_comms: list[str] = []
        worst = RasState.HEALTHY

        for comm in status.communicators:
            finding = self._evaluate_comm(comm, is_initializing)
            per_comm[comm.comm_key] = finding
            total_missing += finding.missing
            total_unresponsive += finding.unresponsive
            total_dead += finding.dead
            total_err += finding.ranks_in_error
            total_dead_transitions += finding.peer_dead_transitions
            if finding.divergence:
                divergence_comms.append(comm.comm_key)
            if finding.state > worst:
                worst = finding.state

        return RasFindings(
            per_comm=per_comm,
            total_missing=total_missing,
            total_unresponsive=total_unresponsive,
            total_dead=total_dead,
            total_ranks_in_error=total_err,
            divergence_comms=divergence_comms,
            worst_state=worst,
            poll_success=True,
            last_collection_age_sec=age,
        )

    # -- per-communicator evaluation ----------------------------------------

    def _evaluate_comm(
        self, comm: RasCommunicator, is_initializing: bool
    ) -> RasCommFinding:
        missing = comm.missing_ranks_count
        unresponsive = sum(
            1 for r in comm.missing_ranks if r.unresponsive
        )
        dead = sum(1 for r in comm.missing_ranks if r.considered_dead)
        ranks_in_error = sum(1 for r in comm.ranks if _rank_in_error(r))
        # Compute once; reused by both the error/missing-path liveness update
        # and the advisory stuck path (avoids a second ranks×types scan).
        all_zero = _is_all_zero_sample(comm)
        peer_dead_transitions = self._track_dead(comm)

        # Priority: considered_dead (cross-node ~60s) → DEAD; per-rank NCCL
        # error state → ERROR; missing/unresponsive → SUSPECT; else advisory.
        divergence: dict[str, int] = {}
        stuck = False
        if dead > 0:
            state = RasState.DEAD
        elif ranks_in_error > 0:
            state = RasState.ERROR
        elif missing > 0 or unresponsive > 0:
            state = RasState.SUSPECT
        else:
            # Advisory collective divergence (skipped during cold start / all-zero).
            self._update_liveness(comm, all_zero)
            divergence, stuck = self._evaluate_liveness(comm, is_initializing, all_zero)
            if stuck:
                state = RasState.STUCK
            elif divergence:
                state = RasState.SUSPECT
            else:
                state = RasState.HEALTHY
        if state in (RasState.DEAD, RasState.ERROR, RasState.SUSPECT):
            self._update_liveness(comm, all_zero)

        self._record_state(comm.comm_key, state)
        return RasCommFinding(
            comm_key=comm.comm_key,
            state=state,
            missing=missing,
            unresponsive=unresponsive,
            dead=dead,
            ranks_in_error=ranks_in_error,
            divergence=divergence,
            stuck=stuck,
            peer_dead_transitions=peer_dead_transitions,
        )

    def _track_dead(self, comm: RasCommunicator) -> int:
        """Newly confirmed considered_dead (False→True) this poll (per comm)."""
        prev = self._prev_dead.get(comm.comm_key, set())
        curr = {r.rank for r in comm.missing_ranks if r.considered_dead}
        newly_dead = len(curr - prev)
        self._prev_dead[comm.comm_key] = curr
        return newly_dead

    def _evaluate_liveness(
        self, comm: RasCommunicator, is_initializing: bool, all_zero: bool
    ) -> tuple[dict[str, int], bool]:
        """Per-rank liveness: drop all-zero samples, compare consecutive
        non-zero samples, require sustained freeze + cross-rank spread>0."""
        tracker = self._trackers.setdefault(comm.comm_key, _CommTracker())

        if is_initializing:
            # Cold start: collective cadence unknown; never report stuck.
            return {}, False

        if all_zero:
            # No progress information this poll — do not touch history.
            return {}, False

        divergence: dict[str, int] = {}
        # Per collective type, whether any rank advanced this poll.
        type_advanced: dict[str, bool] = {}

        for rank in comm.ranks:
            per_type = tracker.counts.setdefault(rank.rank, {})
            for ctype, count in rank.collective_counts.items():
                if count == 0:
                    continue
                prev = per_type.get(ctype)
                if prev is None:
                    per_type[ctype] = count
                    continue
                if count != prev:
                    # This rank advanced on this type → reset its freeze.
                    tracker.frozen_polls.pop((rank.rank, ctype), None)
                    per_type[ctype] = count
                    type_advanced[ctype] = True
                else:
                    key = (rank.rank, ctype)
                    tracker.frozen_polls[key] = tracker.frozen_polls.get(key, 0) + 1

        # Cross-rank spread per collective type, over the non-zero sample.
        for ctype in self._union_types(comm):
            values = [
                r.collective_counts.get(ctype, 0)
                for r in comm.ranks
                if r.collective_counts.get(ctype, 0) != 0
            ]
            if len(values) >= 2:
                spread = max(values) - min(values)
                if spread > 0:
                    divergence[ctype] = spread
            # Stamp "recently advanced" so a freeze counter that crosses a
            # single leader jitter-pause still counts.
            if type_advanced.get(ctype):
                tracker.recent_advance[ctype] = self._stuck_polls

        # Decay recent-advance counters once per poll.
        for ctype in list(tracker.recent_advance.keys()):
            tracker.recent_advance[ctype] -= 1
            if tracker.recent_advance[ctype] <= 0:
                del tracker.recent_advance[ctype]

        # Stuck only if a rank froze stuck_polls samples on a diverging type
        # (spread>0) whose leaders recently advanced — a subgroup lag, not a
        # symmetric hang (nobody advances, spread stays ~constant).
        stuck = False
        if divergence:
            for (rkey, ctype), polls in tracker.frozen_polls.items():
                if (
                    polls >= self._stuck_polls
                    and ctype in divergence
                    and ctype in tracker.recent_advance
                ):
                    stuck = True
                    break

        return divergence, stuck

    def _update_liveness(self, comm: RasCommunicator, all_zero: bool) -> None:
        """Keep liveness history consistent on the error/missing path so a
        later recovery is detected cleanly (advance resets freezes)."""
        if all_zero:
            return
        tracker = self._trackers.setdefault(comm.comm_key, _CommTracker())
        for rank in comm.ranks:
            per_type = tracker.counts.setdefault(rank.rank, {})
            for ctype, count in rank.collective_counts.items():
                if count != 0:
                    per_type.setdefault(ctype, count)

    @staticmethod
    def _union_types(comm: RasCommunicator) -> list[str]:
        seen: set[str] = set()
        for r in comm.ranks:
            seen.update(r.collective_counts.keys())
        # Stable order: known collectives first, then any extras.
        ordered = [c for c in KNOWN_COLLECTIVES if c in seen]
        ordered.extend(sorted(seen - set(KNOWN_COLLECTIVES)))
        return ordered

    # -- helpers ------------------------------------------------------------

    def _finalize_unknown(self) -> RasFindings:
        return RasFindings(
            worst_state=RasState.UNKNOWN,
            poll_success=True,
            last_collection_age_sec=None,
        )

    def _record_state(self, comm_key: str, state: RasState) -> None:
        prev = self._prev_state.get(comm_key)
        if prev is not None and prev != state:
            logger.warning(
                "NCCL RAS comm %s state transition: %s -> %s", comm_key, prev.name, state.name
            )
        self._prev_state[comm_key] = state


# ---------------------------------------------------------------------------
# Daemon collector (job-wide singleton, world rank 0)
# ---------------------------------------------------------------------------


class RasMonitor:
    """Job-wide RAS collector daemon; polls STATUS, runs the detector, publishes."""

    def __init__(
        self,
        *,
        client: RasSocketClient,
        detector: RasDetector,
        poll_interval: float,
        metrics_collector=None,
        is_initializing_fn=None,
        name: str = "nccl-ras-collector",
    ) -> None:
        self._client = client
        self._detector = detector
        self._interval = poll_interval
        self._metrics = metrics_collector
        # is_initializing_fn() -> bool; defaults to "never initializing" so the
        # monitor is self-contained when the scheduler does not wire the flag.
        self._is_initializing = is_initializing_fn or (lambda: False)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._name = name

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=self._name
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(self._interval * 2, 1.0))
            if thread.is_alive():
                logger.warning(
                    "NCCL RAS collector thread did not stop within %.1fs",
                    max(self._interval * 2, 1.0),
                )
            self._thread = None

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self._interval):
                self._poll_once()
        except Exception:  # pragma: no cover - daemon resilience
            logger.error("NCCL RAS collector thread crashed", exc_info=True)

    def _poll_once(self) -> None:
        status = self._client.poll_status()
        now = time.time()
        findings = self._detector.evaluate(
            status, now, is_initializing=self._is_initializing()
        )
        # Log on transitions/non-healthy even when metrics are off (log-only fallback).
        self._log_findings(findings)
        if self._metrics is not None:
            self._metrics.log_nccl_ras(findings)

    @staticmethod
    def _log_findings(findings: RasFindings) -> None:
        if findings.worst_state == RasState.HEALTHY and findings.poll_success:
            return  # avoid log spam on healthy polls
        if not findings.poll_success:
            logger.warning(
                "NCCL RAS poll failed (poll_success=0); "
                "missing=%d unresponsive=%d dead=%d error_ranks=%d",
                findings.total_missing,
                findings.total_unresponsive,
                findings.total_dead,
                findings.total_ranks_in_error,
            )
            return
        logger.warning(
            "NCCL RAS findings: worst_state=%s missing=%d unresponsive=%d "
            "dead=%d error_ranks=%d divergence_comms=%s age=%s",
            findings.worst_state.name,
            findings.total_missing,
            findings.total_unresponsive,
            findings.total_dead,
            findings.total_ranks_in_error,
            findings.divergence_comms,
            findings.last_collection_age_sec,
        )

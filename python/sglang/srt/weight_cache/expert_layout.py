# SPDX-License-Identifier: Apache-2.0
"""The weight cache daemon's record of where its routed experts are.

With the weight cache on, an engine's routed expert weights are the daemon's
tensors, mapped zero-copy. An EPLB rebalance moves experts in place, so it
moves them for the daemon and for every engine that maps it, while only the
rebalancing engine's ExpertLocationMetadata follows. The daemon therefore keeps
the layout of its weights -- the ``physical_to_logical_map`` -- with a version:

* An engine attaches to its rank's daemon before it builds its expert location
  metadata, and builds the metadata from the daemon's map. A restarted or a
  second engine then routes with the layout the shared weights are in.
* A rebalance moves each chunk of layers in two phases: ``begin`` marks the
  move pending on every rank's daemon, ``commit`` stores the new rows and bumps
  the version after the weights have moved. An engine that disconnects while
  its move is pending leaves the daemon dirty, and a dirty daemon serves no
  weights.
* Only an engine that is alone on a daemon may move experts: a second engine
  that maps the weights would keep routing with the old layout.

The attach connection stays open for the engine's lifetime and is its
presence: the kernel closes it however the engine exits, so liveness needs no
PIDs, which another container cannot see.
"""

import hashlib
import json
import logging
import os
import selectors
import socket
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.runtime_context import get_exec, get_model, get_parallel, get_resources

from .protocol import get_socket_path, recv_msg, send_msg

logger = logging.getLogger(__name__)

ATTACH = "expert_layout_attach"
QUERY = "expert_layout_query"
BEGIN = "expert_layout_begin"
COMMIT = "expert_layout_commit"
ABORT = "expert_layout_abort"
REQUEST_TYPES = frozenset((ATTACH, QUERY, BEGIN, COMMIT, ABORT))

# Reply statuses of the requests above. BUSY means not now (another engine
# maps the weights); REFUSED means that retrying will not help.
OK = "ok"
BUSY = "busy"
REFUSED = "refused"

# The fetch_state status for weights the daemon refuses to serve because of
# their expert layout.
LAYOUT_REFUSED = "layout_refused"

# What a rank found when it attached at startup.
TRACKED = "tracked"  # the daemon tracks the layout of its weights
UNTRACKED = "untracked"  # the daemon runs without --enable-eplb
LEGACY = "legacy"  # the daemon predates these requests
ABSENT = "absent"  # no daemon listens at the socket path

# How long a starting engine waits for another engine's move to end.
PENDING_MOVE_TIMEOUT = 300.0
_PENDING_MOVE_POLL_INTERVAL = 0.5


def layout_digest(rows: Sequence[Sequence[int]]) -> str:
    """SHA-256 of a physical_to_logical_map given as rows of ints."""
    text = json.dumps([[int(x) for x in row] for row in rows], separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _busy(reason: str) -> Dict[str, Any]:
    return {"status": BUSY, "reason": reason}


def _refused(reason: str) -> Dict[str, Any]:
    return {"status": REFUSED, "reason": reason}


def _layers(layer_ids: Sequence[int]) -> str:
    """Layer ids for messages, as runs: "layer 3", "layers 0-3,7"."""
    runs: List[List[int]] = []
    for layer_id in sorted(layer_ids):
        if runs and layer_id == runs[-1][1] + 1:
            runs[-1][1] = layer_id
        else:
            runs.append([layer_id, layer_id])
    text = ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
    return f"layer {text}" if len(layer_ids) == 1 else f"layers {text}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


_RESTART_HINT = (
    "Restart the weight cache daemons, and the engines that use them, so that "
    "the weights are loaded from disk again."
)


# ---------------------------------------------------------------------------
# Daemon side
# ---------------------------------------------------------------------------


@dataclass
class _PendingMove:
    engine_id: str
    base_version: int
    layer_ids: List[int]
    started: float


class ExpertLayoutRegistry:
    """Daemon side: the expert layout of the served weights, its version, the
    pending move, and the attached engines.

    Every method takes the lock: the serve loop handles requests while the
    presence thread reports engines that went away.
    """

    def __init__(self, rows: Optional[Sequence[Sequence[int]]]):
        self._lock = threading.Lock()
        self._rows = None if rows is None else [[int(x) for x in row] for row in rows]
        self._digest = None if self._rows is None else layout_digest(self._rows)
        self._version = 0
        self._pending: Optional[_PendingMove] = None
        self._dirty: Optional[str] = None
        self._attached: Dict[str, str] = {}  # engine_id -> label
        self._untracked_consumers = 0

    @classmethod
    def from_metadata(cls, metadata) -> "ExpertLayoutRegistry":
        """Track the layout of the daemon's ExpertLocationMetadata; ``None``
        (no EPLB) tracks nothing."""
        if metadata is None:
            return cls(None)
        return cls(metadata.physical_to_logical_map_cpu.tolist())

    @property
    def tracked(self) -> bool:
        return self._rows is not None

    def _label(self, engine_id: str) -> str:
        return self._attached.get(engine_id, engine_id)

    def _snapshot_locked(self) -> Dict[str, Any]:
        pending = self._pending
        return {
            "tracked": self.tracked,
            "version": self._version,
            "digest": self._digest,
            "physical_to_logical_map": (
                None if self._rows is None else [list(row) for row in self._rows]
            ),
            "pending": (
                None
                if pending is None
                else {
                    "engine": self._label(pending.engine_id),
                    "base_version": pending.base_version,
                    "layer_ids": list(pending.layer_ids),
                    "seconds": time.monotonic() - pending.started,
                }
            ),
            "dirty": self._dirty,
            "attached": len(self._attached),
            "untracked_consumers": self._untracked_consumers,
        }

    def query(self) -> Dict[str, Any]:
        with self._lock:
            return {"status": OK, "layout": self._snapshot_locked()}

    def attach(self, engine_id: str, label: str) -> Dict[str, Any]:
        with self._lock:
            if engine_id in self._attached:
                return _refused(f"engine {label} is attached already")
            self._attached[engine_id] = label
            logger.info(
                f"[weight_cache] engine {label} attached "
                f"({len(self._attached)} attached, layout version {self._version})"
            )
            return {"status": OK, "layout": self._snapshot_locked()}

    def detach(self, engine_id: str) -> None:
        """The engine's attach connection closed: it exited, however it did."""
        with self._lock:
            label = self._label(engine_id)
            self._attached.pop(engine_id, None)
            pending = self._pending
            if pending is not None and pending.engine_id == engine_id:
                self._pending = None
                self._dirty = (
                    f"engine {label} went away while it moved the routed experts "
                    f"of {_layers(pending.layer_ids)} from layout version "
                    f"{pending.base_version}, so they may hold a mix of two layouts"
                )
                logger.error(
                    f"[weight_cache] expert layout is now unknown: {self._dirty}"
                )
            else:
                logger.info(
                    f"[weight_cache] engine {label} detached "
                    f"({len(self._attached)} attached)"
                )

    def _check_mover_locked(self, engine_id: str) -> Optional[Dict[str, Any]]:
        if not self.tracked:
            return _refused(
                "this weight cache daemon does not track the expert layout "
                "(it runs without --enable-eplb)"
            )
        if self._dirty is not None:
            return _refused(f"the expert layout is unknown: {self._dirty}")
        if engine_id not in self._attached:
            return _refused("the engine is not attached to this weight cache daemon")
        if self._untracked_consumers:
            return _refused(
                f"{self._untracked_consumers} engine(s) fetched these weights "
                f"without following the expert layout and may still map them, so "
                f"no expert may move until the weight cache daemon restarts"
            )
        return None

    def _parse_layer_ids_locked(self, layer_ids: Any) -> Optional[List[int]]:
        if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
            return None
        if not all(_is_int(x) and 0 <= x < len(self._rows) for x in layer_ids):
            return None
        if len(set(layer_ids)) != len(layer_ids):
            return None
        return list(layer_ids)

    def begin(
        self, engine_id: str, base_version: Any, layer_ids: Any
    ) -> Dict[str, Any]:
        """Mark a move of ``layer_ids`` pending, if this engine may move them now."""
        with self._lock:
            refusal = self._check_mover_locked(engine_id)
            if refusal is not None:
                return refusal
            if base_version != self._version:
                return _refused(
                    f"the engine moves experts from layout version {base_version}, "
                    f"but the weights are in version {self._version}"
                )
            layer_ids = self._parse_layer_ids_locked(layer_ids)
            if layer_ids is None:
                return _refused("the move names no valid layers")
            if self._pending is not None:
                if self._pending.engine_id == engine_id:
                    return _refused("the engine has a move pending already")
                return _busy(
                    f"engine {self._label(self._pending.engine_id)} is moving experts"
                )
            others = len(self._attached) - 1
            if others:
                return _busy(f"{others} other engine(s) map these weights")
            self._pending = _PendingMove(
                engine_id, base_version, layer_ids, time.monotonic()
            )
            return {"status": OK}

    def _pending_matches_locked(
        self, engine_id: str, base_version: Any, layer_ids: Any
    ) -> bool:
        pending = self._pending
        return (
            pending is not None
            and pending.engine_id == engine_id
            and pending.base_version == base_version
            and isinstance(layer_ids, (list, tuple))
            and list(layer_ids) == pending.layer_ids
        )

    def commit(
        self, engine_id: str, base_version: Any, layer_ids: Any, rows: Any
    ) -> Dict[str, Any]:
        """Store the new rows of the pending move: the weights have moved."""
        with self._lock:
            if not self._pending_matches_locked(engine_id, base_version, layer_ids):
                return _refused("no such move is pending")
            num_physical_experts = len(self._rows[0]) if self._rows else 0
            if (
                not isinstance(rows, (list, tuple))
                or len(rows) != len(layer_ids)
                or not all(
                    isinstance(row, (list, tuple))
                    and len(row) == num_physical_experts
                    and all(_is_int(x) for x in row)
                    for row in rows
                )
            ):
                # The pending move stays: the weights moved, but to a layout
                # the daemon cannot record. The engine going away makes it dirty.
                return _refused(
                    f"the new rows do not fit {len(layer_ids)} layer(s) of "
                    f"{num_physical_experts} physical experts"
                )
            for layer_id, row in zip(layer_ids, rows):
                self._rows[layer_id] = list(row)
            self._digest = layout_digest(self._rows)
            self._version += 1
            self._pending = None
            logger.info(
                f"[weight_cache] expert layout version {self._version - 1} -> "
                f"{self._version}: engine {self._label(engine_id)} moved the "
                f"routed experts of {_layers(layer_ids)}"
            )
            return {"status": OK, "version": self._version, "digest": self._digest}

    def abort(
        self, engine_id: str, base_version: Any, layer_ids: Any
    ) -> Dict[str, Any]:
        """Drop the pending move: another rank cannot move, so nothing moved."""
        with self._lock:
            if not self._pending_matches_locked(engine_id, base_version, layer_ids):
                return _refused("no such move is pending")
            self._pending = None
            return {"status": OK}

    def check_fetch(self, layout: Optional[Dict[str, Any]]) -> Optional[str]:
        """Why fetch_state must not serve the weights, or None to serve them.

        ``layout`` is the engine's ``expert_layout`` field: its engine_id,
        version, and the digest of the map it routes with. Engines without the
        field (older ones) route with their own initial layout.
        """
        with self._lock:
            if not self.tracked:
                return None
            if self._dirty is not None:
                return (
                    f"the expert layout of its weights is unknown: {self._dirty}. "
                    f"{_RESTART_HINT}"
                )
            if self._pending is not None:
                return (
                    f"engine {self._label(self._pending.engine_id)} is moving the "
                    f"routed experts of {_layers(self._pending.layer_ids)}; retry "
                    f"when it is done"
                )
            if layout is None:
                if self._version:
                    return (
                        f"EPLB moved the routed experts of its weights (expert layout "
                        f"version {self._version}), and an engine that does not "
                        f"follow the daemon's expert layout would route them with "
                        f"its initial layout. Upgrade the engine, or restart the "
                        f"weight cache daemon so that the weights are loaded from "
                        f"disk again."
                    )
                self._untracked_consumers += 1
                logger.warning(
                    "[weight_cache] an engine that does not follow the expert "
                    "layout fetched the weights; no expert may move until this "
                    "daemon restarts"
                )
                return None
            if layout.get("digest") != self._digest:
                return (
                    f"the expert layout the engine routes with (version "
                    f"{layout.get('version')}, digest {str(layout.get('digest'))[:12]}) "
                    f"is not the one its weights are in (version {self._version}, "
                    f"digest {self._digest[:12]})"
                )
            if layout.get("engine_id") not in self._attached:
                self._untracked_consumers += 1
                logger.warning(
                    "[weight_cache] an engine that is not attached fetched the "
                    "weights; no expert may move until this daemon restarts"
                )
            return None


def handle_expert_layout_request(
    registry: ExpertLayoutRegistry, req: Dict[str, Any]
) -> Dict[str, Any]:
    """Daemon side: the reply to one of REQUEST_TYPES."""
    engine_id = req.get("engine_id")
    if not isinstance(engine_id, str) or not engine_id:
        return _refused("the request names no engine_id")
    req_type = req.get("type")
    if req_type == ATTACH:
        return registry.attach(engine_id, str(req.get("label") or engine_id))
    if req_type == QUERY:
        return registry.query()
    if req_type == BEGIN:
        return registry.begin(engine_id, req.get("base_version"), req.get("layer_ids"))
    if req_type == COMMIT:
        return registry.commit(
            engine_id, req.get("base_version"), req.get("layer_ids"), req.get("rows")
        )
    if req_type == ABORT:
        return registry.abort(engine_id, req.get("base_version"), req.get("layer_ids"))
    raise ValueError(f"not an expert layout request: {req_type!r}")


class ExpertLayoutPresence:
    """Daemon side: holds the attach connections, and detaches an engine from
    the registry when its connection closes."""

    def __init__(self, registry: ExpertLayoutRegistry):
        self._registry = registry
        self._selector = selectors.DefaultSelector()
        self._wakeup_recv, self._wakeup_send = socket.socketpair()
        self._wakeup_recv.setblocking(False)
        self._selector.register(self._wakeup_recv, selectors.EVENT_READ, None)
        self._incoming: List[Tuple[socket.socket, str]] = []
        self._incoming_lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="weight-cache-expert-layout-presence", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def watch(self, conn: socket.socket, engine_id: str) -> None:
        """Take over an attach connection until the engine closes it."""
        conn.setblocking(False)
        with self._incoming_lock:
            self._incoming.append((conn, engine_id))
        self._wakeup_send.send(b"\0")

    def stop(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        self._wakeup_send.send(b"\0")
        self._thread.join(timeout=5)
        self._wakeup_send.close()

    def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                for key, _ in self._selector.select(timeout=1.0):
                    if key.data is None:
                        self._drain_wakeups()
                    else:
                        self._check(key.fileobj, key.data)
                with self._incoming_lock:
                    incoming, self._incoming = self._incoming, []
                for conn, engine_id in incoming:
                    self._selector.register(conn, selectors.EVENT_READ, engine_id)
        finally:
            for key in list(self._selector.get_map().values()):
                key.fileobj.close()
            self._selector.close()

    def _drain_wakeups(self) -> None:
        try:
            while self._wakeup_recv.recv(4096):
                pass
        except (BlockingIOError, InterruptedError):
            pass

    def _check(self, conn: socket.socket, engine_id: str) -> None:
        try:
            data = conn.recv(1)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b""
        if data:
            # An engine never writes on its attach connection after attaching.
            logger.warning(
                f"[weight_cache] engine {engine_id} wrote on its attach "
                f"connection; treating it as gone"
            )
        self._selector.unregister(conn)
        conn.close()
        self._registry.detach(engine_id)


# ---------------------------------------------------------------------------
# Engine side
# ---------------------------------------------------------------------------


def _connect(socket_path: str, timeout: float) -> Optional[socket.socket]:
    """Connect to the daemon socket; None when no daemon listens there.

    Same rule as the weight loader: only a socket node owned by this user.
    """
    try:
        st = os.lstat(socket_path)
    except FileNotFoundError:
        return None
    if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError(
            f"[weight_cache] Refusing to connect: {socket_path} is not a socket "
            f"owned by this user."
        )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
    except FileNotFoundError:
        sock.close()
        return None
    except ConnectionRefusedError as e:
        sock.close()
        raise RuntimeError(
            f"[weight_cache] The daemon socket exists at {socket_path} but refused "
            f"the connection. The daemon may have crashed after creating the "
            f"socket. Check daemon logs."
        ) from e
    except BaseException:
        sock.close()
        raise
    return sock


class ExpertLayoutClient:
    """Engine side: one rank's link to its daemon's expert layout record."""

    def __init__(self, socket_path: str, label: str, timeout: float = 30.0):
        self.socket_path = socket_path
        self.label = label
        self.timeout = timeout
        self.engine_id = uuid.uuid4().hex
        self._presence: Optional[socket.socket] = None

    @property
    def attached(self) -> bool:
        return self._presence is not None

    def attach(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Attach to the daemon and read its expert layout.

        Returns ("attached", layout), (LEGACY, None) for a daemon without the
        expert layout requests, or (ABSENT, None) when no daemon listens.
        """
        sock = _connect(self.socket_path, self.timeout)
        if sock is None:
            return ABSENT, None
        try:
            send_msg(
                sock, {"type": ATTACH, "engine_id": self.engine_id, "label": self.label}
            )
            reply = recv_msg(sock)
        except BaseException:
            sock.close()
            raise
        if reply.get("status") == OK:
            # Kept open and never used again: the daemon sees it close when this
            # process exits.
            self._presence = sock
            return "attached", reply["layout"]
        sock.close()
        if reply.get("status") == "error" and str(reply.get("message")).startswith(
            "Unknown request type"
        ):
            return LEGACY, None
        raise RuntimeError(
            f"[weight_cache] The daemon at {self.socket_path} did not attach this "
            f"engine: {reply.get('reason') or reply}"
        )

    def request(self, req_type: str, **fields: Any) -> Dict[str, Any]:
        """One request on its own connection. A daemon that is gone, or does
        not answer, reads as a refusal: callers are collective, so every rank
        must get a reply to agree on."""
        try:
            sock = _connect(self.socket_path, self.timeout)
            if sock is None:
                return _refused(f"no daemon listens at {self.socket_path} any more")
            try:
                send_msg(
                    sock, {"type": req_type, "engine_id": self.engine_id, **fields}
                )
                return recv_msg(sock)
            finally:
                sock.close()
        except Exception as e:
            return _refused(f"the daemon at {self.socket_path} did not answer: {e}")

    def close(self) -> None:
        if self._presence is not None:
            self._presence.close()
            self._presence = None


AllGather = Callable[[Any], List[Any]]


class ExpertLayoutSession:
    """Engine side: what this rank agreed on with the others at startup, and
    the two-phase move of experts in the daemon's weights.

    begin(), commit() and the first rebalance_mode() are collective: every rank
    calls them at the same point, as EPLBManager.rebalance() does.
    """

    def __init__(
        self,
        *,
        client: ExpertLayoutClient,
        kind: str,
        version: Optional[int],
        all_gather: AllGather,
        rebalance_off_reason: Optional[str],
    ):
        self.client = client
        self.kind = kind
        self.version = version
        # Digest of the map this engine routes with; set once it is built.
        self.digest: Optional[str] = None
        # Set by the weight loader: True when the weights are the daemon's,
        # False when they were loaded from disk instead.
        self.weights_shared: Optional[bool] = None
        self._all_gather = all_gather
        self._off_reason = rebalance_off_reason
        self._mode: Optional[Tuple[str, Optional[str]]] = None

    def fetch_fields(self) -> Dict[str, Any]:
        """Extra fetch_state fields, so the daemon can check this engine's layout."""
        return {
            "expert_layout": {
                "engine_id": self.client.engine_id if self.client.attached else None,
                "version": self.version,
                "digest": self.digest,
            }
        }

    def rebalance_mode(self) -> Tuple[str, Optional[str]]:
        """How a rebalance may move experts: ("private", None) when the weights
        are this engine's own, ("coordinated", None) through begin() and
        commit(), or ("off", reason)."""
        if self._mode is None:
            self._mode = self._agree_on_mode()
        if self._mode[0] == "coordinated" and self._off_reason is not None:
            return "off", self._off_reason
        return self._mode

    def _agree_on_mode(self) -> Tuple[str, Optional[str]]:
        if self.weights_shared is False:
            local = ("private", None)
        elif self.weights_shared is None:
            local = ("off", "the weights are not loaded")
        elif self._off_reason is not None:
            local = ("off", self._off_reason)
        else:
            local = ("coordinated", None)
        results = self._all_gather(local)
        modes = {mode for mode, _ in results}
        if len(modes) == 1:
            return results[0]
        reasons = [reason for mode, reason in results if mode == "off"]
        return "off", (
            reasons[0]
            if reasons
            else "some ranks loaded the weights from disk and some from the daemon"
        )

    def begin(self, layer_ids: List[int]) -> Optional[str]:
        """Mark a move of ``layer_ids`` pending on every rank's daemon.

        Returns None when every rank may move them now, else why not. A refusal
        (not merely busy) turns rebalancing off for good.
        """
        layer_ids = list(layer_ids)
        reply = self.client.request(
            BEGIN, base_version=self.version, layer_ids=layer_ids
        )
        status = reply.get("status")
        results = self._all_gather((status, reply.get("reason")))
        if all(s == OK for s, _ in results):
            return None
        if status == OK:
            self.client.request(ABORT, base_version=self.version, layer_ids=layer_ids)
        reason = "; ".join(
            f"rank {rank}: {r}" for rank, (s, r) in enumerate(results) if s != OK
        )
        if any(s not in (OK, BUSY) for s, _ in results):
            self._off_reason = reason
        return reason

    def commit(self, layer_ids: List[int], physical_to_logical_map) -> Optional[str]:
        """Store the moved layers' rows on every rank's daemon.

        ``physical_to_logical_map`` is this engine's whole map after the move,
        on CPU. Returns None when every daemon recorded the move and agrees
        with this engine, else why not; rebalancing is then off for good.
        """
        layer_ids = list(layer_ids)
        expected_digest = layout_digest(physical_to_logical_map.tolist())
        reply = self.client.request(
            COMMIT,
            base_version=self.version,
            layer_ids=layer_ids,
            rows=physical_to_logical_map[layer_ids].tolist(),
        )
        if reply.get("status") != OK:
            local = (False, None, reply.get("reason"))
        elif reply.get("digest") != expected_digest:
            local = (False, reply.get("version"), "the daemon recorded another layout")
        else:
            local = (True, reply.get("version"), None)
        results = self._all_gather(local)
        if all(ok for ok, _, _ in results) and len({v for _, v, _ in results}) == 1:
            self.version = reply["version"]
            self.digest = expected_digest
            return None
        reason = "; ".join(
            f"rank {rank}: {r or f'version {v}'}"
            for rank, (ok, v, r) in enumerate(results)
            if not ok or v != results[0][1]
        )
        self._off_reason = reason
        return reason


def decide_expert_layout(
    summaries: Sequence[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """The startup decision, from every rank's attach summary in rank order.

    Returns whether to build the metadata from the daemons' layout, and why
    rebalancing cannot go through the daemons (None if it can). Raises when
    the layout of the shared weights is unknown or the ranks disagree on it.
    """
    for rank, summary in enumerate(summaries):
        if summary.get("error"):
            raise RuntimeError(
                f"[weight_cache] Rank {rank} could not read the expert layout of "
                f"its weight cache daemon: {summary['error']}"
            )
        if summary["dirty"]:
            raise RuntimeError(
                f"[weight_cache] The weight cache daemon of rank {rank} does not "
                f"know the expert layout of its weights: {summary['dirty']}. "
                f"{_RESTART_HINT}"
            )
    tracked = [(rank, s) for rank, s in enumerate(summaries) if s["kind"] == TRACKED]
    if len(tracked) == len(summaries):
        if len({(s["version"], s["digest"]) for _, s in tracked}) > 1:
            versions = ", ".join(
                f"rank {rank}: version {s['version']}" for rank, s in tracked
            )
            raise RuntimeError(
                f"[weight_cache] The weight cache daemons disagree on the expert "
                f"layout of their weights ({versions}), so it is unknown. "
                f"{_RESTART_HINT}"
            )
        return True, None
    for rank, summary in tracked:
        if summary["version"]:
            raise RuntimeError(
                f"[weight_cache] EPLB moved the routed experts in the weights of "
                f"rank {rank}'s weight cache daemon (expert layout version "
                f"{summary['version']}), but not every rank's daemon tracks the "
                f"layout, so this engine cannot route with it. {_RESTART_HINT}"
            )
    kinds = {s["kind"] for s in summaries}
    if LEGACY in kinds:
        return False, (
            "the weight cache daemon does not track the expert layout (an older "
            "version), so a rebalance would move the experts under every engine "
            "that maps them"
        )
    if UNTRACKED in kinds:
        return False, (
            "the weight cache daemon does not track the expert layout (start it "
            "with --enable-eplb)"
        )
    if kinds == {ABSENT}:
        return False, "this engine did not attach to a weight cache daemon at startup"
    return False, "not every rank found a weight cache daemon at startup"


def start_expert_layout_session(
    *,
    socket_path: str,
    label: str,
    all_gather: AllGather,
    pending_move_timeout: float = PENDING_MOVE_TIMEOUT,
) -> Tuple[ExpertLayoutSession, Optional[List[List[int]]]]:
    """Attach this rank to its daemon and agree with the other ranks.

    Collective. Returns the session and the daemons' physical_to_logical_map
    if the engine must route with it, else None.
    """
    client = ExpertLayoutClient(socket_path, label)
    try:
        try:
            kind, layout = client.attach()
            if layout is not None:
                layout = _wait_for_pending_move(client, layout, pending_move_timeout)
                kind = TRACKED if layout["tracked"] else UNTRACKED
            summary = {
                "kind": kind,
                "dirty": None if layout is None else layout["dirty"],
                "version": None if layout is None else layout["version"],
                "digest": None if layout is None else layout["digest"],
            }
        except Exception as e:
            # Still join the all_gather, so that every rank stops, not just this one.
            summary = {"kind": None, "error": str(e), "dirty": None}
        adopt, off_reason = decide_expert_layout(all_gather(summary))
    except BaseException:
        client.close()
        raise
    session = ExpertLayoutSession(
        client=client,
        kind=kind,
        version=summary["version"],
        all_gather=all_gather,
        rebalance_off_reason=off_reason,
    )
    return session, layout["physical_to_logical_map"] if adopt else None


def _wait_for_pending_move(
    client: ExpertLayoutClient, layout: Dict[str, Any], timeout: float
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    while layout["pending"] is not None and layout["dirty"] is None:
        if time.monotonic() > deadline:
            pending = layout["pending"]
            raise RuntimeError(
                f"[weight_cache] Engine {pending['engine']} has been moving the "
                f"routed experts of {_layers(pending['layer_ids'])} "
                f"in the weight cache daemon's weights for "
                f"{pending['seconds']:.0f} s; this engine cannot learn their "
                f"layout until the move ends."
            )
        time.sleep(_PENDING_MOVE_POLL_INTERVAL)
        reply = client.request(QUERY)
        if reply.get("status") != OK:
            raise RuntimeError(
                f"[weight_cache] The daemon at {client.socket_path} did not report "
                f"its expert layout: {reply.get('reason')}"
            )
        layout = reply["layout"]
    return layout


def _world_all_gather(obj: Any) -> List[Any]:
    import torch.distributed as dist

    group = get_parallel().world_group.cpu_group
    gathered = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, obj, group=group)
    return gathered


def adopt_weight_cache_expert_layout(
    metadata, *, model_config, moe_ep_rank: int, gpu_id: int
):
    """The expert location metadata this engine must route with.

    With the weight cache on, attach to this rank's daemon and, when it tracks
    the layout of its weights, build the metadata from the daemon's map: the
    same path as ``--init-expert-location`` with that map. Collective; also
    publishes the session for the weight loader and EPLBManager.
    """
    if metadata is None or get_model().weight_cache_mode == "off":
        return metadata

    from sglang.srt.eplb.expert_location import ExpertLocationMetadata
    from sglang.srt.platforms import current_platform

    socket_path = get_model().weight_cache_socket or get_socket_path(
        current_platform.get_device_uuid(gpu_id)
    )
    session, rows = start_expert_layout_session(
        socket_path=socket_path,
        label=f"{socket.gethostname()}:{os.getpid()} tp_rank={get_parallel().tp_rank}",
        all_gather=_world_all_gather,
    )
    try:
        if rows is not None and rows != metadata.physical_to_logical_map_cpu.tolist():
            _check_daemon_layout_fits(rows, metadata, version=session.version)
            metadata = ExpertLocationMetadata.init_by_mapping(
                model_config, physical_to_logical_map=rows, moe_ep_rank=moe_ep_rank
            )
            logger.info(
                f"[weight_cache] Routing with the weight cache daemon's expert "
                f"layout (version {session.version}) instead of the initial one."
            )
    except BaseException:
        session.client.close()
        raise
    session.digest = layout_digest(metadata.physical_to_logical_map_cpu.tolist())
    set_expert_layout_session(session)
    return metadata


def _check_daemon_layout_fits(rows, metadata, *, version: int) -> None:
    shape = (len(rows), len(rows[0]) if rows else 0)
    expected_shape = tuple(metadata.physical_to_logical_map_cpu.shape)
    if shape != expected_shape:
        raise RuntimeError(
            f"[weight_cache] The weight cache daemon's expert layout has "
            f"{shape[0]} layers x {shape[1]} physical experts, but this engine "
            f"has {expected_shape[0]} x {expected_shape[1]}. Start the daemon "
            f"and the engine with the same --ep-num-redundant-experts."
        )
    if get_exec().moe.ep_dispatch_algorithm is None:
        raise RuntimeError(
            f"[weight_cache] The weight cache daemon's routed experts are in an "
            f"expert layout (version {version}) other than this engine's initial "
            f"one, and without --enable-eplb (or --init-expert-location) this "
            f"engine routes as if every expert sat in its initial slot. Start "
            f"the engine with --enable-eplb, or restart the weight cache daemon."
        )


def get_expert_layout_session() -> Optional[ExpertLayoutSession]:
    return get_resources().expert_layout_session


def set_expert_layout_session(session: Optional[ExpertLayoutSession]) -> None:
    get_resources().expert_layout_session = session


def mark_expert_weights_shared(shared: bool) -> None:
    """Weight loader: record whether the weights are the daemon's (True) or
    were loaded from disk (False)."""
    session = get_expert_layout_session()
    if session is not None:
        session.weights_shared = shared

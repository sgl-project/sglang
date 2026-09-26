"""Serialize peer recovery with the batches that own its transfer handles."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from typing import Callable

logger = logging.getLogger(__name__)


class _WorkerState(threading.local):
    batch = None


class PeerRecovery:
    def __init__(
        self, agent, metadata: Callable, rebuild: Callable, shutdown: Callable
    ):
        self.agent = agent
        self.metadata = metadata
        self.rebuild = rebuild
        self.shutdown = shutdown
        self._registry_lock = threading.Lock()
        self._locks = defaultdict(threading.RLock)
        self._dirty = set()
        self._workers = _WorkerState()
        # Keep handles alive if status cannot be established before shutdown.
        self._quarantined = []

    def lock(self, peer):
        with self._registry_lock:
            return self._locks[peer]

    @property
    def batch(self):
        return self._workers.batch

    def begin(self, peers):
        assert self.batch is None
        batch = TransferBatch(self, sorted(set(peers)))
        self._workers.batch = batch
        # Publish ownership before any native call can raise.
        batch.acquire()
        return batch

    def end(self):
        if self.batch is not None:
            self.batch.locks.close()
            self._workers.batch = None

    def refresh(self, peer):
        self._dirty.add(peer)
        if self.agent.check_remote_metadata(peer):
            self.agent.remove_remote_agent(peer)
        self.agent.add_remote_agent(self.metadata(peer))
        self.rebuild(peer)
        self._dirty.discard(peer)
        logger.info("Reconnected NIXL peer %s", peer)

    def fatal(self, batch, reason):
        self._quarantined.append(batch)
        self.shutdown(reason)
        # The parent shuts down the engine; this worker must not acknowledge
        # buffer reuse or dequeue another chunk while that happens.
        raise SystemExit(reason)


class TransferBatch:
    def __init__(self, recovery, peers):
        self.recovery = recovery
        self.agent = recovery.agent
        self.peers = peers
        self.locks = ExitStack()
        self.handles = []
        self.failed_peers = set()
        self.failed = False
        self.finished = False

    def acquire(self):
        # Multiple decode ranks can belong to one room. A stable order avoids
        # deadlocks between overlapping rank sets on different transfer queues.
        for peer in self.peers:
            self.locks.enter_context(self.recovery.lock(peer))
        for peer in self.peers:
            if peer in self.recovery._dirty or not self.agent.check_remote_metadata(
                peer
            ):
                self.recovery.refresh(peer)

    def post(self, handle, peer):
        assert peer in self.peers
        self.handles.append((handle, peer))
        # transfer() may raise after posting. Its handle still belongs to the
        # barrier even though the caller never received a return value.
        return self.agent.transfer(handle)

    def wait(
        self, *, failure_seen, timeout, poll_interval, disconnect_errors, missing_errors
    ):
        if self.finished:
            return True, self.failed or failure_seen
        self.failed |= failure_seen
        deadline = time.monotonic() + timeout if self.failed else None
        pending = list(self.handles)
        while pending:
            running = []
            for handle, peer in pending:
                try:
                    state = self.agent.check_xfer_state(handle)
                except disconnect_errors:
                    # NIXL reports a terminal backend error and may remove the
                    # remote metadata. Never replay this request's slot indices.
                    if peer not in self.failed_peers:
                        logger.warning(
                            "NIXL peer %s disconnected; retiring its transfer batch",
                            peer,
                        )
                    self.failed_peers.add(peer)
                    self.failed = True
                    continue
                except missing_errors:
                    self.failed = True
                    self.failed_peers.add(peer)
                    # NIXL 1.4.1 checks metadata existence before consulting an
                    # in-progress handle. Restore that lookup, then poll again;
                    # NOT_FOUND itself does not establish completion.
                    try:
                        if not self.agent.check_remote_metadata(peer):
                            self.agent.add_remote_agent(self.recovery.metadata(peer))
                    except Exception as exc:
                        self.recovery.fatal(
                            self, f"Cannot inspect NIXL transfers to {peer}: {exc}"
                        )
                    running.append((handle, peer))
                    continue
                except Exception as exc:
                    self.recovery.fatal(
                        self, f"Cannot inspect NIXL transfer to {peer}: {exc}"
                    )
                if state == "ERR":
                    self.failed = True
                elif state != "DONE":
                    running.append((handle, peer))
            pending = running
            if not pending:
                break
            if self.failed:
                if deadline is None:
                    deadline = time.monotonic() + timeout
                if time.monotonic() >= deadline:
                    self.recovery.fatal(
                        self, "NIXL failed batch did not settle before recovery timeout"
                    )
                time.sleep(poll_interval)
            else:
                time.sleep(0)

        # Every tracked handle has a terminal result. Release before replacing
        # any prepared descriptor that could still be referenced by a handle.
        for handle, peer in self.handles:
            try:
                self.agent.release_xfer_handle(handle)
            except Exception as exc:
                self.recovery.fatal(
                    self, f"Cannot retire NIXL transfer to {peer}: {exc}"
                )
        self.handles.clear()
        self.finished = True
        for peer in self.peers:
            try:
                if peer in self.failed_peers or not self.agent.check_remote_metadata(
                    peer
                ):
                    self.recovery.refresh(peer)
            except Exception:
                # The handles have already been retired. Keep this peer dirty
                # so a later batch retries registration, without killing the
                # transfer thread or blocking other peers on its queue.
                self.failed = True
                self.recovery._dirty.add(peer)
                logger.exception("Failed to reconnect NIXL peer %s; will retry", peer)
        return True, self.failed

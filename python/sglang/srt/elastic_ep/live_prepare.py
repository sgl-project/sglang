"""Background preparation only: this module must never issue PG collectives.

All participants prepare one cohort. The serving thread observes ``poll()``
and agrees on activation through the existing WORLD forward-boundary protocol.
Target-world CUDA graph capture is scheduled separately after admission.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class PreparationStore:
    """Namespace one scale operation and retain NIXL metadata until commit.

    NIXL's Python exchange deletes its own metadata after connecting. A faster
    old rank can finish before a slower joiner reads it. Keeping these small
    records for the lifetime of this operation avoids that race. Failed
    operations cannot currently be retried; a server restart is required.
    """

    def __init__(self, store, prefix: str):
        self.store = store
        self.prefix = prefix + "/"

    def set(self, key, value):
        return self.store.set(self.prefix + key, value)

    def get(self, key):
        return self.store.get(self.prefix + key)

    def check(self, keys):
        return self.store.check([self.prefix + key for key in keys])

    def multi_get(self, keys):
        return self.store.multi_get([self.prefix + key for key in keys])

    def wait(self, keys, timeout=None):
        keys = [self.prefix + key for key in keys]
        if timeout is None:
            return self.store.wait(keys)
        return self.store.wait(keys, timeout)

    def delete_key(self, key):
        # The operation namespace is never reused after a failure.
        return True


class LiveScalePreparation:
    def __init__(
        self,
        store,
        rank: int,
        old_size: int,
        target_size: int,
        timeout: float,
        operation_id: str,
        manifest: str,
    ):
        self.rank = rank
        self.old_size = old_size
        self.target_size = target_size
        self.store = PreparationStore(store, f"elastic_ep/live_prepare/{operation_id}")
        self.timeout = timeout
        self.started = time.monotonic()
        self.connected = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.thread = None
        self.operation_id = operation_id
        self.manifest = manifest

    def start(self, connect: Callable[[], None]) -> None:
        if self.thread is not None:
            raise RuntimeError("Preparation has already started")

        def run():
            try:
                self.store.set(f"manifest/{self.rank}", self.manifest.encode())
                manifest_keys = [f"manifest/{rank}" for rank in range(self.target_size)]
                while not self.store.check(manifest_keys):
                    self._check_errors()
                    if time.monotonic() - self.started > self.timeout:
                        raise TimeoutError("Timed out validating preparation manifests")
                    time.sleep(0.01)
                if any(
                    value.decode() != self.manifest
                    for value in self.store.multi_get(manifest_keys)
                ):
                    raise ValueError("Live scale participant configuration mismatch")
                connect()
                self.connected.set()
                if self.rank < self.old_size:
                    self.mark_ready()
                ready_keys = [f"ready/{rank}" for rank in range(self.target_size)]
                while True:
                    self._check_errors()
                    if self.store.check(ready_keys):
                        break
                    if time.monotonic() - self.started > self.timeout:
                        raise TimeoutError("Timed out preparing the joining cohort")
                    time.sleep(0.01)
                logger.info(
                    "[Elastic EP][live] prepared rank=%d old=%d target=%d elapsed=%.3fs",
                    self.rank,
                    self.old_size,
                    self.target_size,
                    time.monotonic() - self.started,
                )
            except Exception as exc:
                self.error = str(exc)
                try:
                    self.store.set(f"error/{self.rank}", self.error.encode())
                except Exception:
                    logger.exception("Could not publish scale preparation error")
                logger.exception("[Elastic EP][live] preparation failed")
            finally:
                self.done.set()

        self.thread = threading.Thread(
            target=run, name=f"elastic-ep-prepare-{self.rank}", daemon=True
        )
        self.thread.start()

    def _check_errors(self):
        for rank in range(self.target_size):
            key = f"error/{rank}"
            if self.store.check([key]):
                raise RuntimeError(
                    f"Rank {rank} preparation failed: {self.store.get(key).decode()}"
                )

    def mark_ready(self) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)
        if not self.connected.is_set():
            raise RuntimeError("Cannot publish readiness before connections finish")
        self.store.set(f"ready/{self.rank}", b"1")

    def wait_connected(self) -> None:
        """Joiner startup only; serving threads must use poll()."""
        while not self.connected.wait(0.01):
            if self.done.is_set():
                raise RuntimeError(self.error or "Preparation stopped")
            if time.monotonic() - self.started > self.timeout:
                raise TimeoutError("Timed out connecting the joining cohort")
        self._check_errors()

    def poll(self) -> int:
        """-1: failed, 0: still preparing, 1: every participant is ready."""
        if not self.done.is_set():
            return 0
        return -1 if self.error is not None else 1

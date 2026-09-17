# SPDX-License-Identifier: Apache-2.0
"""Single-owner driver for one KVCR core.

Every KVCR API mutation and ``poll_completed`` call happens on the owner
thread. Other threads post commands (closures) and receive results through
per-operation callbacks that also run on the owner thread; they never touch
the core directly. The loop keeps polling while idle so this rank keeps
serving peers as a KVCR source even when it issues no traffic of its own.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

# Consecutive owner-loop faults tolerated before the adapter declares the
# backend unhealthy. Arbitrary; enough to ride out a transient poll error.
_MAX_CONSECUTIVE_FAULTS = 10
_STOP = object()

Command = Callable[["KVCRAdapter"], None]
Completion = Callable[[Mapping[Any, Any]], None]


class KVCRAdapter:
    """Owns a KVCR instance and its progress polling."""

    def __init__(
        self,
        kvcr: Any,
        *,
        poll_interval_s: float,
        name: str,
        on_unhealthy: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._kvcr = kvcr
        self._poll_interval_s = poll_interval_s
        self._commands: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._completions: dict[int, Completion] = {}
        # Work re-checked every loop iteration (device events, deadlines).
        self._tickers: list[Callable[[float], bool]] = []
        self._thread = threading.Thread(
            target=self._run, name=f"kvcr-linker-owner-{name}", daemon=True
        )
        self._failure: BaseException | None = None
        self._on_unhealthy = on_unhealthy
        self._stopped = threading.Event()
        self._started = False
        self._pending_ops = 0
        self._stats_lock = threading.Lock()
        self._inflight_high_water = 0

    # ---- lifecycle (any thread) ----

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self, timeout_s: float) -> bool:
        """Ask the loop to stop and wait for it; True if it stopped."""
        if not self._started:
            return True
        self._commands.put(_STOP)
        self._stopped.wait(timeout=timeout_s)
        return not self._thread.is_alive()

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    @property
    def healthy(self) -> bool:
        return self._failure is None and (not self._started or self._thread.is_alive())

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("KVCR linker owner thread failed") from self._failure

    # ---- submission (any thread) ----

    def post(self, command: Command) -> None:
        """Run ``command`` on the owner thread."""
        self.raise_if_failed()
        self._commands.put(command)

    # ---- owner-thread helpers (only call from commands/tickers) ----

    @property
    def kvcr(self) -> Any:
        return self._kvcr

    def track(self, op_handle: int, completion: Completion) -> None:
        """Deliver the op's entries to ``completion`` once, on the owner thread."""
        self._completions[op_handle] = completion
        self._pending_ops += 1
        with self._stats_lock:
            self._inflight_high_water = max(
                self._inflight_high_water, self._pending_ops
            )

    def add_ticker(self, ticker: Callable[[float], bool]) -> None:
        """Register a per-iteration hook; it returns whether it did work."""
        self._tickers.append(ticker)

    @property
    def pending_ops(self) -> int:
        """Tracked operations KVCR has not completed, including abandoned ones."""
        return self._pending_ops

    @property
    def inflight_high_water(self) -> int:
        with self._stats_lock:
            return self._inflight_high_water

    # ---- loop ----

    def _run(self) -> None:
        faults = 0
        try:
            while True:
                worked = False
                stop = False
                for _ in range(256):
                    try:
                        item = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if item is _STOP:
                        stop = True
                        break
                    worked = True
                    item(self)
                if stop:
                    return
                try:
                    worked |= self._poll()
                    now = time.monotonic()
                    for ticker in self._tickers:
                        worked |= bool(ticker(now))
                except Exception as error:  # noqa: BLE001 - counted below
                    faults += 1
                    logger.warning(
                        "KVCR linker owner loop fault (%d/%d)",
                        faults,
                        _MAX_CONSECUTIVE_FAULTS,
                        exc_info=True,
                    )
                    if faults >= _MAX_CONSECUTIVE_FAULTS:
                        raise
                    continue
                faults = 0
                if not worked:
                    time.sleep(self._poll_interval_s)
        except BaseException as error:  # noqa: BLE001 - surfaced to the scheduler
            self._failure = error
            logger.error("KVCR linker owner thread stopped", exc_info=True)
            if self._on_unhealthy is not None:
                try:
                    self._on_unhealthy(error)
                except Exception:  # noqa: BLE001
                    logger.exception("KVCR linker unhealthy callback failed")
        finally:
            self._stopped.set()

    def _poll(self) -> bool:
        worked = False
        for op_handle, entries in self._kvcr.poll_completed():
            worked = True
            completion = self._completions.pop(op_handle, None)
            if completion is None:
                logger.warning(
                    "KVCR linker dropped completion for untracked op %s", op_handle
                )
                continue
            self._pending_ops -= 1
            completion(entries)
        return worked


def entries_succeeded(entries: Mapping[Any, Any], keys: Iterable[Any]) -> bool:
    return all(
        (entry := entries.get(key)) is not None and entry.success for key in keys
    )

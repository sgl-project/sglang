"""Single-flight EPLB protocol, independent of device and transport APIs.

Only the forward thread owns transitions and installs layouts. Workers publish
results through Futures; an error is voted on by all ranks before any install.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from enum import IntEnum
from typing import Any, Callable, Protocol


class Phase(IntEnum):
    IDLE = 0
    PLANNING = 1
    STAGING = 2
    DISABLED = 3
    FAILED = 4


class RebalanceAborted(RuntimeError):
    """All ranks drained successfully; installed layers remain valid."""


class UnsafeRebalanceFailure(RuntimeError):
    """Quiescence or install integrity is unknown. The worker must terminate."""


class Control(Protocol):
    def vote(
        self, epoch: int, phase: int, layer: int, ready: bool, error: bool
    ) -> tuple[bool, bool]: ...


WeightWriteback = Callable[[], None]


class Runtime(Protocol):
    def prepare(self, placement: Any) -> tuple[int, ...]: ...

    def record_source_ready(self) -> None: ...

    def stage(self, layer: int) -> WeightWriteback: ...

    def install(self, layer: int, writeback: WeightWriteback) -> None: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


# Keep the runtime and its tensors alive until the failed process exits.
_FAILED_OWNERS: list[Any] = []


class AsyncEplbCoordinator:
    """Run one plan or layer transfer at a time so staging can be reused.

    A failed plan can be disabled once every worker has drained. An uncertain
    transfer or partial install is fatal because other ranks may own live work.
    """

    def __init__(self, control: Control, runtime: Runtime, timeout: float = 60):
        self.control, self.runtime, self.timeout = control, runtime, timeout
        self.phase = Phase.IDLE
        self.epoch = 0
        self._layers: tuple[int, ...] = ()
        self.layers_installed = 0
        self._future: Future | None = None
        self._error: BaseException | None = None
        self._deadline = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb")
        self._closed = False

    @property
    def active(self) -> bool:
        return self.phase in (Phase.PLANNING, Phase.STAGING)

    def start(self, planner: Callable[[], Any]) -> None:
        if self.phase != Phase.IDLE or self._closed:
            raise RuntimeError("EPLB coordinator is not idle")
        self.epoch += 1
        self._layers = ()
        self.layers_installed = 0
        self._error = None

        def plan_layers():
            return self.runtime.prepare(planner())

        self.phase = Phase.PLANNING
        self._deadline = time.monotonic() + self.timeout
        try:
            self._future = self._executor.submit(plan_layers)
        except Exception as error:
            self._fatal("EPLB could not launch the planner", error)

    def _local_ready(self, layer: int) -> bool:
        if self._error is not None:
            return False
        try:
            if self._future is not None and self._future.done():
                self._error = self._future.exception()
                return self._error is None
            if time.monotonic() >= self._deadline:
                operation = (
                    "planning" if self.phase == Phase.PLANNING else f"P2P layer {layer}"
                )
                self._error = TimeoutError(f"EPLB {operation} did not finish")
        except Exception as error:
            self._error = error
        return False

    def _all_ranks_ready(self, layer: int) -> bool:
        ready = self._local_ready(layer)
        try:
            all_ready, any_error = self.control.vote(
                self.epoch, int(self.phase), layer, ready, self._error is not None
            )
        except Exception as error:
            self._fatal("EPLB control agreement failed", error)
        if any_error:
            cause = self._error
            if self.phase == Phase.STAGING:
                self._fatal("EPLB staging failed; P2P completion unknown", cause)
            self.cancel()
            raise RebalanceAborted("EPLB aborted on at least one rank") from cause
        return all_ready

    def poll(self) -> bool:
        """Advance one phase after all ranks agree at a forward boundary."""
        if not self.active:
            return self.phase == Phase.IDLE
        layer = (
            self._layers[self.layers_installed] if self.phase == Phase.STAGING else -1
        )
        if not self._all_ranks_ready(layer):
            return False

        if self.phase == Phase.PLANNING:
            self._layers = self._future.result()
        else:
            try:
                writeback = self._future.result()
                self.runtime.install(layer, writeback)
            except Exception as error:
                self._fatal("EPLB install failed; layout integrity unknown", error)
            self.layers_installed += 1
        self._future = None

        if self.layers_installed == len(self._layers):
            self.phase = Phase.IDLE
            return True

        self.phase = Phase.STAGING
        self._deadline = time.monotonic() + self.timeout
        try:
            self.runtime.record_source_ready()
            next_layer = self._layers[self.layers_installed]
            self._future = self._executor.submit(self.runtime.stage, next_layer)
        except Exception as error:
            self._error = error
            self._future = None
        return False

    def cancel(self) -> None:
        """Collective: no rank may restart or free storage until all are drained."""
        if self._closed:
            return
        if self.phase == Phase.FAILED:
            raise UnsafeRebalanceFailure("Cannot reuse a failed EPLB coordinator")
        local_error = None
        try:
            if self._future is not None:
                # Distinguish an unfinished Future from a worker that finished
                # by raising TimeoutError (for example, a failed CPU plan).
                remaining = max(0.0, self._deadline - time.monotonic())
                if not wait((self._future,), timeout=remaining).done:
                    raise TimeoutError("EPLB worker did not drain")
                try:
                    self._future.result()
                except Exception:
                    if self.phase == Phase.STAGING:
                        # A submission can fail after enqueuing work on peers.
                        raise
            if self.phase == Phase.STAGING and self._error is not None:
                raise self._error
            self.runtime.synchronize()
        except Exception as error:
            local_error = error
        try:
            # Keep the same collective as a forward vote. If a peer is still
            # polling, the phase mismatch is reported on both ranks.
            drained, _ = self.control.vote(
                self.epoch, Phase.DISABLED, -1, local_error is None, False
            )
        except Exception as error:
            self._fatal("EPLB drain agreement failed", error)
        if not drained:
            self._fatal("EPLB could not drain every rank", local_error)
        self._future = None
        self.phase = Phase.DISABLED

    def close(self) -> None:
        if self._closed:
            return
        self.cancel()
        try:
            self.runtime.close()
        except Exception as error:
            self._fatal("EPLB transport close failed", error)
        self._executor.shutdown(wait=True)
        self._closed = True

    def reset(self) -> None:
        self.cancel()
        self._error = None
        self.phase = Phase.IDLE

    def _fatal(self, message: str, cause: BaseException | None) -> None:
        self.phase = Phase.FAILED
        _FAILED_OWNERS.append(self)
        raise UnsafeRebalanceFailure(message) from cause

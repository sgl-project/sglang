"""Single-flight EPLB protocol, independent of device and transport APIs.

Only the forward thread owns transitions and installs layouts. Workers publish
results through Futures; an error is voted on by all ranks before any install.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from enum import IntEnum
from typing import Any, Callable, NoReturn, Protocol


class Phase(IntEnum):
    IDLE = 0
    PLANNING = 1
    STAGING = 2
    DISABLED = 3
    FAILED = 4


class ControlPoint(IntEnum):
    """Vote identifiers outside the nonnegative MoE layer IDs."""

    NO_LAYER = -1
    RECORDER_WINDOW = -2
    PREFILL_ADMISSION = -3
    SNAPSHOT_ADMISSION = -4
    SNAPSHOT_PREPARE = -5
    SNAPSHOT_COMPLETE = -6


class RebalanceAborted(RuntimeError):
    """All ranks drained successfully; installed layers remain valid."""


class UnsafeRebalanceFailure(RuntimeError):
    """Quiescence or install integrity is unknown. The worker must terminate."""


class Control(Protocol):
    wait_seconds: float

    # layer is a nonnegative layer ID or one of the reserved ControlPoint values.
    def vote(
        self, epoch: int, phase: int, layer: int, ready: bool, error: bool
    ) -> tuple[bool, bool]: ...

    def close(self) -> None: ...


WeightWriteback = Callable[[], None]


class Runtime(Protocol):
    def prepare(self, placement: Any) -> tuple[int, ...]: ...

    def record_source_ready(self) -> Future[None] | None: ...

    def stage(self, layer: int, submitted: Future[None] | None) -> WeightWriteback: ...

    def install(self, layer: int, writeback: WeightWriteback) -> None: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


# Keep the runtime and its tensors alive until the failed process exits.
_FAILED_OWNERS: list[AsyncEplbCoordinator] = []


class AsyncEplbCoordinator:
    """Run one plan or layer transfer at a time.

    A failed plan can be disabled once every worker has drained. An uncertain
    transfer or partial install is fatal because other ranks may own live work.
    """

    def __init__(
        self,
        control: Control,
        runtime: Runtime,
        timeout: float = 60,
        *,
        planning_drain_timeout: float = 5,
    ):
        self._control, self.runtime, self.timeout = control, runtime, timeout
        self._planning_drain_timeout = planning_drain_timeout
        self.phase = Phase.IDLE
        self.epoch = 0
        self._layers: tuple[int, ...] = ()
        self.layers_installed = 0
        self.submission_wait_seconds = 0.0
        self._future: Future[tuple[int, ...]] | Future[WeightWriteback] | None = None
        self._error: BaseException | None = None
        self._deadline = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb")
        self._closed = False

    @property
    def active(self) -> bool:
        return self.phase in (Phase.PLANNING, Phase.STAGING)

    @property
    def control_wait_seconds(self) -> float:
        return self._control.wait_seconds

    def vote(self, point: int, ready: bool, error: bool = False) -> tuple[bool, bool]:
        """Vote at the forward thread's current epoch and phase."""
        try:
            return self._control.vote(self.epoch, self.phase, point, ready, error)
        except Exception as exc:
            self._fatal("EPLB control agreement failed", exc)

    def start(self, planner: Callable[[], Any]) -> None:
        if self.phase != Phase.IDLE or self._closed:
            raise RuntimeError("EPLB coordinator is not idle")
        self.epoch += 1
        self._layers = ()
        self.layers_installed = 0
        self._control.wait_seconds = 0.0
        self.submission_wait_seconds = 0.0
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
        all_ready, any_error = self.vote(layer, ready, self._error is not None)
        if any_error:
            cause = self._error
            if self.phase == Phase.STAGING:
                self._fatal("EPLB staging failed; P2P completion unknown", cause)
            self.cancel()
            raise RebalanceAborted("EPLB aborted on at least one rank") from cause
        return all_ready

    def poll(self) -> bool:
        """Advance at a forward boundary. Return True when the coordinator is idle."""
        if not self.active:
            return self.phase == Phase.IDLE
        planning = self.phase == Phase.PLANNING
        vote_layer = (
            ControlPoint.NO_LAYER if planning else self._layers[self.layers_installed]
        )
        if not self._all_ranks_ready(vote_layer):
            return False

        future = self._future
        assert future is not None
        if planning:
            self._layers = future.result()
        else:
            try:
                self.runtime.install(vote_layer, future.result())
                self.layers_installed += 1
            except Exception as error:
                self._fatal("EPLB install failed; layout integrity unknown", error)
        self._future = None

        if self.layers_installed == len(self._layers):
            self.phase = Phase.IDLE
            return True

        # Stage only the next layer. The runtime reuses the same device buffers.
        self._start_stage(self._layers[self.layers_installed])
        return False

    def _start_stage(self, layer: int) -> None:
        self.phase = Phase.STAGING
        self._deadline = time.monotonic() + self.timeout
        try:
            submitted = self.runtime.record_source_ready()
            staged = self._executor.submit(self.runtime.stage, layer, submitted)
        except Exception as error:
            # Peers may already be waiting for our transfers on this layer.
            self._fatal(f"EPLB could not launch transfers for layer {layer}", error)
        self._future = staged
        if submitted is not None:
            self._wait_for_submission(submitted, staged)

    def _wait_for_submission(
        self, submitted: Future[None], staged: Future[WeightWriteback]
    ) -> None:
        # Decode can continue after local P2P submission. Publication still
        # waits for device completion and agreement across ranks.
        start = time.perf_counter()
        try:
            done, _ = wait(
                (submitted, staged),
                timeout=max(0.0, self._deadline - time.monotonic()),
                return_when=FIRST_COMPLETED,
            )
            if staged in done:
                staged.result()
            if submitted not in done:
                raise TimeoutError("EPLB did not submit the layer transfers")
        except Exception as error:
            self._fatal("EPLB could not order P2P before decode", error)
        finally:
            self.submission_wait_seconds += time.perf_counter() - start

    def _drain_local_work(self) -> None:
        future = self._future
        if future is not None:
            remaining = max(0.0, self._deadline - time.monotonic())
            if self.phase == Phase.PLANNING:
                # The CPU planner may finish shortly after its work deadline.
                remaining = max(remaining, self._planning_drain_timeout)
            if not wait((future,), timeout=remaining).done:
                raise TimeoutError("EPLB worker did not drain")
            try:
                future.result()
            except Exception:
                if self.phase == Phase.STAGING:
                    # A submission can fail after enqueuing work on peers.
                    raise
        if self.phase == Phase.STAGING and self._error is not None:
            raise self._error
        self.runtime.synchronize()

    def cancel(self) -> None:
        """Collective: no rank may restart or free storage until all are drained."""
        if self._closed:
            return
        if self.phase == Phase.FAILED:
            raise UnsafeRebalanceFailure("Cannot reuse a failed EPLB coordinator")
        local_error = None
        try:
            self._drain_local_work()
        except Exception as error:
            local_error = error
        try:
            # Keep the same collective as a forward vote. If a peer is still
            # polling, the phase mismatch is reported on both ranks.
            drained, _ = self._control.vote(
                self.epoch,
                Phase.DISABLED,
                ControlPoint.NO_LAYER,
                local_error is None,
                False,
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
            self._control.close()
        except Exception as error:
            self._fatal("EPLB transport close failed", error)
        self._executor.shutdown(wait=True)
        self._closed = True

    def reset(self) -> None:
        self.cancel()
        self._error = None
        self.phase = Phase.IDLE

    def _fatal(self, message: str, cause: BaseException | None) -> NoReturn:
        self.phase = Phase.FAILED
        _FAILED_OWNERS.append(self)
        raise UnsafeRebalanceFailure(message) from cause

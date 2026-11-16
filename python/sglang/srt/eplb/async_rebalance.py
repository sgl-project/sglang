"""Single-flight EPLB protocol, independent of device and transport APIs.

Only the forward thread owns transitions and installs layouts. Workers publish
results through Futures; an error is voted on by all ranks before any install.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class LayerPullPlan:
    local_copies: tuple[tuple[int, int], ...]
    reads: tuple[tuple[int, int, int], ...]  # source rank, source slot, staging slot
    installs: tuple[tuple[int, int], ...]  # staging slot, destination slot

    def install_ranges(self) -> tuple[tuple[int, int, int], ...]:
        """Combine adjacent source/destination slots without copying untouched slots."""
        ranges = []
        for source, destination in self.installs:
            if ranges:
                prev_source, prev_destination, length = ranges[-1]
                if (
                    source == prev_source + length
                    and destination == prev_destination + length
                ):
                    ranges[-1] = (prev_source, prev_destination, length + 1)
                    continue
            ranges.append((source, destination, 1))
        return tuple(ranges)


def plan_layer_pull(
    old: tuple[int, ...],
    new: tuple[int, ...],
    rank: int,
    world_size: int,
    ranks_per_node: int,
) -> LayerPullPlan:
    """Stage every changed destination before overwriting any live source."""
    if world_size < 1 or not 0 <= rank < world_size or ranks_per_node < 1:
        raise ValueError("Invalid EPLB topology")
    if len(old) != len(new) or not old or len(old) % world_size:
        raise ValueError("EPLB requires equal, rank-divisible physical capacities")
    if min(old) < 0 or set(new) != set(old):
        raise ValueError("An EPLB plan must preserve every logical expert")
    slots = len(old) // world_size
    holders_by_expert = {}
    for slot, expert in enumerate(old):
        holders_by_expert.setdefault(expert, []).append(slot)
    local_copies, reads, installs = [], [], []
    staged = {}
    for dst, expert in enumerate(new[rank * slots : (rank + 1) * slots]):
        if old[rank * slots + dst] == expert:
            continue
        if expert not in staged:
            holders = holders_by_expert[expert]
            local = [i for i in holders if i // slots == rank]
            if local:
                local_copies.append((local[0] % slots, dst))
            else:
                near = [
                    i
                    for i in holders
                    if i // slots // ranks_per_node == rank // ranks_per_node
                ]
                candidates = near or holders
                source = candidates[(rank + expert) % len(candidates)]
                reads.append((source // slots, source % slots, dst))
            staged[expert] = dst
        installs.append((staged[expert], dst))
    return LayerPullPlan(tuple(local_copies), tuple(reads), tuple(installs))


class Phase(IntEnum):
    IDLE = 0
    PLANNING = 1
    PREPARING = 2
    STAGING = 3
    DISABLED = 4
    FAILED = 5


class RebalanceAborted(RuntimeError):
    """All ranks drained successfully; installed layers remain valid."""


class UnsafeRebalanceFailure(RuntimeError):
    """Quiescence or install integrity is unknown. The worker must terminate."""


class Control(Protocol):
    def vote(
        self, epoch: int, phase: int, layer: int, ready: bool, error: bool
    ) -> tuple[bool, bool]: ...

    def drained(self, success: bool) -> bool: ...


class Runtime(Protocol):
    def prepare(self, placement: Any) -> tuple[int, ...]: ...

    def preparation_ready(self) -> bool: ...

    def stage(self, layer: int) -> Callable[[], None]: ...

    def install(self, layer: int) -> None: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


# A timeout must not let Python release memory still reachable by external DMA.
_FAILED_OWNERS: list[Any] = []


class AsyncEplbCoordinator:
    def __init__(self, control: Control, runtime: Runtime, timeout: float = 60):
        self.control, self.runtime, self.timeout = control, runtime, timeout
        self.phase = Phase.IDLE
        self.epoch = 0
        self.layers: tuple[int, ...] = ()
        self.index = 0
        self._future: Future | None = None
        self._error: BaseException | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb")
        self._closed = False

    @property
    def active(self) -> bool:
        return self.phase in (Phase.PLANNING, Phase.PREPARING, Phase.STAGING)

    def start(self, planner: Callable[[], Any]) -> None:
        if self.phase != Phase.IDLE or self._closed:
            raise RuntimeError("EPLB coordinator is not idle")
        self.epoch += 1
        self.layers, self.index, self._error = (), 0, None
        self._future = self._executor.submit(planner)
        self.phase = Phase.PLANNING

    def _future_ready(self) -> bool:
        if self._future is None or not self._future.done():
            return False
        self._error = self._future.exception()
        return True

    def poll(self) -> bool:
        """Called by every EP rank at the same target-forward boundary."""
        if not self.active:
            return self.phase == Phase.IDLE
        layer = self.layers[self.index] if self.phase == Phase.STAGING else -1
        try:
            ready = (
                self.runtime.preparation_ready()
                if self.phase == Phase.PREPARING
                else self._future_ready()
            )
        except Exception as error:
            self._error, ready = error, False
        try:
            all_ready, any_error = self.control.vote(
                self.epoch, int(self.phase), layer, ready, self._error is not None
            )
        except Exception as error:
            self._fatal("EPLB control agreement failed", error)
        if any_error:
            cause = self._error
            self.cancel()
            raise RebalanceAborted("EPLB aborted on at least one rank") from cause
        if not all_ready:
            return False
        if self.phase == Phase.PLANNING:
            try:
                self.layers = self.runtime.prepare(self._future.result())
            except Exception as error:
                self._error = error
            self.phase = Phase.PREPARING
            return False
        if self.phase == Phase.PREPARING:
            if not self.layers:
                self.phase = Phase.IDLE
                return True
        else:
            try:
                self.runtime.install(layer)
            except Exception as error:
                self._fatal("EPLB install failed; layout integrity unknown", error)
            self.index += 1
            if self.index == len(self.layers):
                self.phase = Phase.IDLE
                return True
        self.phase = Phase.STAGING
        try:
            work = self.runtime.stage(self.layers[self.index])
            self._future = self._executor.submit(work)
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
                try:
                    self._future.result(timeout=self.timeout)
                except TimeoutError:
                    raise
                except Exception:
                    # Worker returned an error, so it is no longer running.
                    pass
            self.runtime.synchronize()
        except Exception as error:
            local_error = error
        try:
            drained = self.control.drained(local_error is None)
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

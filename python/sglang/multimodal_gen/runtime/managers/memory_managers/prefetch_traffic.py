"""Rank-local coordination between bounded prefetch and collective traffic."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class PrefetchTrafficCoordinator:
    """Pause and drain bounded prefetch submissions around collectives.

    Producers reserve a block before submitting it and publish a CUDA event
    immediately after recording the copy. A collective closes admission and
    synchronizes only the bounded set of admitted block events. This creates a
    hard no-H2D boundary without a device-wide synchronization.
    """

    def __post_init__(self) -> None:
        self._condition = threading.Condition()
        self._active_collectives = 0
        self._generation = 0
        self._latest_completion: tuple[int, torch.cuda.Event] | None = None
        self._users = 0
        self._next_block_token = 0
        self._outstanding_blocks: dict[int, torch.cuda.Event | None] = {}

    def register(self) -> None:
        with self._condition:
            self._users += 1

    def unregister(self) -> None:
        with self._condition:
            if self._users <= 0:
                raise RuntimeError("prefetch coordinator unregister imbalance")
            self._users -= 1

    @property
    def active(self) -> bool:
        with self._condition:
            return self._users > 0

    def begin_collective(self) -> int:
        """Close admission and retire every block admitted before this call."""
        with self._condition:
            self._active_collectives += 1
            self._generation += 1
            generation = self._generation
            while any(event is None for event in self._outstanding_blocks.values()):
                self._condition.wait()
            outstanding = [
                (token, event)
                for token, event in self._outstanding_blocks.items()
                if event is not None
            ]

        try:
            for _, event in outstanding:
                event.synchronize()
        except BaseException:
            with self._condition:
                self._active_collectives -= 1
                self._condition.notify_all()
            raise

        with self._condition:
            for token, _ in outstanding:
                self._outstanding_blocks.pop(token, None)
            self._condition.notify_all()
        return generation

    def end_collective(self, generation: int, event: torch.cuda.Event) -> None:
        with self._condition:
            if self._active_collectives <= 0:
                raise RuntimeError("collective region imbalance")
            if (
                self._latest_completion is None
                or generation > self._latest_completion[0]
            ):
                self._latest_completion = (generation, event)
            self._active_collectives -= 1
            self._condition.notify_all()

    def abort_collective(self) -> None:
        with self._condition:
            if self._active_collectives <= 0:
                raise RuntimeError("collective region imbalance")
            self._active_collectives -= 1
            self._condition.notify_all()

    def before_submit_block(
        self, copy_stream: torch.cuda.Stream, observed_generation: int
    ) -> tuple[int, int]:
        """Wait for admission, transfer the comm dependency, and reserve a block."""
        with self._condition:
            while self._active_collectives:
                self._condition.wait()
            pending = self._latest_completion
            if pending is not None and pending[0] > observed_generation:
                generation, event = pending
                copy_stream.wait_event(event)
                observed_generation = generation
            token = self._next_block_token
            self._next_block_token += 1
            self._outstanding_blocks[token] = None
        return observed_generation, token

    def publish_block(self, token: int, event: torch.cuda.Event) -> None:
        """Publish the recorded completion event for a reserved block."""
        with self._condition:
            if token not in self._outstanding_blocks:
                raise RuntimeError("unknown or retired prefetch block token")
            if self._outstanding_blocks[token] is not None:
                raise RuntimeError("prefetch block event published more than once")
            self._outstanding_blocks[token] = event
            self._condition.notify_all()

    def retire_block(self, token: int) -> None:
        """Forget a producer-retired block; idempotent with collective drain."""
        with self._condition:
            self._outstanding_blocks.pop(token, None)
            self._condition.notify_all()

    def cancel_block(self, token: int) -> None:
        """Cancel a reservation whose CUDA work was never submitted."""
        self.retire_block(token)


_COORDINATORS: dict[int, PrefetchTrafficCoordinator] = {}
_COORDINATORS_LOCK = threading.Lock()


def _device_index(device: torch.device | int) -> int:
    index = device if isinstance(device, int) else device.index
    return torch.cuda.current_device() if index is None else index


def get_prefetch_traffic_coordinator(
    device: torch.device | int,
) -> PrefetchTrafficCoordinator:
    index = _device_index(device)
    with _COORDINATORS_LOCK:
        coordinator = _COORDINATORS.get(index)
        if coordinator is None:
            coordinator = PrefetchTrafficCoordinator()
            _COORDINATORS[index] = coordinator
        return coordinator


def _get_active_prefetch_traffic_coordinator(
    device: torch.device | int,
) -> PrefetchTrafficCoordinator | None:
    # Coordinators are installed during model initialization, before inference
    # threads start. Avoid creating and locking a coordinator for the default,
    # feature-disabled path that executes on every Ulysses collective.
    coordinator = _COORDINATORS.get(_device_index(device))
    return coordinator if coordinator is not None and coordinator.active else None


@contextmanager
def collective_prefetch_guard(device: torch.device):
    """Guard a synchronous current-stream collective when prefetch is active."""
    coordinator = _get_active_prefetch_traffic_coordinator(device)
    if coordinator is None or torch.cuda.is_current_stream_capturing():
        yield
        return

    generation = coordinator.begin_collective()
    try:
        yield
    except BaseException:
        coordinator.abort_collective()
        raise
    else:
        completion = torch.cuda.Event()
        completion.record(torch.cuda.current_stream(device))
        coordinator.end_collective(generation, completion)

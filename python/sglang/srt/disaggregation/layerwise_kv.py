"""Layer-wise Hybrid Attention KV transfer orchestration.

The controller is deliberately independent of CUDA and NIXL bindings so its
job lifecycle and scheduler/worker races can be tested on CPU. The manager
passed to it owns the actual NIXL submission primitive.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sglang.srt.disaggregation.layer_progress import LayerProgress

logger = logging.getLogger(__name__)


class LayerKVSubmitter(Protocol):
    def submit_layerwise_kv(self, job: LayerwiseKVJob, layer_slots: list[int]): ...

    def submit_layerwise_state(
        self, job: LayerwiseStateJob, layer_slots: list[int]
    ): ...


def compact_layer_block_ids(
    layer_slots: list[int], num_layers: int, num_tensor_kinds: int
) -> list[int]:
    """Return tensor-major block ids for validated compact layer slots."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_tensor_kinds <= 0:
        raise ValueError(f"num_tensor_kinds must be positive, got {num_tensor_kinds}")
    slots = [int(slot) for slot in layer_slots]
    if len(slots) != len(set(slots)):
        raise ValueError(f"compact layer slots must be unique, got {slots}")
    invalid = [slot for slot in slots if slot < 0 or slot >= num_layers]
    if invalid:
        raise ValueError(
            f"compact layer slots out of range [0, {num_layers}): {invalid}"
        )
    return [
        kind * num_layers + slot for kind in range(num_tensor_kinds) for slot in slots
    ]


def select_prepped_layer_indices(
    indices: npt.NDArray[np.int32], block_ids: list[int], block_length: int
) -> npt.NDArray[np.int32]:
    """Map per-slot indices into selected blocks of a prepared NIXL dlist."""
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if not block_ids or indices.size == 0:
        return np.empty(0, dtype=np.int32)
    if any(block_id < 0 for block_id in block_ids):
        raise ValueError(f"block ids must be non-negative, got {block_ids}")
    offsets = np.asarray(block_ids, dtype=np.int64) * np.int64(block_length)
    selected = offsets[:, None] + np.asarray(indices, dtype=np.int64)[None, :]
    if selected.size and selected.max() > np.iinfo(np.int32).max:
        raise OverflowError("prepared NIXL descriptor index exceeds int32")
    return selected.ravel().astype(np.int32)


def select_compact_layer_entries(
    entry_layer_ids: list[int], compact_layer_ids: list[int], layer_slots: list[int]
) -> list[int]:
    """Select every tensor-major entry belonging to compact layer slots."""
    slots = [int(slot) for slot in layer_slots]
    if len(slots) != len(set(slots)):
        raise ValueError(f"compact layer slots must be unique, got {slots}")
    invalid = [slot for slot in slots if slot < 0 or slot >= len(compact_layer_ids)]
    if invalid:
        raise ValueError(
            f"compact layer slots out of range [0, {len(compact_layer_ids)}): {invalid}"
        )
    selected_ids = {compact_layer_ids[slot] for slot in slots}
    positions = [
        index
        for index, layer_id in enumerate(entry_layer_ids)
        if layer_id in selected_ids
    ]
    if not positions:
        raise ValueError("selected compact layers have no transfer entries")
    return positions


@dataclasses.dataclass
class LayerwiseKVJob:
    room: int
    chunk_id: int
    agent_name: str
    generation: int
    page_indices: npt.NDArray[np.int32]
    dst_page_indices: npt.NDArray[np.int32]
    sent: set[int] = dataclasses.field(default_factory=set)
    closed: bool = False
    handles: list[Any] = dataclasses.field(default_factory=list)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.room, self.chunk_id, self.agent_name)

    def matches(
        self,
        page_indices: npt.NDArray[np.int32],
        dst_page_indices: npt.NDArray[np.int32],
    ) -> bool:
        return np.array_equal(self.page_indices, page_indices) and np.array_equal(
            self.dst_page_indices, dst_page_indices
        )


@dataclasses.dataclass
class LayerwiseStateJob:
    room: int
    agent_name: str
    generation: int
    src_state_index: int
    dst_state_index: int
    sent: set[int] = dataclasses.field(default_factory=set)
    closed: bool = False
    handles: list[Any] = dataclasses.field(default_factory=list)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def matches(self, src_state_index: int, dst_state_index: int) -> bool:
        return (
            self.src_state_index == src_state_index
            and self.dst_state_index == dst_state_index
        )

    @property
    def key(self) -> tuple[int, str]:
        return (self.room, self.agent_name)


@dataclasses.dataclass(frozen=True)
class LayerwiseStateDrain:
    missing_slots: list[int] | None
    handles: list[Any]
    src_state_index: int | None = None
    dst_state_index: int | None = None


@dataclasses.dataclass(frozen=True)
class LayerwiseKVDrain:
    missing_slots: list[int] | None
    handles: list[Any]


@dataclasses.dataclass(frozen=True)
class LayerwiseKVReservation:
    chunk_id: int
    index_slice: slice


def reserve_layerwise_kv_chunk(
    sender: Any, start_idx: int, end_idx: int, num_pages: int
) -> LayerwiseKVReservation:
    """Reserve the sender cursor used by a forward that has not enqueued yet.

    Under overlap scheduling, forward N+1 is armed before forward N's result
    calls ``sender.send``. Keeping a separate reservation cursor prevents both
    forwards from claiming the same chunk id and destination slice.
    """
    reserved_chunk_id = getattr(sender, "_layerwise_reserved_chunk_id", sender.chunk_id)
    reserved_curr_idx = getattr(sender, "_layerwise_reserved_curr_idx", sender.curr_idx)
    reserved_start_idx = getattr(sender, "_layerwise_reserved_start_idx", start_idx)
    if reserved_start_idx != start_idx:
        raise ValueError(
            "layer-wise KV reservation start mismatch: "
            f"reserved={reserved_start_idx}, planned={start_idx}"
        )

    reservation = LayerwiseKVReservation(
        chunk_id=reserved_chunk_id,
        index_slice=slice(reserved_curr_idx, reserved_curr_idx + num_pages),
    )
    sender._layerwise_reserved_chunk_id = reserved_chunk_id + 1
    sender._layerwise_reserved_curr_idx = reserved_curr_idx + num_pages
    sender._layerwise_reserved_start_idx = end_idx
    return reservation


class LayerwiseKVController:
    """Poll one graph-replayed progress counter and stream ready compact KV slots."""

    def __init__(
        self,
        submitter: LayerKVSubmitter,
        progress: LayerProgress,
        layer_to_kv_slot: list[int],
        num_kv_layers: int,
        submit_batch: int,
        poll_interval_s: float = 0.001,
        layer_to_state_slot: list[int] | None = None,
        num_state_layers: int = 0,
        state_submit_batch: int = 1,
        start_thread: bool = True,
    ) -> None:
        if num_kv_layers <= 0:
            raise ValueError(f"num_kv_layers must be positive, got {num_kv_layers}")
        if submit_batch <= 0:
            raise ValueError(f"submit_batch must be positive, got {submit_batch}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be positive, got {poll_interval_s}")

        mapped_slots = sorted(slot for slot in layer_to_kv_slot if slot >= 0)
        if mapped_slots != list(range(num_kv_layers)):
            raise ValueError(
                "global-to-compact KV mapping must cover each compact slot exactly "
                f"once, got {mapped_slots}"
            )

        self.submitter = submitter
        self.progress = progress
        self.num_layers = len(layer_to_kv_slot)
        self.num_kv_layers = num_kv_layers
        self.submit_batch = submit_batch
        self.poll_interval_s = poll_interval_s

        complete_slots: list[frozenset[int]] = []
        slots: set[int] = set()
        for global_layer_id in range(self.num_layers + 1):
            complete_slots.append(frozenset(slots))
            if global_layer_id < self.num_layers:
                slot = layer_to_kv_slot[global_layer_id]
                if slot >= 0:
                    slots.add(slot)
        self._complete_slots = complete_slots

        self.num_state_layers = num_state_layers
        self.state_submit_batch = state_submit_batch
        if num_state_layers < 0 or state_submit_batch <= 0:
            raise ValueError("state layer count and submit batch must be valid")
        state_mapping = layer_to_state_slot or [-1] * self.num_layers
        if len(state_mapping) != self.num_layers:
            raise ValueError(
                "KV and state global layer mappings must have equal length"
            )
        mapped_state_slots = sorted(slot for slot in state_mapping if slot >= 0)
        if mapped_state_slots != list(range(num_state_layers)):
            raise ValueError(
                "global-to-compact state mapping must cover each compact slot "
                f"exactly once, got {mapped_state_slots}"
            )
        state_slots: set[int] = set()
        self._complete_state_slots: list[frozenset[int]] = []
        for global_layer_id in range(self.num_layers + 1):
            self._complete_state_slots.append(frozenset(state_slots))
            if global_layer_id < self.num_layers:
                slot = state_mapping[global_layer_id]
                if slot >= 0:
                    state_slots.add(slot)

        self._jobs: dict[tuple[int, int, str], LayerwiseKVJob] = {}
        self._retired: dict[tuple[int, int, str], LayerwiseKVJob] = {}
        self._state_jobs: dict[tuple[int, str], LayerwiseStateJob] = {}
        self._retired_state_jobs: dict[tuple[int, str], LayerwiseStateJob] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._poll_loop, daemon=True, name="pd-layerwise-kv"
            )
            self._thread.start()

    def arm(self, jobs: list[LayerwiseKVJob]) -> None:
        new_jobs = {job.key: job for job in jobs}
        if len(new_jobs) != len(jobs):
            raise ValueError("duplicate layer-wise KV job key")
        with self._lock:
            for key, job in self._jobs.items():
                if key not in new_jobs and not job.closed:
                    self._retired[key] = job
            self._jobs = new_jobs
        if jobs:
            self._wake.set()

    def drain(
        self,
        room: int,
        chunk_id: int,
        agent_name: str,
        page_indices: npt.NDArray[np.int32],
        dst_page_indices: npt.NDArray[np.int32],
    ) -> LayerwiseKVDrain:
        key = (room, chunk_id, agent_name)
        with self._lock:
            job = self._jobs.pop(key, None)
            if job is None:
                job = self._retired.pop(key, None)
        if job is None:
            return LayerwiseKVDrain(None, [])

        with job.lock:
            job.closed = True
            handles = list(job.handles)
            if not job.matches(page_indices, dst_page_indices):
                logger.warning(
                    "Layer-wise KV plan mismatch for room=%s chunk=%s peer=%s; "
                    "falling back to the full layer set",
                    room,
                    chunk_id,
                    agent_name,
                )
                return LayerwiseKVDrain(None, handles)
            missing = [
                slot for slot in range(self.num_kv_layers) if slot not in job.sent
            ]
            return LayerwiseKVDrain(missing, handles)

    def arm_state(self, jobs: list[LayerwiseStateJob]) -> None:
        new_jobs = {job.key: job for job in jobs}
        if len(new_jobs) != len(jobs):
            raise ValueError("duplicate layer-wise state job key")
        with self._lock:
            for key, job in self._state_jobs.items():
                if key not in new_jobs and not job.closed:
                    self._retired_state_jobs[key] = job
            self._state_jobs = new_jobs
        if jobs:
            self._wake.set()

    def drain_state(
        self,
        room: int,
        agent_name: str,
        src_state_index: int,
        dst_state_index: int,
    ) -> LayerwiseStateDrain:
        key = (room, agent_name)
        with self._lock:
            job = self._state_jobs.pop(key, None)
            if job is None:
                job = self._retired_state_jobs.pop(key, None)
        if job is None:
            return LayerwiseStateDrain(None, [])
        with job.lock:
            job.closed = True
            handles = list(job.handles)
            if not job.matches(src_state_index, dst_state_index):
                logger.warning(
                    "Layer-wise state plan mismatch for room=%s peer=%s; "
                    "falling back to the full state set",
                    room,
                    agent_name,
                )
                return LayerwiseStateDrain(None, handles)
            missing = [
                slot for slot in range(self.num_state_layers) if slot not in job.sent
            ]
            return LayerwiseStateDrain(
                missing, handles, job.src_state_index, job.dst_state_index
            )

    def close_room(self, room: int) -> list[Any]:
        """Close every job for a request and return all submitted handles."""
        with self._lock:
            jobs = []
            for job_map in (self._jobs, self._retired):
                for key in [key for key in job_map if key[0] == room]:
                    jobs.append(job_map.pop(key))
            state_jobs = []
            for job_map in (self._state_jobs, self._retired_state_jobs):
                for key in [key for key in job_map if key[0] == room]:
                    state_jobs.append(job_map.pop(key))
        handles = []
        for job in [*jobs, *state_jobs]:
            with job.lock:
                job.closed = True
                handles.extend(job.handles)
        return handles

    def drive_once(self, jobs: list[LayerwiseKVJob] | None = None) -> None:
        if jobs is None:
            with self._lock:
                jobs = list(self._jobs.values())
        for job in jobs:
            completed_layers = self.progress.completed_layers(job.generation)
            if completed_layers <= 0:
                continue
            at_end = completed_layers >= self.num_layers
            complete = self._complete_slots[min(completed_layers, self.num_layers)]
            with job.lock:
                if job.closed:
                    continue
                ready = sorted(complete - job.sent)
                if not ready or (len(ready) < self.submit_batch and not at_end):
                    continue
                handle = self.submitter.submit_layerwise_kv(job, ready)
                if handle is not None:
                    job.handles.append(handle)
                    job.sent.update(ready)

        with self._lock:
            state_jobs = list(self._state_jobs.values())
        for job in state_jobs:
            completed_layers = self.progress.completed_layers(job.generation)
            if completed_layers <= 0:
                continue
            at_end = completed_layers >= self.num_layers
            complete = self._complete_state_slots[
                min(completed_layers, self.num_layers)
            ]
            with job.lock:
                if job.closed:
                    continue
                ready = sorted(complete - job.sent)
                if not ready or (len(ready) < self.state_submit_batch and not at_end):
                    continue
                handle = self.submitter.submit_layerwise_state(job, ready)
                if handle is not None:
                    job.handles.append(handle)
                    job.sent.update(ready)

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _poll_loop(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=0.05)
            with self._lock:
                jobs = list(self._jobs.values())
                has_state_jobs = bool(self._state_jobs)
            if not jobs and not has_state_jobs:
                self._wake.clear()
                continue
            try:
                self.drive_once(jobs)
            except Exception:
                logger.exception(
                    "Layer-wise KV poller failed; worker tail will resend missing layers"
                )
            time.sleep(self.poll_interval_s)

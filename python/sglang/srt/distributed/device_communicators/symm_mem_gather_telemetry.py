"""Opt-in host telemetry for the symmetric-memory DP metadata gather.

This module intentionally has no torch dependency.  The gather hot path passes
it data that is already resident in pinned host memory after the gather's
existing stream synchronization.  Records stay in memory until an explicit
profile stop, so profiling does not add I/O to the gather window.
"""

from __future__ import annotations

import json
import os
import weakref
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_SCHEMA_VERSION = 1
_REGISTERED_GATHERERS: weakref.WeakSet[Any] = weakref.WeakSet()


def _ready_mask(ready: Sequence[Any]) -> int:
    mask = 0
    for peer, value in enumerate(ready):
        if int(value):
            mask |= 1 << peer
    return mask


class SymmMemGatherTelemetry:
    """A bounded, host-only recorder owned by one ``SymmMemGather``."""

    def __init__(self, *, world_size: int, group_rank: int, max_records: int):
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= group_rank < world_size:
            raise ValueError("group_rank must be in [0, world_size)")
        if max_records <= 0:
            raise ValueError("max_records must be positive")

        self.world_size = world_size
        self.group_rank = group_rank
        self.max_records = max_records
        self._records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._active = False
        self._output_dir: Optional[Path] = None
        self._profile_id: Optional[str] = None
        self._dp_rank: Optional[int] = None
        self._generation_regressions = 0
        self._last_generation: Optional[int] = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self, *, output_dir: str, profile_id: str, dp_rank: int) -> None:
        if self._active:
            raise RuntimeError("symmetric-memory DP telemetry is already active")
        self._records.clear()
        self._output_dir = Path(output_dir).expanduser()
        self._profile_id = profile_id
        self._dp_rank = dp_rank
        self._generation_regressions = 0
        self._last_generation = None
        self._active = True

    def begin(
        self,
        *,
        generation: int,
        slot: int,
        gather_start_ns: int,
        local_row: Sequence[Any],
    ) -> Optional[dict[str, Any]]:
        if not self._active:
            return None

        if self._last_generation is not None:
            expected = self._last_generation % 0xFFFFFFFF + 1
            if generation != expected:
                self._generation_regressions += 1
        self._last_generation = generation
        return {
            "generation": generation,
            "slot": slot,
            "gather_start_ns": gather_start_ns,
            "local_row": [int(value) for value in local_row],
            "poll_count": 0,
            "ready_mask_rle": [],
            "first_ready_poll": [-1] * self.world_size,
            "first_ready_ns": [-1] * self.world_size,
            "d2h_sync_wall_ns": 0,
            "d2h_sync_wall_max_ns": 0,
            "host_retry_gap_ns": 0,
            "host_retry_gap_max_ns": 0,
            "_last_sync_done_ns": None,
        }

    def note_poll(
        self,
        record: dict[str, Any],
        *,
        ready: Sequence[Any],
        poll_begin_ns: int,
        sync_begin_ns: int,
        sync_done_ns: int,
    ) -> int:
        poll_index = record["poll_count"]
        record["poll_count"] = poll_index + 1

        sync_wall_ns = sync_done_ns - sync_begin_ns
        record["d2h_sync_wall_ns"] += sync_wall_ns
        record["d2h_sync_wall_max_ns"] = max(
            record["d2h_sync_wall_max_ns"], sync_wall_ns
        )

        last_sync_done_ns = record["_last_sync_done_ns"]
        if last_sync_done_ns is not None:
            retry_gap_ns = poll_begin_ns - last_sync_done_ns
            record["host_retry_gap_ns"] += retry_gap_ns
            record["host_retry_gap_max_ns"] = max(
                record["host_retry_gap_max_ns"], retry_gap_ns
            )
        record["_last_sync_done_ns"] = sync_done_ns

        mask = _ready_mask(ready)
        transitions = record["ready_mask_rle"]
        elapsed_ns = sync_done_ns - record["gather_start_ns"]
        if transitions and transitions[-1]["mask"] == mask:
            transitions[-1]["polls"] += 1
            transitions[-1]["last_poll"] = poll_index
            transitions[-1]["last_ns"] = elapsed_ns
        else:
            transitions.append(
                {
                    "mask": mask,
                    "polls": 1,
                    "first_poll": poll_index,
                    "last_poll": poll_index,
                    "first_ns": elapsed_ns,
                    "last_ns": elapsed_ns,
                }
            )

        first_ready_poll = record["first_ready_poll"]
        first_ready_ns = record["first_ready_ns"]
        for peer in range(self.world_size):
            if mask & (1 << peer) and first_ready_poll[peer] < 0:
                first_ready_poll[peer] = poll_index
                first_ready_ns[peer] = elapsed_ns
        return mask

    def finish(
        self,
        record: dict[str, Any],
        *,
        gather_done_ns: int,
        peer_rows: Iterable[Sequence[Any]],
    ) -> None:
        if not self._active:
            return
        record.pop("_last_sync_done_ns")
        record["gather_wall_ns"] = gather_done_ns - record["gather_start_ns"]
        record["peer_rows"] = [[int(value) for value in row] for row in peer_rows]
        max_ready_ns = max(record["first_ready_ns"], default=-1)
        record["slowest_peers"] = [
            peer
            for peer, ready_ns in enumerate(record["first_ready_ns"])
            if ready_ns == max_ready_ns
        ]
        self._records.append(record)

    def stop(self) -> Optional[Path]:
        if not self._active:
            return None
        self._active = False
        assert self._output_dir is not None
        assert self._profile_id is not None
        assert self._dp_rank is not None

        payload = {
            "schema_version": _SCHEMA_VERSION,
            "profile_id": self._profile_id,
            "pid": os.getpid(),
            "dp_rank": self._dp_rank,
            "group_rank": self.group_rank,
            "world_size": self.world_size,
            "max_records": self.max_records,
            "generation_regressions": self._generation_regressions,
            "records": list(self._records),
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"symm-dp-{self._profile_id}-DP-{self._dp_rank}"
            f"-GR-{self.group_rank}-PID-{os.getpid()}.json"
        )
        output_path = self._output_dir / filename
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, separators=(",", ":"))
            output.write("\n")
        temporary_path.replace(output_path)
        return output_path


def register_symm_mem_gather_telemetry(recorder: SymmMemGatherTelemetry) -> None:
    _REGISTERED_GATHERERS.add(recorder)


def start_symm_mem_gather_telemetry(
    *, output_dir: str, profile_id: str, dp_rank: int
) -> int:
    recorders = list(_REGISTERED_GATHERERS)
    for recorder in recorders:
        recorder.start(
            output_dir=output_dir,
            profile_id=profile_id,
            dp_rank=dp_rank,
        )
    return len(recorders)


def stop_symm_mem_gather_telemetry() -> list[Path]:
    paths = []
    for recorder in list(_REGISTERED_GATHERERS):
        path = recorder.stop()
        if path is not None:
            paths.append(path)
    return paths


def common_generation_ids(payloads: Sequence[dict[str, Any]]) -> list[int]:
    """Return generation IDs present in every payload, for offline alignment."""
    if not payloads:
        return []
    generation_sets = [
        {int(record["generation"]) for record in payload["records"]}
        for payload in payloads
    ]
    return sorted(set.intersection(*generation_sets))

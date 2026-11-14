"""Opt-in coordinator-owned native result retirement evidence."""

from __future__ import annotations

import hashlib
import json
from os import PathLike


class ExecutionEventRecorder:
    """Stream real result retirements; only a FINISHED grant seals the prefix."""

    def __init__(self, path: str | PathLike[str]):
        self._stream = open(path, "xb", buffering=0)  # noqa: SIM115
        self._prefix = hashlib.sha256()
        self._bytes = 0
        self._retired = 0
        self._last_ns = 0
        self._finished = None
        self._write(
            {
                "event": "start",
                "schema_version": 1,
                "clock": "virtual_ns",
                "physical_timing_observation": False,
            }
        )

    def _write(self, event):
        if self._stream.closed:
            raise RuntimeError("execution event recorder is closed")
        raw = (
            json.dumps(event, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode()
        if len(raw) > 4 * 1024**2:
            raise ValueError("execution event exceeds the 4 MiB record bound")
        try:
            if self._stream.write(raw) != len(raw):
                raise OSError("short execution event write")
        except BaseException:
            self.close()
            raise
        self._prefix.update(raw)
        self._bytes += len(raw)

    def _check_time(self, at_ns):
        if type(at_ns) is not int or at_ns < self._last_ns:
            raise ValueError("execution retirement time must be monotonic integer ns")

    def record(self, at_ns, request_ids):
        self._check_time(at_ns)
        if self._stream.closed:
            raise RuntimeError("execution event recorder is closed")
        ids = tuple(request_ids)
        if any(type(rid) is not str or not rid for rid in ids) or len(ids) != len(
            set(ids)
        ):
            raise ValueError("execution result requires unique nonempty request IDs")
        if ids:
            self._write(
                {
                    "event": "retired_worker",
                    "end_ns": at_ns,
                    "requests": [{"rid": rid} for rid in ids],
                }
            )
            self._retired += 1
        self._last_ns = at_ns

    def finish(self, run_id, at_ns):
        self._check_time(at_ns)
        if type(run_id) is not str or not run_id:
            raise ValueError("execution footer requires a run ID")
        if self._finished == (run_id, at_ns):
            return
        self._write(
            {
                "event": "finish",
                "status": "FINISHED",
                "run_id": run_id,
                "at_ns": at_ns,
                "retired_forwards": self._retired,
                "prefix_bytes": self._bytes,
                "prefix_sha256": self._prefix.hexdigest(),
            }
        )
        self._last_ns = at_ns
        self._finished = (run_id, at_ns)
        self.close()

    def close(self):
        """Interrupted runs close without a success footer."""
        self._stream.close()

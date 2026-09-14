#!/usr/bin/env python3
"""Incrementally index the append-only PD Flip request SLO ledger."""

import json
from pathlib import Path


class IncrementalLedgerTail:
    """Read each ledger byte once and retain only the latest row per request."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.latest_by_request = {}

    def _reset(self):
        self.offset = 0
        self.latest_by_request = {}

    def refresh(self):
        if not self.path.exists():
            return
        stat = self.path.stat()
        identity = (getattr(stat, "st_dev", None), getattr(stat, "st_ino", None))
        if self.identity is None:
            self.identity = identity
        elif identity != self.identity or stat.st_size < self.offset:
            self.identity = identity
            self._reset()

        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                line_start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    # The writer may still be appending this line.  Re-read it
                    # from the same byte offset on the next refresh.
                    self.offset = line_start
                    break
                self.offset = handle.tell()
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    continue
                if not isinstance(row, dict):
                    continue
                request_id = row.get("request_id")
                event_time = row.get("event_time")
                if request_id is None or not isinstance(event_time, (int, float)):
                    continue
                key = str(request_id)
                previous = self.latest_by_request.get(key)
                if previous is None or event_time >= previous.get("event_time", -1):
                    self.latest_by_request[key] = row

    def latest(self, now=None, window_seconds=0.0, window_start_time=None):
        self.refresh()
        cutoffs = []
        if now is not None and window_seconds > 0:
            cutoffs.append(float(now) - float(window_seconds))
        if window_start_time is not None:
            cutoffs.append(float(window_start_time))
        cutoff = max(cutoffs) if cutoffs else None
        rows = [
            row
            for row in self.latest_by_request.values()
            if cutoff is None or float(row.get("event_time") or 0.0) >= cutoff
        ]
        return sorted(
            rows,
            key=lambda row: (
                float(row.get("event_time") or 0.0),
                str(row.get("request_id")),
            ),
        )

    def terminal_count(self):
        self.refresh()
        return sum(
            1
            for row in self.latest_by_request.values()
            if row.get("status") not in (None, "running")
        )

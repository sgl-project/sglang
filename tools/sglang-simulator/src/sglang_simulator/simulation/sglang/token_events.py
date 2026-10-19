"""Stream exact native output-token commits without retaining token payloads."""

from __future__ import annotations

import json
from collections.abc import Iterable
from os import PathLike


class TokenEventRecorder:
    """Write an opt-in JSONL ledger, finalized only after simulator success.

    ``output_index`` is the zero-based position of the first token in a commit;
    ``token_count`` is its positive increment, not a cumulative output length.
    One counter per request enforces continuity even for requests later cancelled.
    The caller owns lifecycle/phase selection; this ledger records all commits.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        # The simulator owns this stream across batches and closes it on exit.
        self._stream = open(path, "x", encoding="utf-8", newline="\n")  # noqa: SIM115
        self._last_ns = 0
        self._counts: dict[str, int] = {}
        self._total_output_tokens = 0
        self._finished: tuple[str, int] | None = None
        self._write_event({"event": "start", "version": 1, "clock": "virtual_ns"})

    @staticmethod
    def _nonnegative_integer(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    def _check_open(self) -> None:
        if self._stream.closed:
            raise RuntimeError("token event recorder is finished or closed")

    def _check_time(self, at_ns: int) -> None:
        self._nonnegative_integer(at_ns, "at_ns")
        if at_ns < self._last_ns:
            raise ValueError("token event time must be monotonic")

    def _write_event(self, event: dict) -> None:
        # Serialize before touching the file; an I/O failure cannot be followed
        # by a success footer, even if the caller catches the original error.
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        try:
            self._stream.write(line)
            self._stream.flush()
        except BaseException:
            self.close()
            raise

    def record(self, at_ns: int, commits: Iterable[tuple[str, int, int]]) -> None:
        """Record one batch of ``(rid, output_index, token_count)`` triples.

        Validate the entire iterable before writing or updating counters. Empty
        batches advance the checked clock but do not produce output events.
        """
        self._check_open()
        self._check_time(at_ns)
        pending: dict[str, int] = {}
        entries = []
        added = 0
        for rid, output_index, token_count in commits:
            if not isinstance(rid, str) or not rid:
                raise ValueError("rid must be a nonempty string")
            self._nonnegative_integer(output_index, "output_index")
            self._nonnegative_integer(token_count, "token_count")
            if token_count == 0:
                raise ValueError("token_count must be positive")
            expected = pending.get(rid, self._counts.get(rid, 0))
            if output_index != expected:
                raise ValueError(
                    f"non-contiguous token commit for {rid}: "
                    f"expected output_index {expected}, got {output_index}"
                )
            pending[rid] = expected + token_count
            added += token_count
            entries.append(
                {"rid": rid, "output_index": output_index, "token_count": token_count}
            )
        if entries:
            self._write_event(
                {"event": "output_commit", "at_ns": at_ns, "commits": entries}
            )
        self._counts.update(pending)
        self._total_output_tokens += added
        self._last_ns = at_ns

    def finish(self, run_id: str, at_ns: int) -> None:
        """Write the success footer and close; an identical repeat is harmless."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        self._check_time(at_ns)
        if self._finished == (run_id, at_ns):
            return
        self._check_open()
        self._write_event(
            {
                "event": "finish",
                "run_id": run_id,
                "at_ns": at_ns,
                "status": "FINISHED",
                "total_output_tokens": self._total_output_tokens,
            }
        )
        self._last_ns = at_ns
        self._finished = (run_id, at_ns)
        self.close()

    def close(self) -> None:
        """Close an interrupted or completed ledger without adding any events."""
        self._stream.close()

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sglang.srt.debug_utils.comparator.output_types import BaseLog


class LogSink:
    def __init__(self) -> None:
        self._stack: list[list[BaseLog]] = []

    @contextmanager
    def context(self) -> Generator[list[BaseLog], None, None]:
        bucket: list[BaseLog] = []
        self._stack.append(bucket)
        try:
            yield bucket
        finally:
            popped = self._stack.pop()
            assert popped is bucket

    def add(self, log: BaseLog) -> None:
        if self._stack:
            self._stack[-1].append(log)
        else:
            from sglang.srt.debug_utils.comparator.output_types import (
                LogRecord,
                _split_logs,
                report_sink,
            )

            errors, infos = _split_logs([log])
            report_sink.add(LogRecord(errors=errors, infos=infos))


log_sink = LogSink()

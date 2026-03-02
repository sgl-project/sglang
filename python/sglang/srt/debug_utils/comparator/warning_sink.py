from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sglang.srt.debug_utils.comparator.output_types import AnyWarning


class WarningSink:
    def __init__(self) -> None:
        self._stack: list[list[AnyWarning]] = []

    @contextmanager
    def context(self) -> Generator[list[AnyWarning], None, None]:
        bucket: list[AnyWarning] = []
        self._stack.append(bucket)
        try:
            yield bucket
        finally:
            popped = self._stack.pop()
            assert popped is bucket

    def add(self, warning: AnyWarning) -> None:
        if self._stack:
            self._stack[-1].append(warning)
        else:
            from sglang.srt.debug_utils.comparator.output_types import (
                WarningRecord,
                report_sink,
            )

            report_sink.add(WarningRecord(warnings=[warning]))


warning_sink = WarningSink()

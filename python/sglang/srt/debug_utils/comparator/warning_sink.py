from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sglang.srt.debug_utils.comparator.output_types import AnyWarning


class WarningSink:
    def __init__(self) -> None:
        self._stack: list[list[AnyWarning]] = []
        self._output_format: str = "text"

    def set_output_format(self, output_format: str) -> None:
        self._output_format = output_format

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
                print_record,
            )

            print_record(
                WarningRecord(warnings=[warning]),
                output_format=self._output_format,
            )


warning_sink = WarningSink()

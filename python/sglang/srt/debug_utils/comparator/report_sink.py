from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Literal, Optional

from rich.console import Console

from sglang.srt.debug_utils.comparator.output_types import _OutputRecord

Verbosity = Literal["minimal", "normal", "verbose"]


class ReportSink:
    """Unified entry point for all record output."""

    def __init__(self) -> None:
        self._output_format: str = "text"
        self._verbosity: Verbosity = "normal"
        self._report_file: Optional[IO[str]] = None
        self._report_path: Optional[Path] = None
        self._console: Optional[Console] = None

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    def configure(
        self,
        *,
        output_format: str = "text",
        report_path: Optional[Path] = None,
        verbosity: Verbosity = "normal",
    ) -> None:
        self._output_format = output_format
        self._verbosity = verbosity

        if report_path is not None:
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                self._report_file = open(report_path, "w", encoding="utf-8")
                self._report_path = report_path
            except OSError as exc:
                print(
                    f"Warning: cannot open report file {report_path}: {exc}",
                    file=sys.stderr,
                )

    def add(self, record: _OutputRecord) -> None:
        self._print_to_stdout(record)

        if self._report_file is not None:
            self._report_file.write(record.model_dump_json())
            self._report_file.write("\n")
            self._report_file.flush()

    def close(self) -> None:
        if self._report_file is not None:
            self._report_file.close()
            self._report_file = None

    @property
    def report_path(self) -> Optional[Path]:
        return self._report_path

    def _reset(self) -> None:
        self.close()
        self._output_format = "text"
        self._verbosity = "normal"
        self._report_path = None
        self._console = None

    def _get_console(self) -> Console:
        if self._console is None:
            self._console = Console()
        return self._console

    def _print_to_stdout(self, record: _OutputRecord) -> None:
        if self._output_format == "json":
            print(record.model_dump_json())
        else:
            console: Console = self._get_console()
            console.print(record.to_rich())
            console.print()  # blank line between records


report_sink = ReportSink()

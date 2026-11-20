"""Where planned cases and their results go: the printed table, the JSONL records, ..."""

import math
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .case import BenchInfo, Case, CaseResult
from .config import DISABLE_LOG_BANDWIDTH, DISABLE_LOG_FLOPS, OUTPUT_DIR, RANK
from .result import BenchResult

UNIT_SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


class Reporter(Protocol):
    def begin(self, *, info: BenchInfo) -> None: ...

    def before_case(self, case: Case) -> None: ...

    def on_result(self, result: CaseResult) -> None: ...

    def finish(self) -> None: ...


def make_reporters(
    *, print_prefix: Optional[str], print_suffix: Optional[str]
) -> List[Reporter]:
    reporters: List[Reporter] = [
        TableReporter(print_prefix=print_prefix, print_suffix=print_suffix)
    ]
    # A multi-GPU case is recorded once, with rank 0's measurement.
    if OUTPUT_DIR is not None and RANK == 0:
        # Lazy: pynvml is not installed on every platform (e.g. ROCm images).
        from .record import RecordReporter

        reporters.append(RecordReporter(output_dir=OUTPUT_DIR))
    return reporters


class _Table:
    SEP = " | "

    def __init__(self) -> None:
        self._headers: List[str] = []
        self._mins: List[int] = []
        self._pads: List[int] = []
        self._aligns: List[str] = []
        self._seps: set = set()
        self._rows: List[List[str]] = []

    @staticmethod
    def format_latency(r: float) -> str:
        if math.isnan(r):
            return "N/A"
        return f"{r:.4f}"

    @staticmethod
    def format_bandwidth(b: float) -> str:
        if math.isnan(b):
            return "N/A"
        return f"{b:.2f}"

    @staticmethod
    def format_flops(f: float) -> str:
        if math.isnan(f):
            return "N/A"
        return f"{f:.3f}"

    def col(
        self,
        header: str = "",
        *,
        min_width: int = 10,
        pad: int = 2,
        align: str = ">",
    ) -> None:
        self._headers.append(header)
        self._mins.append(min_width)
        self._pads.append(pad)
        self._aligns.append(align)

    def sep(self) -> None:
        self._seps.add(len(self._headers))

    def row(self, *cells: Any) -> None:
        assert len(cells) == len(self._headers)
        self._rows.append([str(c) for c in cells])

    def print(self, prefix: Optional[str], suffix: Optional[str]) -> None:
        widths = [
            max(max(len(c) + p for c in [h, *(r[i] for r in self._rows)]), mw)
            for i, (h, mw, p) in enumerate(zip(self._headers, self._mins, self._pads))
        ]
        total = sum(widths) + len(self.SEP) * len(self._seps)

        def fmt(cells: List[str]) -> str:
            parts: List[str] = []
            for i, (cell, w, a) in enumerate(zip(cells, widths, self._aligns)):
                if i in self._seps:
                    parts.append(self.SEP)
                parts.append(f"{cell:{a}{w}}")
            return "".join(parts)

        if prefix is not None:
            print("=" * total)
            print(prefix)
        print("=" * total)
        print(fmt(self._headers))
        print("-" * total)
        for r in self._rows:
            print(fmt(r))
        print("=" * total)
        if suffix is not None:
            print(suffix)
            print("=" * total)


class TableReporter:
    """One row per parameter row, one column per line; rows every line skipped are hidden."""

    def __init__(
        self, *, print_prefix: Optional[str], print_suffix: Optional[str]
    ) -> None:
        self._print_prefix = print_prefix
        self._print_suffix = print_suffix

    def begin(self, *, info: BenchInfo) -> None:
        self._info = info
        self._results: Dict[Tuple[int, str], CaseResult] = {}

    def before_case(self, case: Case) -> None:
        pass

    def on_result(self, result: CaseResult) -> None:
        self._results[(result.case.row, result.case.line)] = result

    def finish(self) -> None:
        info = self._info
        lines = [str(v) for v in info.line_vals]
        done = [r.result for r in self._results.values() if r.result is not None]
        log_bw = not DISABLE_LOG_BANDWIDTH and any(r.gbps() is not None for r in done)
        log_flops = not DISABLE_LOG_FLOPS and any(r.tflops() is not None for r in done)

        table = _Table()
        table.col(min_width=0, pad=0, align="<")  # id column (tight, left-aligned)
        for name in info.axis_names:
            table.col(name)
        scale = UNIT_SCALE[info.unit]
        # (column unit, cell formatter, value of one result)
        sections: List[
            Tuple[str, Callable[[float], str], Callable[[BenchResult], Optional[float]]]
        ] = [(info.unit, table.format_latency, lambda r: r.times[0] / scale)]
        if log_bw:
            sections.append(("GB/s", table.format_bandwidth, BenchResult.gbps))
        if log_flops:
            sections.append(("TFLOPS", table.format_flops, BenchResult.tflops))
        for unit, _, _ in sections:
            table.sep()
            for line in lines:
                table.col(f"{line}({unit})", min_width=15)

        kwargs_by_row = {row: r.case.kwargs for (row, _), r in self._results.items()}
        for row in sorted(kwargs_by_row):
            got = [self._results.get((row, line)) for line in lines]
            results = [None if g is None else g.result for g in got]
            if all(r is None for r in results):
                continue
            cells: List[Any] = [row]
            cells.extend(kwargs_by_row[row][name] for name in info.axis_names)
            for _, fmt, value in sections:
                cells.extend(
                    fmt(_or_nan(None if r is None else value(r))) for r in results
                )
            table.row(*cells)

        table.print(self._print_prefix, self._print_suffix)


def _or_nan(value: Optional[float]) -> float:
    return math.nan if value is None else value

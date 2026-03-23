"""Shared test helpers for comparator tests."""

from __future__ import annotations

import re
from io import StringIO
from typing import Optional

from rich.console import Console

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0,
    suite="stage-a-cpu-only",
    nightly=True,
    disabled="helper module, no tests",
)

from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorInfo,
    TensorStats,
)

DEFAULT_PERCENTILES: dict[int, float] = {
    1: -1.8,
    5: -1.5,
    50: 0.0,
    95: 1.5,
    99: 1.8,
}

DEFAULT_ABS_DIFF_PERCENTILES: dict[int, float] = {
    1: 0.0001,
    5: 0.0001,
    50: 0.0002,
    95: 0.0004,
    99: 0.0005,
}


def make_stats(
    mean: float = 0.0,
    abs_mean: float = 0.8,
    std: float = 1.0,
    min: float = -2.0,
    max: float = 2.0,
    percentiles: Optional[dict[int, float]] = None,
) -> TensorStats:
    return TensorStats(
        mean=mean,
        abs_mean=abs_mean,
        std=std,
        min=min,
        max=max,
        percentiles=percentiles if percentiles is not None else DEFAULT_PERCENTILES,
    )


def make_diff(
    rel_diff: float = 0.0001,
    max_abs_diff: float = 0.0005,
    mean_abs_diff: float = 0.0002,
    abs_diff_percentiles: Optional[dict[int, float]] = None,
    diff_threshold: float = 1e-3,
    passed: bool = True,
) -> DiffInfo:
    return DiffInfo(
        rel_diff=rel_diff,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        abs_diff_percentiles=(
            abs_diff_percentiles
            if abs_diff_percentiles is not None
            else DEFAULT_ABS_DIFF_PERCENTILES
        ),
        max_diff_coord=[2, 3],
        baseline_at_max=1.0,
        target_at_max=1.0005,
        diff_threshold=diff_threshold,
        passed=passed,
    )


_ANSI_ESCAPE_RE = re.compile(r"\033\[([0-9;]*)m")


def assert_rich_tags_balanced(markup: str) -> None:
    """Render Rich markup to ANSI and verify no styles are active at the end.

    Tracks ANSI style state through the output. A ``\\033[0m`` (reset)
    clears all active styles; any other ``\\033[Nm`` sets a style.
    At the end of the output, no style should remain active.
    """
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=10000, highlight=False)
    console.print(markup, end="")
    ansi_output: str = buf.getvalue()

    if "\033[" not in ansi_output:
        return

    styled = False
    for match in _ANSI_ESCAPE_RE.finditer(ansi_output):
        params: str = match.group(1)
        if params == "0" or params == "":
            styled = False
        else:
            styled = True

    assert not styled, (
        f"ANSI styles still active at end of output — likely unclosed Rich tag.\n"
        f"Last 200 chars of ANSI output: {ansi_output[-200:]!r}"
    )


def make_tensor_info(
    shape: Optional[list[int]] = None,
    dtype: str = "torch.float32",
    stats: Optional[TensorStats] = None,
    sample: Optional[str] = None,
) -> TensorInfo:
    return TensorInfo(
        shape=shape if shape is not None else [4, 8],
        dtype=dtype,
        stats=stats if stats is not None else make_stats(),
        sample=sample,
    )

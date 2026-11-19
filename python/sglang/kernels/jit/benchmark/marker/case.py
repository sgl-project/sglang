"""What a benchmark runs: one `Case` per (parameter row, line), planned before any run."""

import enum
import hashlib
import json
import os
import pathlib
from typing import Any, Dict, FrozenSet, Iterable, NamedTuple, Optional, Tuple

import msgspec
import torch

from .result import BenchResult


class Tolerance(msgspec.Struct, frozen=True, kw_only=True):
    """Allowed slowdown of a case: max(rel * baseline, abs_us)."""

    rel: float = 0.05
    abs_us: float = 0.5


class BenchInfo(NamedTuple):
    bench_id: str
    source_file: str
    axis_names: Tuple[str, ...]
    line_vals: Tuple[Any, ...]
    unit: str
    gated_lines: FrozenSet[str]
    gate_reason: str
    tolerance: Tolerance


class Case(NamedTuple):
    bench_id: str
    params: Dict[str, Any]  # canonical values; they feed case_id and the records
    line: str
    world_size: int  # ranks the case runs on; part of its identity
    case_id: str
    row: int  # index of the parameter row in the planned grid
    kwargs: Dict[str, Any]  # raw values the benchmark fn is called with


class CaseResult(NamedTuple):
    case: Case
    result: Optional[BenchResult]  # None when the case raised BenchSkip
    skip_reason: Optional[str]


def canonical_value(value: Any) -> Any:
    """JSON form of a parametrize / line value; it feeds case_id, so it must be
    stable across processes (no object addresses)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [canonical_value(v) for v in value]
    if isinstance(value, (torch.dtype, torch.device, enum.Enum)):
        return str(value)
    raise TypeError(
        f"benchmark parameter {value!r} has type {type(value).__name__}; use "
        "bool/int/float/str/None, torch.dtype/device, an Enum, or lists/tuples of them"
    )


def check_canonical(values: Iterable[Any]) -> None:
    """Raise TypeError unless every value has a `canonical_value` form."""
    for value in values:
        canonical_value(value)


def make_case_id(
    *,
    bench_id: str,
    params: Dict[str, Any],
    line: str,
    world_size: int,
) -> str:
    key = json.dumps(
        [bench_id, params, line, world_size],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def make_bench_id(*, source_file: str, fn_name: str) -> str:
    path = pathlib.Path(os.path.abspath(source_file))
    for parent in path.parents:
        if (parent / "python" / "sglang").is_dir():
            return f"{path.relative_to(parent)}::{fn_name}"
    return f"{path}::{fn_name}"

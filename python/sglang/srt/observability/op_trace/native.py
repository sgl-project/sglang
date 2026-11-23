"""Records of the native CUPTI collector (`sglang.kernels.jit.debug.cupti_collector`)
-> the kineto-shaped trace JSON the analyzer reads."""

from __future__ import annotations

import gzip
import json
import os
from typing import Optional

import numpy as np

from sglang.kernels.jit.debug.cupti_collector import RECORD_FIELDS

_COL = {name: i for i, name in enumerate(RECORD_FIELDS)}
_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def write_kineto_like_trace(
    path: str,
    records: np.ndarray,
    names: list[str],
    graph_execs: Optional[list[tuple[int, int]]],
) -> None:
    """Device events in the kineto JSON shape the analyzer reads, plus the
    graph execs in creation order (``opTraceExecs``) and this process's pid."""
    rows = records.tolist()
    # Relative to the first record, so float microseconds keep ns precision.
    base_ns = min((r[_COL["start"]] for r in rows), default=0)
    events = [_event(row=r, names=names, base_ns=base_ns) for r in rows]
    payload = {
        "traceEvents": events,
        "baseTimeNanoseconds": base_ns,
        "opTracePid": os.getpid(),
        "opTraceExecs": [list(e) for e in graph_execs or []],
    }
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


def _event(*, row: list[int], names: list[str], base_ns: int) -> dict:
    c = _COL
    return {
        "ph": "X",
        "cat": _CATEGORIES[row[c["kind"]]],
        "name": names[row[c["name_id"]]],
        "ts": (row[c["start"]] - base_ns) / 1e3,
        "dur": (row[c["end"]] - row[c["start"]]) / 1e3,
        "args": {
            "graph id": row[c["graph_id"]],
            "graph node id": row[c["graph_node_id"]],
            "stream": row[c["stream_id"]],
            "correlation": row[c["correlation_id"]],
            "grid": [row[c["grid_x"]], row[c["grid_y"]], row[c["grid_z"]]],
        },
    }

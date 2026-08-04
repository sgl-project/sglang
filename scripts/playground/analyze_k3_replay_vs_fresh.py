#!/usr/bin/env python3
"""Compare one CUDA-graph verify row with the equivalent fresh-prefill row."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


GRAPH_RE = re.compile(
    r"K3_GRAPH_ROW_TRACE rank=(?P<rank>\d+) replay_id=(?P<replay>\d+).*?"
    r"layer_id=(?P<layer>\d+) stage=(?P<stage>\w+).*?"
    r"row_abs=(?P<abs>\[[^\n]*?\]) row_sum=(?P<sum>\[[^\n]*?\]) "
    r"row_sq=(?P<sq>\[[^\n]*?\])"
)
FRESH_RE = re.compile(
    r"K3_HIDDEN_ROW_TRACE layer_id=(?P<layer>\d+) stage=(?P<stage>\w+) "
    r"mode=EXTEND.*?shape=\((?P<rows>\d+),.*?"
    r"row_abs=(?P<abs>\[[^\n]*?\]) row_sum=(?P<sum>\[[^\n]*?\]) "
    r"row_sq=(?P<sq>\[[^\n]*?\])"
)


def parse_values(match: re.Match[str], index: int, scale: float = 1.0):
    values = {}
    for name in ("abs", "sum", "sq"):
        seq = ast.literal_eval(match.group(name))
        values[name] = float(seq[index]) / scale
    return values


def parse_graph(path: Path, rank: int, replay: int, row: int):
    records = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = GRAPH_RE.search(line)
        if not match:
            continue
        if int(match.group("rank")) != rank or int(match.group("replay")) != replay:
            continue
        key = (int(match.group("layer")), match.group("stage"))
        records[key] = parse_values(match, row)
    return records


def parse_fresh(path: Path, token_row: int, tp_scale: float):
    records = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FRESH_RE.search(line)
        if not match:
            continue
        rows = int(match.group("rows"))
        index = token_row if rows > token_row else 0
        key = (int(match.group("layer")), match.group("stage"))
        records[key] = parse_values(match, index, tp_scale)
    return records


def relative_percent(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1e-12) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_log", type=Path)
    parser.add_argument("fresh_log", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--replay", type=int, default=1)
    parser.add_argument("--graph-row", type=int, default=0)
    parser.add_argument("--fresh-row", type=int, default=6)
    parser.add_argument("--fresh-tp-scale", type=float, default=16.0)
    parser.add_argument("--reference-graph-rank", type=int)
    parser.add_argument("--reference-replay", type=int)
    parser.add_argument("--reference-graph-row", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    graph = parse_graph(args.graph_log, args.rank, args.replay, args.graph_row)
    if args.reference_graph_rank is None:
        fresh = parse_fresh(args.fresh_log, args.fresh_row, args.fresh_tp_scale)
    else:
        fresh = parse_graph(
            args.fresh_log,
            args.reference_graph_rank,
            args.reference_replay if args.reference_replay is not None else args.replay,
            args.reference_graph_row,
        )
    common = sorted(set(graph) & set(fresh))
    print(f"graph={len(graph)} fresh={len(fresh)} common={len(common)}")
    print("layer stage                         abs_graph   abs_fresh   abs_diff%   sum_diff%    sq_diff%")
    first_over = None
    for key in common:
        g = graph[key]
        f = fresh[key]
        diffs = {name: relative_percent(g[name], f[name]) for name in ("abs", "sum", "sq")}
        if first_over is None and diffs["abs"] > args.threshold:
            first_over = (key, diffs, g, f)
        print(
            f"{key[0]:5d} {key[1]:28s} {g['abs']:11.6f} {f['abs']:11.6f} "
            f"{diffs['abs']:10.4f} {diffs['sum']:11.4f} {diffs['sq']:11.4f}"
        )
    print(f"first_abs_over_{args.threshold}%={first_over}")


if __name__ == "__main__":
    main()

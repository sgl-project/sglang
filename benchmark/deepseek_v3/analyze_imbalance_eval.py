#!/usr/bin/env python3
"""
Post-process logs produced by `run_imbalance_eval.py`.

Given an output directory that contains files like:
  server_<mode>_<eplb|no_eplb>_in<input_len>.log

This script parses `[deepep_eplb_load]` entries and computes the average
imbalance per stage across layers (rank0 only), then prints a compact summary
and writes `results_analyzed.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


_LINE_RE = re.compile(
    r"\[deepep_eplb_load\].*?"
    r"mode=(\w+).*?"
    r"layer=(\d+).*?"
    r"ep_rank=(\d+)/(\d+).*?"
    r"stage=(\w+).*?"
    r"imbal=([\d.]+)x"
)


@dataclass(frozen=True)
class CaseKey:
    mode: str
    enable_eplb: bool
    input_len: int


def _parse_one_log(path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns:
        stage -> layer_id -> [imbal_values]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    stage_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for line in content.split("\n"):
        for m in _LINE_RE.finditer(line):
            _mode, layer_id, ep_rank, _ep_world, stage, imbal = m.groups()
            if ep_rank == "0":
                stage_data[stage][layer_id].append(float(imbal))

    return stage_data


def _avg_stage(stage_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for stage, layer_map in stage_data.items():
        vals: List[float] = []
        for _layer, vs in layer_map.items():
            vals.extend(vs)
        out[stage] = (sum(vals) / len(vals)) if vals else 0.0
    return out


def _discover_logs(out_dir: str) -> List[Tuple[CaseKey, str]]:
    logs: List[Tuple[CaseKey, str]] = []
    pat = re.compile(r"^server_(?P<mode>[^_]+)_(?P<eplb>eplb|no_eplb)_in(?P<in>\d+)\.log$")
    for name in sorted(os.listdir(out_dir)):
        m = pat.match(name)
        if not m:
            continue
        mode = m.group("mode")
        enable_eplb = m.group("eplb") == "eplb"
        input_len = int(m.group("in"))
        logs.append((CaseKey(mode=mode, enable_eplb=enable_eplb, input_len=input_len), os.path.join(out_dir, name)))
    return logs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = args.out_dir
    items = _discover_logs(out_dir)
    if not items:
        raise SystemExit(f"No server_*.log found under: {out_dir}")

    results = []
    for key, path in items:
        stage_data = _parse_one_log(path)
        avg = _avg_stage(stage_data)
        results.append(
            {
                "mode": key.mode,
                "enable_eplb": key.enable_eplb,
                "input_len": key.input_len,
                "avg_imbalance": avg,
                "layers_per_stage": {k: len(v) for k, v in stage_data.items()},
                "log_file": os.path.basename(path),
            }
        )

    out_path = os.path.join(out_dir, "results_analyzed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print a compact summary
    by_in = defaultdict(list)
    for r in results:
        by_in[r["input_len"]].append(r)

    print(f"[ok] wrote {out_path}")
    for in_len in sorted(by_in.keys()):
        print(f"\n=== input_len={in_len} ===")
        print(f"{'Mode':<12} {'EPLB':<6} {'pre_eplb':<10} {'post_eplb':<10} {'post_wf':<10}  layers(pre/post/postwf)")
        for r in sorted(by_in[in_len], key=lambda x: (x["mode"], x["enable_eplb"])):
            avg = r["avg_imbalance"]
            layers = r.get("layers_per_stage", {})
            pre = avg.get("pre_eplb", 0.0)
            post = avg.get("post_eplb", 0.0)
            postwf = avg.get("post_waterfill", 0.0)
            pre_s = f"{pre:.4f}x" if pre else "N/A"
            post_s = f"{post:.4f}x" if post else "N/A"
            postwf_s = f"{postwf:.4f}x" if postwf else "N/A"
            layers_s = f"{layers.get('pre_eplb',0)}/{layers.get('post_eplb',0)}/{layers.get('post_waterfill',0)}"
            print(
                f"{r['mode']:<12} {('Y' if r['enable_eplb'] else 'N'):<6} {pre_s:<10} {post_s:<10} {postwf_s:<10}  {layers_s}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


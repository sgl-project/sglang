#!/usr/bin/env python3
"""Summarize SGLANG_DEBUG_DSV41_PROBE_STATS JSON dumps (idea 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def flags(ev: dict) -> list[str]:
    out = []
    if ev.get("none"):
        out.append("NONE")
        return out
    if ev.get("n") == 0:
        out.append("EMPTY")
    if ev.get("nan"):
        out.append(f"NAN={ev['nan']}")
    if ev.get("inf"):
        out.append(f"INF={ev['inf']}")
    rms = ev.get("rms")
    name = ev.get("name") or ""
    if isinstance(rms, (int, float)):
        if rms == 0.0:
            out.append("RMS0")
        elif rms > 1e4:
            out.append("RMS_HUGE")
        elif rms > 100 and any(
            s in name for s in (".q", "qb.", "q_lora", "csa2.q")
        ):
            out.append("Q_EXPLODE")
        elif rms < 1e-6 and (ev.get("n") or 0) > 0:
            out.append("RMS_TINY")
    z = ev.get("zero_frac")
    if isinstance(z, float) and z > 0.95:
        out.append("MOSTLY_ZERO")
    if ev.get("n_neg"):
        out.append(f"NEG={ev['n_neg']}")
    if name.endswith("_pad") and isinstance(rms, (int, float)) and rms > 100:
        out.append("PAD_JUNK")
    if name.endswith("_local") and isinstance(rms, (int, float)) and rms > 100:
        out.append("LOCAL_EXPLODE")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dir", nargs="?", default="/tmp/dsv41-probe")
    args = p.parse_args()
    files = sorted(Path(args.dir).glob("rank*_step*.json"))
    if not files:
        print(f"no dumps in {args.dir}", file=sys.stderr)
        return 1
    for path in files:
        data = json.loads(path.read_text())
        print(f"\n== {path.name} rank={data.get('rank')} step={data.get('step')} tag={data.get('tag')} events={len(data.get('events') or [])}")
        for ev in data.get("events") or []:
            name = ev.get("name")
            mark = flags(ev)
            extra = ""
            if name == "logits":
                extra = f" argmax={ev.get('argmax')} entropy={ev.get('entropy')} top8={ev.get('top8_ids')}"
            if "topk" in (name or "") and "n_unique" in ev:
                extra += f" unique={ev.get('n_unique')} min={ev.get('min')} max={ev.get('max')}"
            skip = {
                "name",
                "shape",
                "dtype",
                "n",
                "nan",
                "inf",
                "zero_frac",
                "rms",
                "mean",
                "absmax",
                "min",
                "max",
                "none",
                "argmax",
                "entropy",
                "top8_ids",
                "top8_vals",
                "n_unique",
                "n_neg",
            }
            meta = {k: v for k, v in ev.items() if k not in skip}
            if meta:
                extra += " " + " ".join(f"{k}={v}" for k, v in meta.items())
            rms = ev.get("rms")
            absmax = ev.get("absmax")
            shape = ev.get("shape")
            shape_s = f" {shape}" if shape else ""
            print(
                f"  {name:36} rms={rms!s:>12} absmax={absmax!s:>12}{shape_s} {','.join(mark)}{extra}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

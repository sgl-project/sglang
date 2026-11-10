#!/usr/bin/env python3
"""Bucket an nsys sqlite export of T=6 TARGET_VERIFY.

nsys export stores kernel/NVTX names as StringIds integers. Join them.
NVTX_EVENTS.globalTid and KERNEL.globalPid share a process key at id>>24.
Dummy warmup ranges are ~15 ms; real T=6 verifies are >50 ms.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


def k24(x: int) -> int:
    return int(x) >> 24


def classify(name: str) -> str:
    n = (name or "").lower()
    if "nccl" in n:
        return "nccl"
    if "spill_copy" in n or "spill_assign" in n or "spill_page" in n:
        return "spill"
    if "marlin" in n or "mxfp4_gemv" in n or ("mxfp8" in n and "gemv" in n):
        return "marlin_gemv"
    if any(
        k in n
        for k in (
            "pack_swa",
            "pack_kv",
            "pack_index",
            "index_logits",
            "sparse_decode",
            "fast_topk",
        )
    ):
        return "csa2"
    if "hc::" in n or "mix_sinkhorn" in n or "dsv41_hc" in n:
        return "csa2_hc"
    if "einsum" in n or "bmm" in n or "softmax" in n:
        return "sparse_oracle"
    if any(
        k in n
        for k in (
            "sgemm",
            "gemm",
            "cublas",
            "wmma",
            "cutlass",
            "volta_s",
            "volta_h",
            "volta_fp16",
        )
    ):
        return "gemm"
    if "memcpy" in n:
        return "copy_elem"
    if "elementwise" in n or "direct_copy" in n or "vectorized" in n:
        return "torch_elem"
    return "other"


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cur.fetchone() is not None


def first_table(cur: sqlite3.Cursor, candidates: list[str]) -> str | None:
    for c in candidates:
        if table_exists(cur, c):
            return c
    return None


def load_strings(cur: sqlite3.Cursor) -> dict[int, str]:
    if not table_exists(cur, "StringIds"):
        return {}
    return {int(i): v for i, v in cur.execute("SELECT id, value FROM StringIds")}


def resolve_name(strings: dict[int, str], *ids_or_names) -> str:
    for x in ids_or_names:
        if x is None:
            continue
        if isinstance(x, str) and x:
            return x
        if isinstance(x, int):
            if x in strings:
                return strings[x] or ""
            # already a name-like int that missed the join
            continue
    return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--verify-nvtx", default="dsv41_target_verify")
    p.add_argument("--min-verify-ms", type=float, default=50.0)
    args = p.parse_args()
    path = Path(args.sqlite)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    strings = load_strings(cur)

    kern_t = first_table(
        cur,
        [
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            "KERNEL",
        ],
    )
    if kern_t is None:
        print("ERROR: no kernel table", file=sys.stderr)
        return 1
    nvtx_t = first_table(cur, ["NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_MARKER"])
    if nvtx_t is None:
        print("ERROR: no NVTX table", file=sys.stderr)
        return 1

    ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for start, end, tid in cur.execute(
        f"SELECT start, end, globalTid FROM {nvtx_t} "
        "WHERE text = ? AND end > start",
        (args.verify_nvtx,),
    ):
        if (end - start) / 1e6 > args.min_verify_ms:
            ranges[k24(tid)].append((int(start), int(end)))

    n_real = sum(len(v) for v in ranges.values())
    print(
        f"real dsv41_target_verify ranges (>{args.min_verify_ms} ms): {n_real} "
        f"across {len(ranges)} ranks",
        flush=True,
    )

    def in_real(rk: int, mid: int):
        for s, e in ranges[rk]:
            if s <= mid <= e:
                return s, e
        return None

    per_range: dict[tuple, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    nkern_range: dict[tuple, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    nlaunch_range: dict[tuple, int] = defaultdict(int)
    other_top: Counter[str] = Counter()

    cur.execute(f"PRAGMA table_info({kern_t})")
    kcols = [r[1] for r in cur.fetchall()]
    dcol = "demangledName" if "demangledName" in kcols else "shortName"
    scol = "shortName" if "shortName" in kcols else dcol
    pidcol = "globalPid" if "globalPid" in kcols else "globalTid"
    q = f"SELECT start, end, {pidcol}, {dcol}, {scol} FROM {kern_t}"
    for start, end, gpid, dname, sname in cur.execute(q):
        rk = k24(gpid)
        hit = in_real(rk, (start + end) // 2)
        if not hit:
            continue
        name = resolve_name(strings, dname, sname)
        b = classify(name)
        dur = (end - start) / 1e6
        per_range[(rk, hit)][b] += dur
        nkern_range[(rk, hit)][b] += 1
        nlaunch_range[(rk, hit)] += 1
        if b == "other":
            other_top[name.split("(")[0][-80:]] += (end - start)

    sync_range: dict[tuple, int] = defaultdict(int)
    launch_api_range: dict[tuple, int] = defaultdict(int)
    graph_launch_range: dict[tuple, int] = defaultdict(int)
    sync_time: dict[tuple, float] = defaultdict(float)
    api_t = first_table(
        cur,
        ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"],
    )
    if api_t:
        cur.execute(f"PRAGMA table_info({api_t})")
        acols = [r[1] for r in cur.fetchall()]
        tidcol = (
            "globalTid"
            if "globalTid" in acols
            else ("globalPid" if "globalPid" in acols else None)
        )
        nid = "nameId" if "nameId" in acols else ("name" if "name" in acols else None)
        if tidcol and nid:
            for start, end, tid, nameId in cur.execute(
                f"SELECT start, end, {tidcol}, {nid} FROM {api_t}"
            ):
                name = resolve_name(strings, nameId)
                rk = k24(tid)
                hit = in_real(rk, start)
                if not hit:
                    continue
                nl = name.lower()
                if "synchronize" in nl:
                    sync_range[(rk, hit)] += 1
                    if end and end > start:
                        sync_time[(rk, hit)] += (end - start) / 1e6
                if "launchkernel" in nl:
                    launch_api_range[(rk, hit)] += 1
                if "graphlaunch" in nl:
                    graph_launch_range[(rk, hit)] += 1

    keys = sorted(per_range, key=lambda x: (x[0], x[1][0]))
    print("=== per-verify GPU buckets (ms) inside dsv41_target_verify ===")
    agg: dict[str, list[float]] = defaultdict(list)
    bucket_names = (
        "nccl",
        "spill",
        "gemm",
        "sparse_oracle",
        "csa2",
        "csa2_hc",
        "marlin_gemv",
        "torch_elem",
        "copy_elem",
        "other",
    )
    for key in keys:
        rk, (s, e) = key
        wall = (e - s) / 1e6
        b = per_range[key]
        gpu = sum(b.values())
        idle = wall - gpu
        rec = dict(
            wall=wall,
            gpu=gpu,
            idle=idle,
            klaunch=nlaunch_range[key],
            syncs=sync_range[key],
            sync_ms=sync_time[key],
            api_launch=launch_api_range[key],
            graph_launch=graph_launch_range[key],
        )
        for name in bucket_names:
            rec[name] = b[name]
        print(
            f"wall={wall:6.1f} gpu={gpu:6.1f} idle={idle:6.1f} "
            f"nccl={b['nccl']:6.1f} spill={b['spill']:5.1f} "
            f"gemm={b['gemm']:5.1f} oracle={b['sparse_oracle']:5.1f} "
            f"csa2={b['csa2']:5.1f} hc={b['csa2_hc']:5.1f} "
            f"marlin={b['marlin_gemv']:5.1f} elem={b['torch_elem']:5.1f} "
            f"other={b['other']:5.1f} kl={nlaunch_range[key]:5d} "
            f"gl={graph_launch_range[key]:4d} sync={sync_range[key]:4d} "
            f"sync_ms={sync_time[key]:6.1f}"
        )
        for k, v in rec.items():
            agg[k].append(v)

    print("\n=== mean / min / max (ms unless count) ===")
    print("n=", len(keys))
    for k, xs in agg.items():
        print(
            f"  {k:16s} mean={sum(xs)/len(xs):8.1f}  "
            f"min={min(xs):8.1f}  max={max(xs):8.1f}"
        )

    print("\n=== top other kernel time (ms total across verifies) ===")
    for name, ns in other_top.most_common(15):
        print(f"  {ns / 1e6:8.1f}  {name}")

    print("\n=== aligned verify index nccl/spill min-max ===")
    by_rank: dict[int, list[tuple]] = defaultdict(list)
    for key in keys:
        rk, (s, e) = key
        by_rank[rk].append(
            (
                s,
                per_range[key]["nccl"],
                per_range[key]["spill"],
                (e - s) / 1e6,
                nlaunch_range[key],
                sync_range[key],
            )
        )
    nver = min(len(v) for v in by_rank.values()) if by_rank else 0
    for i in range(nver):
        nccls, spills, walls, launches, syncs = [], [], [], [], []
        for seq in by_rank.values():
            seq = sorted(seq)
            nccls.append(seq[i][1])
            spills.append(seq[i][2])
            walls.append(seq[i][3])
            launches.append(seq[i][4])
            syncs.append(seq[i][5])
        print(
            f"  v{i}: wall {min(walls):.1f}-{max(walls):.1f} "
            f"nccl {min(nccls):.1f}-{max(nccls):.1f} "
            f"spill {min(spills):.1f}-{max(spills):.1f} "
            f"kl {min(launches)}-{max(launches)} "
            f"sync {min(syncs)}-{max(syncs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

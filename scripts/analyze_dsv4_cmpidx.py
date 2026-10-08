#!/usr/bin/env python3
"""DSV4/A5 probe-log analyzer - compact hit-vs-miss diff.

Parses the [CMPIDX]/[IDXK]/[C4KV]/[OSHAPE] probe lines emitted by
sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py, splits the
lines into requests, classifies the cache-MISS request (largest prefill) versus
the cache-HIT requests, and prints ONLY the differences. Paste this output back
instead of the whole log.

Usage:
    python3 scripts/analyze_dsv4_cmpidx.py /tmp/run.log

Probe the run first, e.g.:
    DSV4_DUMP_CMPIDX=all DSV4_DUMP_IDXK=0 DSV4_DUMP_C4KV=0 DSV4_DUMP_OSHAPE=all \
        DSV4_DUMP_MIN_POS=17522 DSV4_DUMP_MAX_POS=17650 \
        bash start_test.sh 2>&1 | tee /tmp/run.log
    bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh
"""
import re
import sys
from collections import defaultdict

TAG = re.compile(r"\[(CMPIDX|IDXK|C4KV|OSHAPE)\]")
KV = re.compile(r"(\w+)=(\[[^\]]*\]|\([^)]*\)|[^\s]+)")
MAX_REQ_SHOWN = 20
MAX_DIFF_SHOWN = 8


def _int(d, k, default=-1):
    try:
        return int(d.get(k, default))
    except (TypeError, ValueError):
        return default


def parse(path):
    recs = defaultdict(list)
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            mt = TAG.search(ln)
            if mt:
                recs[mt.group(1)].append(dict(KV.findall(ln)))
    return recs


def segment(rows):
    """Split rows into requests (new request when lastpos decreases).

    Returns [{'steps': {lastpos: {layer: row}}, 'ntok0': first ntok or None}].
    """
    reqs = []
    for r in rows:
        lp = _int(r, "lastpos")
        if not reqs or lp < reqs[-1]["last_lp"]:
            reqs.append({"steps": {}, "last_lp": lp, "ntok0": None})
        req = reqs[-1]
        req["last_lp"] = lp
        if req["ntok0"] is None and "ntok" in r:
            req["ntok0"] = _int(r, "ntok")
        req["steps"].setdefault(lp, {})[r.get("layer")] = r
    return reqs


def sig(tag, rec):
    if tag == "OSHAPE":
        return "|".join(
            rec.get(k, "?")
            for k in ("ratio", "ori_page", "ori_bt", "s2Size",
                      "cmp_page", "cmp_bt", "sparseK")
        )
    return rec.get("md5")


def compare(tag, a_steps, b_steps):
    diffs = []
    common = sorted(set(a_steps) & set(b_steps))
    for lp in common:
        for ly in set(a_steps[lp]) & set(b_steps[lp]):
            a, b = sig(tag, a_steps[lp][ly]), sig(tag, b_steps[lp][ly])
            if a != b:
                diffs.append((lp, ly, a, b))
    return len(common), diffs


def main(path):
    recs = parse(path)
    for tag in ("CMPIDX", "IDXK", "C4KV", "OSHAPE"):
        print(f"parsed {tag}: {len(recs.get(tag, []))} lines")
    if not recs.get("CMPIDX"):
        print("\nno [CMPIDX] lines found. Enable DSV4_DUMP_CMPIDX and re-run.")
        return

    cmp_reqs = segment(recs["CMPIDX"])
    miss_i = max(range(len(cmp_reqs)), key=lambda i: (cmp_reqs[i]["ntok0"] or 0))
    print("\n== requests (classified by [CMPIDX] prefill ntok) ==")
    for i, rq in enumerate(cmp_reqs[:MAX_REQ_SHOWN]):
        role = "MISS" if i == miss_i else "HIT "
        print(f"  req{i:<2} {role} prefill_ntok={rq['ntok0']}"
              f"  steps={len(rq['steps'])}")
    if len(cmp_reqs) > MAX_REQ_SHOWN:
        print(f"  ... {len(cmp_reqs) - MAX_REQ_SHOWN} more requests")
    miss = cmp_reqs[miss_i]
    print(f"\nMISS = req{miss_i} (largest prefill ntok); each HIT compared to it.")

    for tag in ("CMPIDX", "IDXK", "C4KV", "OSHAPE"):
        rows = recs.get(tag, [])
        if not rows:
            continue
        reqs = cmp_reqs if tag == "CMPIDX" else segment(rows)
        print(f"\n== [{tag}] ==")
        if tag != "CMPIDX" and len(reqs) != len(cmp_reqs):
            print(f"  note: {len(reqs)} requests here vs {len(cmp_reqs)} in "
                  "[CMPIDX]; aligning by index")
        for i, rq in enumerate(reqs):
            if i == miss_i or i >= len(cmp_reqs):
                continue
            if miss_i >= len(reqs):
                print(f"  req{miss_i} missing for [{tag}]; cannot compare")
                continue
            n, diffs = compare(tag, reqs[miss_i]["steps"], rq["steps"])
            print(f"  MISS(req{miss_i}) vs HIT(req{i}):"
                  f" common(lastpos,layer)={n}  differing={len(diffs)}")
            for (lp, ly, a, b) in diffs[:MAX_DIFF_SHOWN]:
                print(f"     lastpos={lp} layer={ly}"
                      f"\n        miss={a}\n        hit ={b}")
            if len(diffs) > MAX_DIFF_SHOWN:
                print(f"     ... {len(diffs) - MAX_DIFF_SHOWN} more")
            if not diffs:
                print("     IDENTICAL")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

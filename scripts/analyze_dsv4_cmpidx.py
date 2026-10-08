#!/usr/bin/env python3
"""DSV4/A5 probe-log analyzer - compact hit-vs-miss diff.

Parses the [CMPIDX]/[IDXK]/[C4KV]/[OSHAPE] probe lines emitted by
sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py, splits the
lines into requests, classifies the cache-MISS request (largest prefill) versus
the cache-HIT requests, and prints ONLY the differences. Paste this output back
instead of the whole log.

Usage:
    python3 scripts/analyze_dsv4_cmpidx.py /tmp/run.log

Notes on [CMPIDX]: the whole-tensor md5 is only comparable when the row count
(ntok) matches. The prefill rows have different ntok between a full-prefill miss
(e.g. 17523) and a suffix-prefill hit (e.g. 1139), so for those we compare the
LAST-row ``tail`` (same absolute query token on both sides). Decode rows
(ntok=1) are same-shape and compared by md5.
"""
import re
import sys
from collections import defaultdict

TAG = re.compile(r"\[(CMPIDX|IDXK|C4KV|OSHAPE|XIN)\]")
KV = re.compile(r"(\w+)=(\[[^\]]*\]|\([^)]*\)|[^\s]+)")
MAX_REQ_SHOWN = 20
MAX_DIFF_SHOWN = 8
OSHAPE_KEYS = ("ratio", "ori_page", "ori_bt", "s2Size", "cmp_page", "cmp_bt")


def _int(d, k, default=-1):
    try:
        return int(d.get(k, default))
    except (TypeError, ValueError):
        return default


def parse(path):
    """Parse probe lines into field-dicts per tag.

    Rows produced before c85e85c lack ``lastpos`` on [IDXK]/[C4KV]; infer it
    from the most recent [CMPIDX] line (same forward step), so old logs align.
    """
    recs = defaultdict(list)
    cur_lp = None
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            mt = TAG.search(ln)
            if not mt:
                continue
            tag = mt.group(1)
            d = dict(KV.findall(ln))
            if tag == "CMPIDX" and "lastpos" in d:
                cur_lp = d["lastpos"]
            elif "lastpos" not in d and cur_lp is not None:
                d["lastpos"] = cur_lp
            recs[tag].append(d)
    return recs


def segment(rows):
    """Split rows into requests (new request when lastpos decreases)."""
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


def oshape_sig(rec):
    return "|".join(rec.get(k, "?") for k in OSHAPE_KEYS)


def tag_sig(tag, rec):
    if tag == "OSHAPE":
        return oshape_sig(rec)
    if tag in ("IDXK", "C4KV"):
        # both carry two windows: deep prefix (written once) + tail (rewritten)
        return f"pre={rec.get('pre')}|tail={rec.get('tail')}"
    return rec.get("md5")


def compare(tag, a_steps, b_steps):
    """Return (n_lastpos, same_shape_diffs, prefill_diffs, n_prefill_rows)."""
    same_shape, prefill, n_lp, n_pre = [], [], 0, 0
    for lp in sorted(set(a_steps) & set(b_steps)):
        n_lp += 1
        for ly in set(a_steps[lp]) & set(b_steps[lp]):
            a, b = a_steps[lp][ly], b_steps[lp][ly]
            if tag == "CMPIDX" and a.get("ntok") != b.get("ntok"):
                # prefill: whole-tensor md5 not comparable -> compare last-row tail
                n_pre += 1
                ta, tb = a.get("tail", "?"), b.get("tail", "?")
                if ta != tb:
                    prefill.append((lp, ly, ta, tb))
                continue
            ea, eb = tag_sig(tag, a), tag_sig(tag, b)
            if ea != eb:
                same_shape.append((lp, ly, ea, eb))
    return n_lp, same_shape, prefill, n_pre


def main(path):
    recs = parse(path)
    for tag in ("CMPIDX", "IDXK", "C4KV", "OSHAPE", "XIN"):
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
    print(f"\nMISS = req{miss_i} (largest prefill ntok); each HIT compared to it.")

    for tag in ("CMPIDX", "IDXK", "C4KV", "OSHAPE", "XIN"):
        rows = recs.get(tag, [])
        if not rows:
            continue
        reqs = cmp_reqs if tag == "CMPIDX" else segment(rows)
        print(f"\n== [{tag}] ==")
        if tag != "CMPIDX" and len(reqs) != len(cmp_reqs):
            print(f"  note: {len(reqs)} requests here vs {len(cmp_reqs)} in "
                  "[CMPIDX]; aligning by index")
        if miss_i >= len(reqs):
            print(f"  req{miss_i} missing for [{tag}]; cannot compare")
            continue
        for i, rq in enumerate(reqs):
            if i == miss_i or i >= len(cmp_reqs):
                continue
            n, same, pre, n_pre = compare(tag, reqs[miss_i]["steps"], rq["steps"])
            print(f"  MISS(req{miss_i}) vs HIT(req{i}): lastpos={n}"
                  f"  SAME-SHAPE differing={len(same)} (real)"
                  f"  PREFILL rows={n_pre} differing={len(pre)} (tail-only)")
            for (lp, ly, a, b) in (same + pre)[:MAX_DIFF_SHOWN]:
                print(f"     lastpos={lp} layer={ly}"
                      f"\n        miss={a}\n        hit ={b}")
            if len(same) + len(pre) > MAX_DIFF_SHOWN:
                print(f"     ... {len(same) + len(pre) - MAX_DIFF_SHOWN} more")
            if not same and not pre:
                print("     IDENTICAL")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

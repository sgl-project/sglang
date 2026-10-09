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

TAG = re.compile(r"\[(CMPIDX|IDXK|C4KV|C128KV|OSHAPE|XIN|LHID)\]")
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
    if tag in ("IDXK", "C4KV", "C128KV"):
        # logical = md5 of the pages the request actually reads (via its page table)
        return f"logical={rec.get('logical')}"
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
            if (
                tag != "CMPIDX"
                and a.get("ntok") is not None
                and a.get("ntok") != b.get("ntok")
            ):
                # shape-confounded (e.g. XIN prefill-vs-suffix): not a real diff
                continue
            ea, eb = tag_sig(tag, a), tag_sig(tag, b)
            if ea != eb:
                same_shape.append((lp, ly, ea, eb))
    return n_lp, same_shape, prefill, n_pre


def report_lhid(rows):
    """Order-independent [LHID] verdict.

    Every request (4 miss + 1 hit) runs deterministically, so at a given
    (ntok, lastpos, layer) the hidden-state md5 must be a *single* value across
    all requests. A group with >1 distinct md5 is a REAL divergence at that
    layer/step. This needs no request segmentation (robust to interleaving).
    """
    groups = defaultdict(set)
    for r in rows:
        if "md5" not in r or r["md5"] == "empty":
            continue
        try:
            ntok = int(r.get("ntok", "-1"))
            lp = int(r.get("lastpos", "-1"))
        except (TypeError, ValueError):
            continue
        if ntok <= 0 or lp < 0:
            continue
        groups[(ntok, lp, _int(r, "layer"))].add(r["md5"])
    div = [(lp, ly, sorted(s)) for (ntok, lp, ly), s in groups.items() if len(s) > 1]
    if not div:
        print("  no divergence: every (ntok,lastpos,layer) group has one md5")
        print("  -> hit and miss hidden IDENTICAL on all sampled layers/steps")
        return
    div.sort(key=lambda t: (t[0], t[1]))
    lp0 = div[0][0]
    print(f"  groups={len(groups)}  divergent_groups={len(div)}"
          f"  first_lastpos={lp0}")
    print(f"  FIRST divergent step lastpos={lp0}; layers there = "
          f"{sorted(ly for lp, ly, _ in div if lp == lp0)}")
    print("  sample (lastpos, layer, distinct_md5):")
    for lp, ly, s in div[:MAX_DIFF_SHOWN]:
        print(f"     lastpos={lp} layer={ly}  md5s={s}")
    if len(div) > MAX_DIFF_SHOWN:
        print(f"     ... {len(div) - MAX_DIFF_SHOWN} more divergent groups")


def main(path):
    recs = parse(path)
    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "OSHAPE", "XIN", "LHID"):
        print(f"parsed {tag}: {len(recs.get(tag, []))} lines")
    if recs.get("CMPIDX"):
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
    else:
        # No [CMPIDX]: fall back to per-tag segments; req0 = MISS (best effort).
        cmp_reqs = None
        miss_i = 0
        print("\nno [CMPIDX] lines; per-tag request split used (req0 = MISS).")

    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "OSHAPE", "XIN", "LHID"):
        rows = recs.get(tag, [])
        if not rows:
            continue
        if tag == "LHID":
            print("\n== [LHID] per-layer hidden (order-independent) ==")
            report_lhid(rows)
            continue
        reqs = cmp_reqs if (tag == "CMPIDX" and cmp_reqs is not None) else segment(rows)
        print(f"\n== [{tag}] ==")
        if cmp_reqs is not None and tag != "CMPIDX" and len(reqs) != len(cmp_reqs):
            print(f"  note: {len(reqs)} requests here vs {len(cmp_reqs)} in "
                  "[CMPIDX]; aligning by index")
        if not reqs or miss_i >= len(reqs):
            print(f"  req{miss_i} missing for [{tag}]; cannot compare")
            continue
        for i, rq in enumerate(reqs):
            if i == miss_i or (cmp_reqs is not None and i >= len(cmp_reqs)):
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

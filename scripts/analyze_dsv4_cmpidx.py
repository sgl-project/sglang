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

TAG = re.compile(r"\[(CMPIDX|IDXK|C4KV|C128KV|C128X|OSHAPE|XIN|LHID|IDXIN|LIMETA)\]")
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
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for ln in fh:
            ln = ln.lstrip("\ufeff")
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


def report_group(rows, key, label, extra=()):
    """Order-independent verdict for any md5-carrying tag.

    All requests are deterministic, so at a given (ntok, lastpos, layer) the
    ``key`` value (md5 / logical) must be a SINGLE value across all requests.
    A group with >1 distinct value is a REAL divergence at that layer/step.
    Needs no request segmentation (robust to warmup/interleaving).
    ``extra`` appends more fields to the signature (e.g. nblk/ptab).
    """
    groups = defaultdict(set)
    have_ntok = any("ntok" in r for r in rows)
    for r in rows:
        v = r.get(key)
        if v is None or v == "empty":
            continue
        if extra:
            v = "|".join([str(v)] + [f"{e}={r.get(e, '?')}" for e in extra])
        try:
            lp = int(r.get("lastpos", "-1"))
        except (TypeError, ValueError):
            continue
        if lp < 0:
            continue
        ntok = None
        if "ntok" in r:
            try:
                ntok = int(r["ntok"])
            except (TypeError, ValueError):
                ntok = None
        if ntok is not None and ntok <= 0:
            continue
        groups[(ntok, lp, _int(r, "layer"))].add(v)
    if not have_ntok:
        print(f"  [{label}] NOTE: log has no ntok (old probe) -> grouped by "
              "(lastpos,layer); the prefill row may be shape-confounded")
    div = [(lp, ly, sorted(s)) for (ntok, lp, ly), s in groups.items() if len(s) > 1]
    if not div:
        print(f"  [{label}] no divergence: every (ntok,lastpos,layer) group "
              f"has one {key}")
        print("  -> hit and miss identical on all sampled layers/steps")
        return
    div.sort(key=lambda t: (t[0], t[1]))
    lp0 = div[0][0]
    print(f"  [{label}] groups={len(groups)}  divergent={len(div)}"
          f"  first_lastpos={lp0}")
    print(f"  FIRST divergent step lastpos={lp0}; layers there = "
          f"{sorted(ly for lp, ly, _ in div if lp == lp0)}")
    div_lps = sorted(set(lp for lp, _, _ in div))
    _show = div_lps[:30]
    print(f"  divergent lastpos ({len(div_lps)}): {_show}"
          + (" ..." if len(div_lps) > 30 else ""))
    print(f"  sample (lastpos, layer, distinct_{key}):")
    for lp, ly, s in div[:MAX_DIFF_SHOWN]:
        print(f"     lastpos={lp} layer={ly}  {key}s={s}")
    if len(div) > MAX_DIFF_SHOWN:
        print(f"     ... {len(div) - MAX_DIFF_SHOWN} more divergent groups")


def report_lhid(rows):
    report_group(rows, "md5", "LHID")


def _parse_blkx(rec, field="blkx"):
    """Parse a ``<field>=[<block>:<hash>, ...]`` field into {block: hash}."""
    v = rec.get(field)
    if not v:
        return {}
    s = v.strip().replace('"', "'")
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    out = {}
    for item in s.split("'"):
        item = item.strip().strip(",").strip()
        if ":" not in item:
            continue
        b, h = item.split(":", 1)
        try:
            out[int(b)] = h
        except ValueError:
            continue
    return out


def report_blkx(rows, label, field="blkx"):
    """Per-c128-block (``position // 128``) hit==miss verdict.

    A block's hash must be unique across all requests (deterministic), so a
    ``(layer, block)`` group with >1 distinct hash is a REAL divergence there.
    This is NOT shape-confounded (a block = the same absolute positions on both
    the full-prefill miss and the suffix hit).  Locates the FIRST divergent
    block and the layers diverging there (=> the first divergent layer).
    """
    groups = defaultdict(set)
    for r in rows:
        ly = _int(r, "layer")
        for b, h in _parse_blkx(r, field).items():
            groups[(ly, b)].add(h)
    if not groups:
        print(f"  [{label}] no blkx= field (probe without per-block hashing)")
        return
    div = [(ly, b, sorted(s)) for (ly, b), s in groups.items() if len(s) > 1]
    if not div:
        print(f"  [{label}] no per-block divergence: every (layer,block) "
              f"group has one hash")
        return
    div.sort(key=lambda t: (t[1], t[0]))  # first by block, then by layer
    b0 = div[0][1]
    layers0 = sorted(ly for ly, b, _ in div if b == b0)
    print(f"  [{label}] groups={len(groups)} divergent={len(div)} "
          f"first_block={b0}")
    print(f"  FIRST divergent block={b0}; layers there = {layers0}")
    blocks = sorted(set(b for _, b, _ in div))
    print(f"  divergent blocks ({len(blocks)}): {blocks[:40]}"
          + (" ..." if len(blocks) > 40 else ""))
    for ly, b, s in div[:MAX_DIFF_SHOWN]:
        print(f"     block={b} layer={ly}  hashes={s}")
    if len(div) > MAX_DIFF_SHOWN:
        print(f"     ... {len(div) - MAX_DIFF_SHOWN} more divergent groups")


def report_uniq(rows, label, fields, key="mode"):
    """Print the distinct tuples of ``fields`` grouped by ``key``.

    For scalar probes ([IDXIN]/[LIMETA]) the whole line differs by construction
    (hit vs miss have different lengths), so we only surface WHICH values appear
    -- enough to see e.g. slq=1139 (hit) vs slq=17523 (miss).
    """
    by = defaultdict(set)
    for r in rows:
        by[r.get(key, "?")].add(tuple(r.get(f, "?") for f in fields))
    for k in sorted(by, key=str):
        vals = sorted(by[k], key=str)
        print(f"  [{label}] {key}={k}  distinct({','.join(fields)})={len(vals)}")
        for t in vals[:MAX_DIFF_SHOWN]:
            print("     " + "  ".join(f"{f}={v}" for f, v in zip(fields, t)))


def main(path):
    recs = parse(path)
    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA"):
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

    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA"):
        rows = recs.get(tag, [])
        if not rows:
            continue
        if tag == "IDXIN":
            print("\n== [IDXIN] c4-indexer raw inputs (per-block q/w + seq lens) ==")
            report_blkx(rows, "IDXIN.q_quant", field="qblk")
            report_blkx(rows, "IDXIN.weights", field="wblk")
            report_uniq(rows, "IDXIN", ("slq", "slk", "qshape", "wshape"))
            continue
        if tag == "LIMETA":
            print("\n== [LIMETA] metadata core-partition ==")
            report_uniq(
                rows, "LIMETA", ("slq", "slk", "base", "nLI", "nLD"), key="bs"
            )
            report_uniq(rows, "LIMETA.LI", ("LI",), key="bs")
            report_uniq(rows, "LIMETA.LD", ("LD",), key="bs")
            continue
        if tag == "LHID":
            print("\n== [LHID] per-layer hidden (order-independent) ==")
            report_lhid(rows)
            print("  ---- [LHID] per-c128-block ----")
            report_blkx(rows, "LHID")
            continue
        if tag == "C128X":
            print("\n== [C128X] c128 compressor input x (per-block) ==")
            report_blkx(rows, "C128X")
            continue
        if tag == "C128KV":
            print("\n== [C128KV] c128 compressed-KV (order-independent) ==")
            report_group(rows, "logical", "C128KV", extra=("nblk", "ptab"))
            continue
        if tag == "CMPIDX":
            print("\n== [CMPIDX] c4-indexer top-k (per-c128-block, S167.3) ==")
            report_blkx(rows, "CMPIDX")
            # fall through to the whole-tensor segment compare below as well
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

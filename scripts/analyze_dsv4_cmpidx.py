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

TAG = re.compile(r"\[(CMPIDX|IDXK|C4KV|C128KV|C128X|OSHAPE|XIN|LHID|IDXIN|LIMETA|C4ST)\]")
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


def _is_prefill(rec):
    """True unless the probe line is a decode step (``mode=2``).

    Decode steps advance lastpos each step, so their per-block hashes are not
    comparable across requests and must be excluded from the per-block verdict.
    """
    return rec.get("mode") != "2"


def report_blkx(rows, label, field="blkx", keep=None):
    """Per-c128-block (``position // 128``) hit==miss verdict.

    A block's hash must be unique across all requests (deterministic), so a
    ``(layer, block)`` group with >1 distinct hash is a REAL divergence there.
    This is NOT shape-confounded (a block = the same absolute positions on both
    the full-prefill miss and the suffix hit).  Locates the FIRST divergent
    block and the layers diverging there (=> the first divergent layer).
    ``keep`` optionally filters rows (e.g. prefill-only to drop decode noise).
    """
    groups = defaultdict(set)
    for r in rows:
        if keep is not None and not keep(r):
            continue
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


C4ST_RUNBOOK = (
    "[C4ST-RUNBOOK] DSV4_DUMP_C4ST=2 bash start_test.sh > /tmp/dsv4_c4st.log 2>&1 & "
    "bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_c4st.log"
)


def _c4st_start(rec):
    """First absolute position of the C4ST ``start=[...]`` field (or None)."""
    v = rec.get("start")
    if not v:
        return None
    s = v.strip().replace('"', "'")
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    for item in s.split(","):
        item = item.strip().strip("'").strip()
        if not item:
            continue
        try:
            return int(item)
        except ValueError:
            continue
    return None


def _c4st_md5(blks, block):
    """Single md5 for ``block`` over a side's blkx maps (comma-joined if >1)."""
    vals = sorted({b[block] for b in blks if b.get(block)})
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else ",".join(vals)


def _c4st_suffix(blks):
    """Compact signature of post blocks 128..136 (must match hit vs miss)."""
    return ",".join(f"{b}:{_c4st_md5(blks, b) or '?'}" for b in range(128, 137))


def _c4st_slots(recs, block, field="locx"):
    """Parse ``locx=[<block>:n<count>:[a, b, ...], ...]`` slot ids for ``block``.

    Takes an iterable of records (a side may dump several rows), unions the
    integer slot ids found for ``block``, and returns them SORTED.  Returns
    ``None`` when no record carries the block, so callers can print ``NONE``.
    Tolerates spaces, missing entries, list/quotes noise and unparseable tokens.
    """
    found = False
    slots = set()
    for rec in recs:
        v = rec.get(field)
        if not v:
            continue
        s = v.strip().replace('"', "'")
        for item in s.split("'"):
            item = item.strip().strip(",").strip()
            if item.startswith("["):
                item = item[1:].strip()
            if not item or ":" not in item:
                continue
            head, _, rest = item.partition(":")
            try:
                b = int(head)
            except ValueError:
                continue
            if b != block:
                continue
            _n, _, lst = rest.partition(":")
            lst = lst.strip()
            if lst.startswith("["):
                lst = lst[1:]
            if lst.endswith("]"):
                lst = lst[:-1]
            for tok in lst.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    slots.add(int(tok))
                except ValueError:
                    continue
            found = True
    return sorted(slots) if found else None


def verdict_c4st(rows):
    """Deterministic device A/B verdict for the [C4ST] pre/post compress state.

    Rows are split by ``tag`` (pre = before the op, post = after); only
    ``idx="1"`` (c4-indexer state) rows are considered, and rows without a tag
    are skipped as unknown.  For each tag the cache-MISS request is the record
    with the SMALLEST ``start`` (the full prefill, start=0) and the cache-HIT
    request the LARGEST ``start`` (the suffix prefill, e.g. 16384).  For every
    block seen on either side we compare BOTH the content md5 AND the
    compress-state slot ids (the ``locx`` addresses): identical content at a
    different address means the state was SHIFTED.  Block 127 is reported like
    any other block -- it is NOT required to be present or divergent.
    """

    def _blk_map(value):
        """Parse ``<block>:<hash>`` entries (quoted or bare) into {block: hash}."""
        out = {}
        if not value:
            return out
        for m in re.finditer(r"(\d+)\s*:\s*([^,\[\]\s'\"]+)", value):
            try:
                out.setdefault(int(m.group(1)), m.group(2))
            except ValueError:
                continue
        return out

    def _slot_map(value, block):
        """Parse ``<block>:n<count>:[a, b, ...]`` slot ids for block, or None."""
        if not value:
            return None
        found = False
        slots = set()
        for m in re.finditer(r"(\d+)\s*:\s*n\d*\s*:\s*\[([^\]]*)\]", value):
            if int(m.group(1)) != block:
                continue
            found = True
            for tok in m.group(2).split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    slots.add(int(tok))
                except ValueError:
                    continue
        return sorted(slots) if found else None

    def _side_slots(recs, block):
        found = False
        slots = set()
        for r in recs:
            got = _slot_map(r.get("locx"), block)
            if got is not None:
                found = True
                slots.update(got)
        return sorted(slots) if found else None

    def _content(mmiss, mhit):
        if mmiss is None or mhit is None:
            return "N/A"
        return "DIFF" if mmiss != mhit else "SAME"

    def _slots_cmp(smiss, shit):
        if smiss is None or shit is None:
            return "N/A"
        return "SHIFTED" if smiss != shit else "SAME"

    def _fmt(slots):
        return str(slots) if slots is not None else "NONE"

    # 1. split by tag, keep only idx="1" (rows without a tag are skipped).
    by_tag = {"pre": [], "post": []}
    for r in rows:
        if r.get("idx") != "1":
            continue
        t = r.get("tag")
        if t in by_tag:
            by_tag[t].append(r)

    # 2. per tag: MISS = smallest start, HIT = largest start.
    sides = {}
    for name in ("pre", "post"):
        parsed = []
        for r in by_tag[name]:
            st = _c4st_start(r)
            if st is not None:
                parsed.append((st, r))
        if not parsed:
            print(f"[C4ST-VERDICT] no C4ST {name} pairing")
            continue
        starts = [st for st, _ in parsed]
        miss_st, hit_st = min(starts), max(starts)
        miss_recs = [r for st, r in parsed if st == miss_st]
        hit_recs = [r for st, r in parsed if st == hit_st]
        sides[name] = {
            "miss_recs": miss_recs,
            "hit_recs": hit_recs,
            "miss_blks": [_blk_map(r.get("blkx")) for r in miss_recs],
            "hit_blks": [_blk_map(r.get("blkx")) for r in hit_recs],
        }

    if not sides:
        return

    # 3. per (tag, block) compact verdict; block set = union of every blkx seen.
    present = set()
    for s in sides.values():
        for bm in s["miss_blks"] + s["hit_blks"]:
            present.update(bm)
    blocks = sorted(present)

    out = []  # (tag, block, miss_md5, hit_md5, content, slots, miss_slots, hit_slots)
    for name in ("pre", "post"):
        s = sides.get(name)
        if s is None:
            continue
        for b in blocks:
            mmiss = _c4st_md5(s["miss_blks"], b)
            mhit = _c4st_md5(s["hit_blks"], b)
            smiss = _side_slots(s["miss_recs"], b)
            shit = _side_slots(s["hit_recs"], b)
            out.append((name, b, mmiss, mhit, _content(mmiss, mhit),
                        _slots_cmp(smiss, shit), smiss, shit))

    # keep it compact: with many blocks only print the interesting ones.
    show_all = len(blocks) <= 20
    for (name, b, mmiss, mhit, content, slots, smiss, shit) in out:
        if not show_all and content != "DIFF" and slots != "SHIFTED":
            continue
        print(f"[C4ST-VERDICT] {name} block={b} "
              f"miss={mmiss if mmiss else 'NONE'} "
              f"hit={mhit if mhit else 'NONE'} "
              f"content={content} slots={slots} "
              f"miss_slots={_fmt(smiss)} hit_slots={_fmt(shit)}")

    # 4. one decision line for the post-op (suffix) state.
    post_diff = [b for (n, b, _, _, c, _, _, _) in out
                 if n == "post" and c == "DIFF"]
    post_shift = [b for (n, b, _, _, _, sl, _, _) in out
                  if n == "post" and sl == "SHIFTED"]
    boundary127 = "N/A"
    for (n, b, _, _, c, _, _, _) in out:
        if n == "post" and b == 127:
            boundary127 = c
    print("[C4ST-VERDICT] DECISION "
          f"post_content_diff_blocks={','.join(str(x) for x in post_diff)} "
          f"post_slot_shift_blocks={','.join(str(x) for x in post_shift)} "
          f"boundary127={boundary127}")
    if boundary127 == "SAME" and post_diff:
        print("=> boundary history OK; divergence is in the SUFFIX state "
              "produced by the op")


def main(path):
    recs = parse(path)
    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA", "C4ST"):
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

    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA", "C4ST"):
        rows = recs.get(tag, [])
        if not rows:
            continue
        if tag == "C4ST":
            print("\n== [C4ST] c4 compress STATE rows (indexer idx=1, per-block) ==")
            report_blkx(
                rows,
                "C4ST.idx1.pre",
                keep=lambda r: r.get("idx") == "1" and r.get("tag") == "pre",
            )
            report_blkx(
                rows,
                "C4ST.idx1.post",
                keep=lambda r: r.get("idx") == "1" and r.get("tag") == "post",
            )
            report_blkx(
                rows,
                "C4ST.idx1.any",
                keep=lambda r: r.get("idx") == "1" and not r.get("tag"),
            )
            continue
        if tag == "IDXIN":
            print("\n== [IDXIN] c4-indexer raw inputs (per-block q/w + seq lens) ==")
            report_blkx(rows, "IDXIN.q_quant", field="qblk", keep=_is_prefill)
            report_blkx(rows, "IDXIN.weights", field="wblk", keep=_is_prefill)
            report_uniq(rows, "IDXIN.seq", ("slq", "slk", "qshape", "wshape"))
            report_uniq(rows, "IDXIN.keymeta", ("klog", "kraw", "meta", "bt"))
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
            print("\n== [CMPIDX] c4-indexer top-k (per-c128-block, prefill) ==")
            report_blkx(rows, "CMPIDX", keep=_is_prefill)
            # fall through to the whole-tensor segment compare below as well
        if tag == "IDXK":
            print("\n== [IDXK] c4 index-K (per-logical-page, prefill) ==")
            report_blkx(rows, "IDXK", keep=_is_prefill)
            report_uniq(rows, "IDXK.ptab", ("ptab",))
            # fall through to the whole-logical segment compare below as well
        if tag == "C4KV":
            print("\n== [C4KV] c4 attention-KV (per-logical-page, prefill) ==")
            report_blkx(rows, "C4KV", keep=_is_prefill)
            report_uniq(rows, "C4KV.ptab", ("ptab",))
            # fall through to the whole-logical segment compare below as well
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

    verdict_c4st(recs.get("C4ST", []))
    print(C4ST_RUNBOOK)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

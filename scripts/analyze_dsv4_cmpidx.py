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
import hashlib
import re
import sys
from collections import defaultdict

TAG = re.compile(
    r"\[(CMPIDX|IDXK|C4KV|C128KV|C128X|OSHAPE|XIN|LHID|IDXIN|LIMETA"
    r"|C4ST|C4STP|KSTATELOC|C4POS|C4DEC|WSTAGE)\]"
)
# [C4DEC] kbase= contains SPACES (``mBaseNum=.. kBaseNum=.. raw_tok=..``), so
# the generic ``KV`` regex would truncate it at the first space.  Capture the
# full value up to the following ``rdcarry=`` field.
C4DEC_KBASE = re.compile(r"kbase=(.*?)\s+rdcarry=")
KV = re.compile(r"(\w+)=(\[[^\]]*\]|\([^)]*\)|[^\s]+)")
# [IDXIN] kpg= is emitted by ``_read_page_table_pages`` as
# ``n=<count>,first=[..],last=[..]`` (Python list repr, spaces after commas).
# The generic KV regex would truncate it at the first inner space, so match the
# known structured form first and fall back to the loose ``kpg=([^ ]+)`` form.
KPG_STRUCT = re.compile(r"kpg=(n=\d+,first=\[[^\]]*\],last=\[[^\]]*\])")
KPG_LOOSE = re.compile(r"kpg=([^\s]+)")
# [IDXIN] S233 additive fields. ``knorm=<mean>:<npairs>:<ropeDim>`` is a
# RoPE-INVARIANT magnitude (per-pair L2 mean) of the logical index-K slab;
# ``ropephase=r=<ratio>,sps=[..],shift=[..],npb=[..],loc=[..],abs=[..]`` is the
# call-origin RoPE phase index.  Neither contains spaces, so the generic KV
# regex captures the whole value; ``ROPE_SHIFT`` pulls the ``shift`` sub-field.
ROPE_SHIFT = re.compile(r"shift=\[([^\]]*)\]")
# ``sps=[..]`` = the call's start position(s); 0 => a full prefill from position
# 0 (cache-MISS), >0 => a suffix prefill / continuation (cache-HIT).
ROPE_SPS = re.compile(r"sps=\[([^\]]*)\]")
# [KSTATELOC] is emitted positionally (no key=value for the numbers):
#   op=W <batch> <seq> <col> <sl> <blk> <row> out=<curSeqIdx>
#   op=R <batch> <seq> <col> <sl> <blk> <row>
KSTATELOC_W = re.compile(
    r"\[KSTATELOC\]\s+op=W\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"\s+out=(\d+)"
)
KSTATELOC_R = re.compile(
    r"\[KSTATELOC\]\s+op=R\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
)
SEQ_TUP = re.compile(r"\((\d+),(\d+),(\d+)\)")
# Each HTTP request to the serving endpoint logs one ``/v1/chat/completions``
# line, so those lines are the STRONGEST request boundary (one curl == one
# request).  ``parse`` stamps every probe row with ``_req`` = the number of
# boundary lines seen so far, and ``_idxin_chat_split`` groups on it.
CHAT_BOUNDARY = re.compile(r"/v1/chat/completions")
MAX_REQ_SHOWN = 20
MAX_DIFF_SHOWN = 8
MAX_OFFENDER_SHOWN = 10
MAX_ALIGNED_SHOWN = 20
OSHAPE_KEYS = ("ratio", "ori_page", "ori_bt", "s2Size", "cmp_page", "cmp_bt")


def _int(d, k, default=-1):
    try:
        return int(d.get(k, default))
    except (TypeError, ValueError):
        return default


def _float(d, k, default=None):
    """Float value of field ``k`` (or ``default`` when missing/unparseable)."""
    try:
        v = d.get(k)
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def parse(path):
    """Parse probe lines into field-dicts per tag.

    Rows produced before c85e85c lack ``lastpos`` on [IDXK]/[C4KV]; infer it
    from the most recent [CMPIDX] line (same forward step), so old logs align.

    Every probe row is also stamped with ``_req`` = the number of
    ``/v1/chat/completions`` boundary lines seen so far, so the strongest
    request split (one HTTP request) can attribute each row to its request.
    """
    recs = defaultdict(list)
    cur_lp = None
    chat_n = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for ln in fh:
            ln = ln.lstrip("\ufeff")
            if CHAT_BOUNDARY.search(ln):
                chat_n += 1
                recs["#CHAT"].append({"n": chat_n})
            mt = TAG.search(ln)
            if not mt:
                continue
            tag = mt.group(1)
            d = dict(KV.findall(ln))
            d["_req"] = chat_n
            if tag == "IDXIN" and "kpg=" in ln:
                # Overwrite the KV-truncated kpg= with the full structured value.
                d["kpg"] = _parse_kpg(ln)
            if tag == "KSTATELOC":
                # Positional line: keep the raw text for the dedicated parser.
                d["_raw"] = ln
            if tag == "C4DEC":
                # kbase= spans spaces: overwrite the KV-truncated value.
                m = C4DEC_KBASE.search(ln)
                if m:
                    d["kbase"] = m.group(1).strip()
            if tag == "CMPIDX" and "lastpos" in d:
                cur_lp = d["lastpos"]
            elif "lastpos" not in d and cur_lp is not None:
                d["lastpos"] = cur_lp
            recs[tag].append(d)
    return recs


def _parse_kpg(line):
    """Full ``kpg=`` value from an [IDXIN] line (structured, else best-effort)."""
    m = KPG_STRUCT.search(line)
    if m:
        return m.group(1)
    m = KPG_LOOSE.search(line)
    return m.group(1) if m else "?"


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


def _lhid_hpos(rec, field="hpos"):
    """Parse ``<field>=[pos:md5|pos:md5|..]`` into ``{abs_pos: md5}``.

    ``hpos`` (S252) is the per-ABSOLUTE-POSITION single-row hash of a layer's
    output hidden state, emitted in the WINDOWED mode (both ``DSV4_DUMP_MIN_POS``
    and ``DSV4_DUMP_MAX_POS`` set). The whole-tensor ``md5`` is shape-confounded
    (MISS prefill ntok=17523 vs HIT ntok=1139) and ``blkx`` keys by c128 block, so
    only ``hpos`` aligns a MISS and a HIT at the SAME absolute position for the
    per-layer sweep. Mirrors the ``[IDXIN]`` ``qpos=`` template.

    ``field`` selects the source key so the SAME parser serves the ``[XIN]``
    ``xpos=`` field (S253); default ``"hpos"`` keeps the [LHID] caller unchanged.
    """
    v = rec.get(field)
    if not v:
        return {}
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    out = {}
    for item in s.split("|"):
        if ":" not in item:
            continue
        p, h = item.split(":", 1)
        try:
            out[int(p)] = h
        except ValueError:
            continue
    return out


def report_lhid_pos(rows):
    """Per-ABSOLUTE-POSITION hit==miss verdict for the per-layer hidden sweep.

    Group by ``(layer, absolute position)``: each group's single-row hash must be
    unique across requests (deterministic), so a group with >1 distinct hash is a
    REAL divergence at that layer/position. Locates the FIRST divergent layer at
    the FIRST divergent position -> the origin layer of the divergence (bracket
    with ``[XIN]`` to tell whether it is inside that layer's read).
    """
    groups = defaultdict(set)
    for r in rows:
        ly = _int(r, "layer")
        for pos, h in _lhid_hpos(r).items():
            groups[(ly, pos)].add(h)
    if not groups:
        print("  [LHID-POS] no hpos= field (old probe, or a non-windowed run: set "
              "DSV4_DUMP_MIN_POS and DSV4_DUMP_MAX_POS)")
        return
    div = [(ly, pos, sorted(s)) for (ly, pos), s in groups.items() if len(s) > 1]
    if not div:
        print(f"  [LHID-POS] no per-position divergence: every (layer,pos) group "
              f"has one hash ({len(groups)} groups)")
        return
    div.sort(key=lambda t: (t[1], t[0]))
    p0 = div[0][1]
    layers0 = sorted(ly for ly, p, _ in div if p == p0)
    print(f"  [LHID-POS] groups={len(groups)} divergent={len(div)} first_pos={p0}")
    print(f"  FIRST divergent position={p0}; layers there = {layers0}")
    for ly, p, s in div[:MAX_DIFF_SHOWN]:
        print(f"     pos={p} layer={ly}  hashes={s}")
    if len(div) > MAX_DIFF_SHOWN:
        print(f"     ... {len(div) - MAX_DIFF_SHOWN} more divergent groups")


LHID_ACCEPT_RUNBOOK = (
    "[LHID-ACCEPT-RUNBOOK] windowed hpos probe (MISS vs HIT per absolute "
    "position):\n"
    "  DSV4_DUMP_LHID=<layers> DSV4_DUMP_C4POS=1 "
    "DSV4_DUMP_MIN_POS=<win_lo> DSV4_DUMP_MAX_POS=<win_hi> "
    "<serve> ; <curl miss> ; <curl hit> ; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_lhid.log\n"
    "  S1 gate: every dumped decoder layer's hpos must be IDENTICAL across "
    "MISS/HIT at every shared absolute position (divergent=0). S2 sanity: two "
    "MISS requests must be IDENTICAL at every shared position. S3 is MANUAL "
    "(curl#1..5 output-text byte-equality)."
)


def _lhid_by_layer(group):
    """``{layer: {pos: md5}}`` of one request's [LHID] ``hpos`` rows.

    The FIRST row carrying an absolute position wins (one h hash per position),
    mirroring ``_xin_by_layer`` for the ``[XIN]`` ``xpos`` field.
    """
    out = defaultdict(dict)
    for r in group:
        ly = _int(r, "layer")
        for pos, h in _lhid_hpos(r).items():
            out[ly].setdefault(pos, h)
    return out


def report_lhid_accept(lhid_rows, c4pos_rows, idxin_rows):
    """``[LHID-ACCEPT]`` read-only acceptance gate for the MISS==HIT fix.

    Encodes the fix CONTRACT as a verdict so the eventual fix is gated by a
    check that is RED on today's logs (the ``[LHID-ACCEPT] S1 ... => FAIL`` line)
    and GREEN once the shared-position hidden state matches:

      * S1 (fix gate): across MISS vs HIT, at EVERY shared absolute position,
        EVERY dumped decoder layer's ``hpos`` hash must be identical. The count
        of shared (layer, position) groups carrying >1 distinct hash is
        ``divergent``; ``divergent==0`` -> PASS, else FAIL.
      * S2 (sanity, must always PASS): two-MISS self-control -- two MISS requests
        must have IDENTICAL ``hpos`` at every shared position -> IDENTICAL,
        otherwise MISMATCH.
      * S3 (placeholder): the curl#1..5 OUTPUT-TEXT byte-equality is MANUAL, so
        the line is printed with ``<manual>`` for the user to fill.

    Alignment-correct: compares per ABOSLUTE position via ``hpos`` (never a
    whole-tensor md5 across differing ``ntok``). MISS/HIT is split by the
    [C4POS] ``first`` label with the [IDXIN] ``sps`` / largest-``ntok``
    fallbacks (the SOURCE is printed). When ``hpos`` is absent the gate prints
    exactly what is missing and NEVER a false PASS.
    """
    print("\n== [LHID-ACCEPT] MISS-vs-HIT hidden-state acceptance gate ==")
    if not lhid_rows:
        print("  [LHID-ACCEPT] no [LHID] rows (dump gate off?)")
        print("  [LHID-ACCEPT] S1 divergent=? => CANNOT-VERIFY "
              "(missing: [LHID] rows)")
        print("  [LHID-ACCEPT] S2 two-MISS SKIP (no [LHID] rows)")
        print("  [LHID-ACCEPT] S3 curl#1..5 byte-equal: <manual>")
        print(LHID_ACCEPT_RUNBOOK)
        return
    if not any("hpos" in r for r in lhid_rows):
        print("  [LHID-ACCEPT] S1 divergent=? => CANNOT-VERIFY "
              "(missing: [LHID] `hpos=[pos:md5|..]` per-absolute-position field; "
              "re-run WINDOWED with DSV4_DUMP_MIN_POS + DSV4_DUMP_MAX_POS)")
        print("  [LHID-ACCEPT] S2 two-MISS SKIP (no `hpos` field)")
        print("  [LHID-ACCEPT] S3 curl#1..5 byte-equal: <manual>")
        print(LHID_ACCEPT_RUNBOOK)
        return

    reqs = _c4pos_requests(lhid_rows)
    source, miss_idx, hit_idx, note = _cmp_aligned_roles(
        reqs, c4pos_rows, idxin_rows)
    print(f"  [LHID-ACCEPT] split_source={source} miss_req={miss_idx} "
          f"hit_req={hit_idx}")
    print(f"  [LHID-ACCEPT] rules: {note}")
    miss_maps = [_lhid_by_layer(reqs[i]) for i in miss_idx]
    hit_maps = [_lhid_by_layer(reqs[i]) for i in hit_idx]

    # --- S1 (fix gate): MISS vs HIT identical at every shared absolute pos ---
    if not miss_idx or not hit_idx:
        print("  [LHID-ACCEPT] S1 divergent=? => CANNOT-VERIFY "
              f"(need >=1 MISS (first==0) and >=1 HIT (first>0); got "
              f"miss={len(miss_idx)} hit={len(hit_idx)})")
    else:
        divergent = 0
        compared = 0
        first_bad = None
        layers = sorted({ly for m in miss_maps for ly in m})
        for ly in layers:
            miss_pos = set()
            for m in miss_maps:
                miss_pos |= set(m.get(ly, {}))
            hit_pos = set()
            for h in hit_maps:
                hit_pos |= set(h.get(ly, {}))
            for pos in sorted(miss_pos & hit_pos):
                compared += 1
                hashes = set()
                for m in miss_maps:
                    if pos in m.get(ly, {}):
                        hashes.add(m[ly][pos])
                for h in hit_maps:
                    if pos in h.get(ly, {}):
                        hashes.add(h[ly][pos])
                if len(hashes) > 1:
                    divergent += 1
                    if first_bad is None:
                        first_bad = (ly, pos, sorted(hashes))
        verdict = "PASS" if divergent == 0 else "FAIL"
        print(f"  [LHID-ACCEPT] S1 divergent={divergent} => {verdict} "
              f"(shared layer/pos groups compared={compared})")
        if first_bad is not None:
            ly0, p0, h0 = first_bad
            print(f"  [LHID-ACCEPT] S1 first divergence layer={ly0} pos={p0} "
                  f"hashes={h0}")

    # --- S2 (sanity): two MISS requests MUST be identical ---
    if len(miss_maps) < 2:
        print(f"  [LHID-ACCEPT] S2 two-MISS SKIP (need >=2 MISS requests; got "
              f"{len(miss_maps)})")
    else:
        ctl_pos = ctl_diff = 0
        first_bad2 = None
        base = miss_maps[0]
        for ly in sorted(base):
            for pos in sorted(base[ly]):
                if not all(ly in m and pos in m[ly] for m in miss_maps[1:]):
                    continue
                ctl_pos += 1
                hashes = {m[ly][pos] for m in miss_maps}
                if len(hashes) > 1:
                    ctl_diff += 1
                    if first_bad2 is None:
                        first_bad2 = (ly, pos, sorted(hashes))
        if ctl_pos == 0:
            print("  [LHID-ACCEPT] S2 two-MISS NO-SHARED-POSITION "
                  "(cannot sanity-check)")
        elif ctl_diff == 0:
            print(f"  [LHID-ACCEPT] S2 two-MISS IDENTICAL "
                  f"(at all {ctl_pos} shared position(s))")
        else:
            print(f"  [LHID-ACCEPT] S2 two-MISS MISMATCH "
                  f"({ctl_diff}/{ctl_pos} shared position(s) differ; first "
                  f"layer={first_bad2[0]} pos={first_bad2[1]} "
                  f"hashes={first_bad2[2]})")

    # --- S3 (placeholder): curl#1..5 output-text byte-equality is MANUAL ---
    print("  [LHID-ACCEPT] S3 curl#1..5 byte-equal: <manual>")
    print(LHID_ACCEPT_RUNBOOK)


XIN_POS_RUNBOOK = (
    "[XIN-POS-RUNBOOK] per-absolute-position indexer input x hash "
    "(window around P*):\n"
    "  DSV4_DUMP_XIN=<layer> DSV4_DUMP_C4POS=1 "
    "DSV4_DUMP_MIN_POS=<win_lo> DSV4_DUMP_MAX_POS=<win_hi> "
    "<serve> ; <curl miss> ; <curl hit> ; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_cmpidx.log\n"
    "  xpos=[pos:md5|pos2:md5|..] is emitted by the backend under the "
    "DSV4_DUMP_XIN gate, in WINDOWED mode (both DSV4_DUMP_MIN_POS and "
    "DSV4_DUMP_MAX_POS set)"
)


def _xin_xpos(rec):
    """Parse ``xpos=[pos:md5|..]`` into ``{abs_pos: md5}`` (S253).

    ``xpos`` is the per-ABSOLUTE-POSITION single-row hash of the c4-indexer INPUT
    ``x`` (the ``[XIN]`` probe). The whole-tensor ``md5`` is shape-confounded
    (MISS full prefill ntok=17523 vs HIT suffix ntok=1139) and a multi-row batch
    mixes rows, so only ``xpos`` aligns a MISS and a HIT at the SAME absolute
    position. Mirrors ``_lhid_hpos`` ([LHID] ``hpos=``, S252).
    """
    return _lhid_hpos(rec, "xpos")


def _xin_by_layer(group):
    """``{layer: {pos: md5}}`` of one request's [XIN] ``xpos`` rows.

    The FIRST row carrying an absolute position wins (one x hash per position).
    """
    out = defaultdict(dict)
    for r in group:
        ly = _int(r, "layer")
        for pos, h in _xin_xpos(r).items():
            out[ly].setdefault(pos, h)
    return out


def report_xin_pos(xin_rows, c4pos_rows, idxin_rows):
    """``[XIN-POS]`` MISS-vs-HIT indexer input x hash per absolute position.

    The whole-tensor [XIN] ``md5`` is shape-confounded (MISS prefill ntok=17523
    vs HIT ntok=1139) and a multi-row batch mixes rows, so it cannot answer "did
    the indexer INPUT x change at the SAME absolute position". This report uses
    the backend per-position ``xpos=[pos:md5|..]`` field:

      * requests are split by the [C4POS] ``first`` label (``first==0`` MISS,
        ``first>0`` HIT), falling back to the [IDXIN] ``sps`` rule / largest
        ``ntok`` -- the SOURCE used is printed;
      * at every (layer, absolute position) reached by BOTH a MISS and a HIT, the
        single-row ``xpos`` hash is compared -> ``X_SAME`` / ``X_DIFF``;
      * ``X_DIFF`` localises the divergence to the indexer input at that layer
        (the origin is at/above x); ``X_SAME`` pushes it DOWNSTREAM of x;
      * a two-MISS control verifies two MISS requests agree at a shared position
        (MISS vs MISS MUST be identical) and WARNS when they do not.
    """
    print("\n== [XIN-POS] indexer input x per absolute position ==")
    if not xin_rows:
        print("  [XIN-POS] no [XIN] rows")
        return
    if not any("xpos" in r for r in xin_rows):
        print("  [XIN-POS] [XIN] has no per-position `xpos=` field -> cannot "
              "compare x at absolute positions. Re-run with the backend field "
              "(DSV4_DUMP_XIN=<layer>) and a WINDOW (set DSV4_DUMP_MIN_POS and "
              "DSV4_DUMP_MAX_POS): the backend emits `xpos=[pos:md5|..]`.")
        print(XIN_POS_RUNBOOK)
        return
    xin_reqs = _c4pos_requests(xin_rows)
    source, miss_idx, hit_idx, note = _cmp_aligned_roles(
        xin_reqs, c4pos_rows, idxin_rows)
    print(f"  [XIN-POS] split_source={source} miss_req={miss_idx} "
          f"hit_req={hit_idx}")
    print(f"  [XIN-POS] rules: {note}")
    if not miss_idx or not hit_idx:
        print("  [XIN-POS] need >=1 MISS (first==0) and >=1 HIT (first>0) with "
              "DIFFERENT labels; none")
        print(XIN_POS_RUNBOOK)
        return

    miss_maps = [_xin_by_layer(xin_reqs[i]) for i in miss_idx]
    hit_maps = [_xin_by_layer(xin_reqs[i]) for i in hit_idx]

    def _hit_pos(ly, pos):
        for hm in hit_maps:
            if ly in hm and pos in hm[ly]:
                return hm[ly][pos]
        return None

    results = []  # (layer, pos, miss_hash, hit_hash, same)
    for ly in sorted(miss_maps[0]):
        for pos in sorted(miss_maps[0][ly]):
            mhash = miss_maps[0][ly][pos]
            hhash = _hit_pos(ly, pos)
            if hhash is None:
                continue
            results.append((ly, pos, mhash, hhash, mhash == hhash))
    diffs = [r for r in results if not r[4]]
    print(f"  [XIN-POS] positions_compared={len(results)} "
          f"same={len(results) - len(diffs)} diff={len(diffs)}")
    ordered = diffs + [r for r in results if r[4]]
    for ly, pos, mhash, hhash, same in ordered[:MAX_ALIGNED_SHOWN]:
        verdict = "X_SAME" if same else "X_DIFF"
        print(f"  [XIN-POS] layer={ly} pos={pos} X_miss={mhash} X_hit={hhash} "
              f"=> {verdict}")
    if len(ordered) > MAX_ALIGNED_SHOWN:
        print(f"  ... {len(ordered) - MAX_ALIGNED_SHOWN} more positions")
    print(f"  [XIN-POS] VERDICT {'X_SAME' if not diffs else 'X_DIFF'}")
    if not diffs:
        print("  -> indexer input x is IDENTICAL MISS vs HIT at every shared "
              "absolute position; the divergence is DOWNSTREAM of x")
    else:
        print("  -> indexer input x DIFFERS MISS vs HIT at the SAME absolute "
              "position: the origin is at/above this layer's x")

    # Two-MISS control: two MISS requests must agree at every shared position.
    if len(miss_maps) >= 2:
        base = miss_maps[0]
        ctl_pos = ctl_same = ctl_diff = 0
        first_bad = None
        for ly in sorted(base):
            for pos in sorted(base[ly]):
                if not all(ly in m and pos in m[ly] for m in miss_maps[1:]):
                    continue
                ctl_pos += 1
                hashes = {m[ly][pos] for m in miss_maps}
                if len(hashes) == 1:
                    ctl_same += 1
                else:
                    ctl_diff += 1
                    if first_bad is None:
                        first_bad = (ly, pos)
        if ctl_pos == 0:
            print("  [XIN-POS] two-MISS control: no position shared by both "
                  "MISS requests (cannot sanity-check)")
        elif ctl_diff == 0:
            print(f"  [XIN-POS] two-MISS control: IDENTICAL at all {ctl_pos} "
                  f"shared position(s) (sanity OK)")
        else:
            print(f"  [XIN-POS] WARN two-MISS control: {ctl_diff}/{ctl_pos} "
                  f"shared position(s) DIFFER between two MISS requests "
                  f"(first layer={first_bad[0]} pos={first_bad[1]})")
    print(XIN_POS_RUNBOOK)


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


def _blkx_groups(rows, field="blkx", keep=None):
    """``{(layer, block): {hashes}}`` over the (optionally filtered) rows.

    The shared per-block grouping used by every ``blkx``-based verdict: a block's
    hash must be unique across requests (deterministic), so a ``(layer, block)``
    group with >1 distinct hash is a REAL divergence there.
    """
    groups = defaultdict(set)
    for r in rows:
        if keep is not None and not keep(r):
            continue
        ly = _int(r, "layer")
        for b, h in _parse_blkx(r, field).items():
            groups[(ly, b)].add(h)
    return groups


def _blkx_divergent(rows, field="blkx", keep=None):
    """``{(layer, block): sorted(distinct hashes)}`` for every divergent group.

    A group with >1 distinct hash is the SAME divergence ``report_blkx`` prints as
    ``divergent=`` (identical criterion).  Callers that must agree with that
    section REUSE this instead of re-keying the pool a second way; a page found
    divergent here (e.g. page 136) can then never be missed downstream.
    """
    return {(ly, b): sorted(s)
            for (ly, b), s in _blkx_groups(rows, field, keep).items()
            if len(s) > 1}


def report_blkx(rows, label, field="blkx", keep=None):
    """Per-c128-block (``position // 128``) hit==miss verdict.

    A block's hash must be unique across all requests (deterministic), so a
    ``(layer, block)`` group with >1 distinct hash is a REAL divergence there.
    This is NOT shape-confounded (a block = the same absolute positions on both
    the full-prefill miss and the suffix hit).  Locates the FIRST divergent
    block and the layers diverging there (=> the first divergent layer).
    ``keep`` optionally filters rows (e.g. prefill-only to drop decode noise).
    """
    groups = _blkx_groups(rows, field, keep)
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


def _idxk_evolving_pages(rows):
    """``{(layer, page)}`` whose stored hash is NOT unique WITHIN one request.

    A c4 page is written row-by-row; the PREFILL stops a few rows short of the
    LAST page it touches (the prefill-absent tail) and those rows commit one per
    decode step.  The [IDXK] probe hashes the WHOLE page, so a request that
    samples the page while it is still being assembled sees a DIFFERENT md5 than
    a request that samples it AFTER it is complete (``npu_scatter_nd_update_``
    never clears untouched rows -> last-writer-wins).  A page whose hash changes
    across the SAME request's own steps is therefore an EVOLVING (partially
    written) page, and any MISS-vs-HIT difference confined to it is a
    sampling-TIMING artifact, not a compressor-output difference.
    """
    per_req = defaultdict(lambda: defaultdict(set))
    for r in rows:
        req = _int(r, "_req", -1)
        ly = _int(r, "layer")
        for b, h in _parse_blkx(r, "blkx").items():
            per_req[(ly, b)][req].add(h)
    return {key for key, by_req in per_req.items()
            if any(len(hs) > 1 for hs in by_req.values())}


def _idxk_evolve_table(rows):
    """``{(layer, page): [(req, lastpos, hash), ...]}`` in emission order."""
    tb = defaultdict(list)
    for r in rows:
        req = _int(r, "_req", -1)
        lp = _int(r, "lastpos", -1)
        ly = _int(r, "layer")
        for b, h in _parse_blkx(r, "blkx").items():
            tb[(ly, b)].append((req, lp, h))
    return tb


def _idxk_aligned_groups(rows, evolving):
    """Prefill ``(layer, page) -> {hashes}`` EXCLUDING the evolving tail pages."""
    groups = defaultdict(set)
    for r in rows:
        if not _is_prefill(r):
            continue
        ly = _int(r, "layer")
        for b, h in _parse_blkx(r, "blkx").items():
            if (ly, b) in evolving:
                continue
            groups[(ly, b)].add(h)
    return groups


def report_idxk_aligned(rows):
    """ALIGNED [IDXK] verdict: drop the evolving-tail sampling artifact.

    The raw ``[IDXK]`` per-page hash is only comparable when the page has the
    SAME assembly stage on both sides.  The prefill of a cache-MISS samples the
    LAST page it touches while that page is still PARTIALLY written (the
    prefill-absent tail rows commit later, one per decode step), whereas a cache
    HIT -- sampled after an earlier request's decode completed the page -- sees
    the FULL page.  Comparing whole-page md5s across those two stages is a
    sampling artifact.

    This report:
      1. lists the EVOLVING pages (hash changes within a single request);
      2. for each, prints every request's first (prefill-sampled) and last
         (fully-assembled) hash + whether ALL requests converge to one final
         hash (the direct MISS-vs-HIT same-stage comparison);
      3. recomputes the prefill per-page divergence EXCLUDING the evolving pages
         -> the aligned verdict: index-K CLEAN (artifact) or a REAL difference.
    """
    if not rows:
        return
    evolving = _idxk_evolving_pages(rows)
    raw_div = _blkx_divergent(rows, "blkx", keep=_is_prefill)
    print("\n== [IDXK-ALIGNED] prefill-written pages only (drop evolving tail) ==")
    if not evolving:
        print("  [IDXK-ALIGNED] no evolving page (every page stable within a "
              "request) -> aligned verdict == raw verdict")
        return
    print(f"  [IDXK-ALIGNED] evolving (partially-written) pages: "
          f"{sorted(evolving)}")
    tb = _idxk_evolve_table(rows)
    _still_diff = set()  # evolving pages whose FULLY-ASSEMBLED hash still differs
    for key in sorted(evolving):
        by_req = {}
        for req, lp, h in tb[key]:
            slot = by_req.setdefault(
                req, {"first_lp": lp, "first": h, "last_lp": lp, "last": h})
            if lp >= slot["last_lp"]:
                slot["last_lp"], slot["last"] = lp, h
            if lp <= slot["first_lp"]:
                slot["first_lp"], slot["first"] = lp, h
        finals = sorted({s["last"] for s in by_req.values()})
        print(f"  [IDXK-ALIGNED] layer={key[0]} page={key[1]}: "
              f"prefill-sampled vs fully-assembled per request")
        for req in sorted(by_req):
            s = by_req[req]
            role = "HIT " if s["first"] == s["last"] else "MISS"
            print(f"     req{req} {role} first(lp={s['first_lp']})="
                  f"{s['first']} last(lp={s['last_lp']})={s['last']}")
        conv = "CONVERGED" if len(finals) == 1 else "STILL-DIFF"
        if len(finals) > 1:
            _still_diff.add(key)
        print(f"     fully-assembled hashes={finals} => {conv}")
    aligned_groups = _idxk_aligned_groups(rows, evolving)
    aligned_div = sorted((ly, b) for (ly, b), s in aligned_groups.items()
                         if len(s) > 1)
    artifact = sorted(k for k in raw_div if k in evolving and k not in _still_diff)
    # A page whose FULLY-ASSEMBLED hash still differs across requests is a REAL
    # divergence, NOT a sampling artifact (sampling only explains a first-vs-last
    # gap WITHIN one request that CONVERGES).  Previously such pages were dropped
    # from `real` and the verdict wrongly printed CLEAN.
    real = sorted(set(k for k in raw_div if k not in evolving) | _still_diff)
    if not real:
        print(f"  [IDXK-ALIGNED] VERDICT: every raw divergent page "
              f"{sorted(raw_div)} is an EVOLVING tail page {artifact} "
              f"-> MISS==HIT on all prefill-WRITTEN pages => index-K CLEAN "
              f"(raw divergence is a sampling-time artifact)")
    else:
        _sd = sorted(_still_diff)
        print(f"  [IDXK-ALIGNED] VERDICT: artifact page(s)={artifact}; REAL "
              f"divergence remains on {real}"
              + (f" (fully-assembled STILL differs on {_sd})" if _sd else "")
              + " => index-K NOT clean")
    if aligned_div:
        print(f"  [IDXK-ALIGNED] non-evolving divergent groups: {aligned_div}")


def report_cmpidx_div(rows, cmp_reqs, miss_i):
    """One best-effort ``[CMPIDX-DIV]`` line after the [CMPIDX] report.

    Uses the existing per-block hashing (``_parse_blkx``) to list the blocks whose
    hash is not unique across requests (=> divergent), plus the MISS/HIT prefill
    ``ntok`` from the [CMPIDX] request split.  Skipped when there is no split.
    """
    if not rows or not cmp_reqs or miss_i >= len(cmp_reqs):
        return
    miss_ntok = cmp_reqs[miss_i]["ntok0"]
    hit_ntoks = [rq["ntok0"] for i, rq in enumerate(cmp_reqs)
                 if i != miss_i and rq["ntok0"] is not None]
    hit_ntok = hit_ntoks[0] if hit_ntoks else None
    groups = defaultdict(set)
    for r in rows:
        if not _is_prefill(r):
            continue
        ly = _int(r, "layer")
        for b, h in _parse_blkx(r, "blkx").items():
            groups[(ly, b)].add(h)
    div = sorted({b for (ly, b), s in groups.items() if len(s) > 1})
    if not div:
        blocks_txt = "none"
    elif len(div) <= 20:
        blocks_txt = ",".join(str(b) for b in div)
    else:
        blocks_txt = str(len(div))
    first_txt = str(div[0]) if div else "none"
    print(f"[CMPIDX-DIV] miss={miss_ntok} hit={hit_ntok} "
          f"divergent_blocks={blocks_txt} first={first_txt}")


CMPIDX_ALIGNED_RUNBOOK = (
    "[CMPIDX-ALIGNED-RUNBOOK] per-absolute-position indexer top-k SET "
    "(window around P*):\n"
    "  DSV4_DUMP_CMPIDX=<layer> DSV4_DUMP_C4POS=1 "
    "DSV4_DUMP_MIN_POS=<win_lo> DSV4_DUMP_MAX_POS=<win_hi> "
    "DSV4_DUMP_CMPIDX_ROWS=8 <serve> ; <curl miss> ; <curl hit> ; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_cmpidx.log\n"
    "  kidx=[pos:i,i,..|pos2:i,..] is emitted by the backend under the same "
    "DSV4_DUMP_CMPIDX gate"
)

CMPIDX_ORDERED_RUNBOOK = (
    "[CMPIDX-ORDERED-ALIGNED-RUNBOOK] per-absolute-position indexer top-k "
    "SEQUENCE (window around P*):\n"
    "  DSV4_DUMP_CMPIDX=<layer> DSV4_DUMP_C4POS=1 "
    "DSV4_DUMP_MIN_POS=<win_lo> DSV4_DUMP_MAX_POS=<win_hi> "
    "DSV4_DUMP_CMPIDX_ROWS=0 <serve> ; <curl miss> ; <curl hit> ; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_cmpidx.log\n"
    "  kord=[pos:i,i,..|pos2:i,..] is emitted by the backend under the same "
    "DSV4_DUMP_CMPIDX gate (STORED order, NO sort/dedup)"
)


def _cmpidx_kord(rec):
    """Parse ``kord=[pos:i,i,..|pos2:i,..]`` into ``{pos: [i, i, ..]}``.

    ``kord`` is the per-ABSOLUTE-POSITION indexer top-k key list IN STORED ORDER
    (NO sorting, NO dedup; pads ``<0`` dropped) -- DISTINCT from ``kidx`` which
    is sorted-unique.  The ORDER is what the sparse kernel consumes sequentially
    (``GetRealCmpS2Idx``/``ProcessSparseKv``), so a permutation that ``kidx`` (a
    SET) cannot see shows up here.  Empty when the field is absent (older probe).
    """
    v = rec.get("kord")
    if not v:
        return {}
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    out = {}
    for entry in s.split("|"):
        entry = entry.strip()
        if ":" not in entry:
            continue
        p, _, lst = entry.partition(":")
        try:
            pos = int(p.strip())
        except ValueError:
            continue
        seq = []
        for tok in lst.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                seq.append(int(tok))
            except ValueError:
                continue
        out[pos] = seq
    return out


def _cmpidx_by_layer_order(group):
    """``{layer: {pos: (seq, mode)}}`` of one request's [CMPIDX] ``kord`` rows.

    ``seq`` is the STORED-ORDER top-k list; a row's ``mode`` (``2`` ==
    ``ForwardMode.DECODE``) labels its positions ``decode`` vs ``prefill``; the
    FIRST row carrying a position wins.
    """
    out = defaultdict(dict)
    for r in group:
        mode = "decode" if r.get("mode") == "2" else "prefill"
        ly = _int(r, "layer")
        for pos, seq in _cmpidx_kord(r).items():
            out[ly].setdefault(pos, (seq, mode))
    return out


def _cmpidx_seq_hash(seq):
    """Short md5 of an ORDERED top-k index list (``empty`` for an empty list)."""
    if not seq:
        return "empty"
    return hashlib.md5(
        ",".join(str(v) for v in seq).encode("ascii")).hexdigest()[:12]


def _seq_first_diff(a, b):
    """``(offset, a_val, b_val)`` of the first position where two seqs differ.

    Returns ``None`` when the sequences are identical (same length, every element
    equal).  On a length mismatch the first out-of-range offset is returned with
    the missing side's value as ``None``.
    """
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return (i, a[i], b[i])
    if len(a) != len(b):
        i = min(len(a), len(b))
        return (i, a[i] if i < len(a) else None, b[i] if i < len(b) else None)
    return None


def _cmpidx_kidx(rec):
    """Parse ``kidx=[pos:i,i,..|pos2:i,..]`` into ``{pos: sorted unique idxs}``.

    ``kidx`` is the per-ABSOLUTE-POSITION indexer top-k key list the backend
    emits (already sorted unique; pads ``<0`` dropped).  Empty when the field is
    absent (older probe), so the caller can say what is missing.
    """
    v = rec.get("kidx")
    if not v:
        return {}
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    out = {}
    for entry in s.split("|"):
        entry = entry.strip()
        if ":" not in entry:
            continue
        p, _, lst = entry.partition(":")
        try:
            pos = int(p.strip())
        except ValueError:
            continue
        idxs = []
        for tok in lst.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idxs.append(int(tok))
            except ValueError:
                continue
        out[pos] = sorted(set(idxs))
    return out


def _cmpidx_by_layer(group):
    """``{layer: {pos: (sorted_idxs, mode)}}`` of one request's [CMPIDX] rows.

    A row's ``mode`` (``2`` == ``ForwardMode.DECODE``) labels its positions
    ``decode`` vs ``prefill``; the FIRST row carrying a position wins.
    """
    out = defaultdict(dict)
    for r in group:
        mode = "decode" if r.get("mode") == "2" else "prefill"
        ly = _int(r, "layer")
        for pos, idxs in _cmpidx_kidx(r).items():
            out[ly].setdefault(pos, (idxs, mode))
    return out


def _cmpidx_ntok0(group):
    """Largest ``ntok`` of one request's [CMPIDX] rows (prefill length, or -1)."""
    return max((_int(r, "ntok") for r in group), default=-1)


def _cmp_aligned_roles(cmp_reqs, c4pos_rows, idxin_rows):
    """``(source, miss_idx, hit_idx, note)`` for the ALIGNED [CMPIDX] compare.

    PREFERS the [C4POS] ``first`` label (``first==0`` -> MISS full prefill,
    ``first>0`` -> HIT suffix prefill) mapped onto the [CMPIDX] request indices
    via the SAME request split.  Falls back to the [IDXIN] ``sps`` rule (aligned
    by the ``/v1/chat/completions`` boundary), then to largest-prefill-``ntok``.
    MISS and HIT indices are DISJOINT by construction (a MISS request is never
    paired with another MISS); equal-label pairs are handled only by the
    two-MISS control inside ``report_cmpidx_aligned``.
    """
    c4_reqs = _c4pos_requests(c4pos_rows) if c4pos_rows else []
    if c4_reqs and len(c4_reqs) == len(cmp_reqs):
        miss, hit = [], []
        for i, g in enumerate(c4_reqs):
            m = _c4pos_prefill_maps(g)
            if not m:
                continue
            first = min(f for f, _r, _p in m.values())
            (miss if first == 0 else hit).append(i)
        if miss and hit:
            return ("c4pos", miss, hit,
                    f"c4pos first==0 -> MISS {miss}, first>0 -> HIT {hit}")
    if idxin_rows and _idxin_split_source(idxin_rows) == "chat/completions":
        ireqs = _idxin_requests(idxin_rows)
        if len(ireqs) == len(cmp_reqs):
            _m, miss_set, hit_idx, source, note = _idxin_roles(ireqs)
            if miss_set and hit_idx:
                return (f"idxin:{source}", list(miss_set), list(hit_idx), note)
    if len(cmp_reqs) >= 2:
        ntoks = [_cmpidx_ntok0(g) for g in cmp_reqs]
        mx = max(ntoks)
        # MISS = ALL requests at the largest prefill ntok (the harness sends N
        # identical full-prefill MISS followed by 1 suffix HIT, so the old
        # single-max rule collapsed req0..req{N-1} into one MISS and mislabeled
        # req1..req{N-1} as HIT). HIT = every request below the max.
        miss = [j for j in range(len(cmp_reqs)) if cmp_reqs[j] and ntoks[j] == mx]
        hit = [j for j in range(len(cmp_reqs)) if cmp_reqs[j] and ntoks[j] != mx]
        if miss and hit:
            return ("largest-ntok", miss, hit,
                    f"no [C4POS]/[IDXIN]; MISS=ALL largest-ntok prefill "
                    f"(ntok={mx}) {miss}")
        # `ntok` cannot separate the requests (all equal, or missing/unparsed):
        # fall back to the harness convention -- N full-prefill MISS then ONE
        # suffix HIT => the LAST request is the HIT, earlier ones are MISS.
        n = len(cmp_reqs)
        miss, hit = list(range(n - 1)), [n - 1]
        if miss and hit:
            return ("harness-last-hit", miss, hit,
                    "ntok cannot separate requests; harness convention: "
                    f"LAST=HIT(req{hit[0]}), earlier=MISS")
    return "none", [], [], "no usable MISS/HIT split"


def _cmpidx_set_hash(idxs):
    """Short md5 of a sorted top-k index list (``empty`` for an empty list)."""
    if not idxs:
        return "empty"
    return hashlib.md5(
        ",".join(str(v) for v in idxs).encode("ascii")).hexdigest()[:12]


def report_cmpidx_aligned(cmp_rows, c4pos_rows, idxin_rows):
    """``[CMPIDX-ALIGNED]`` MISS-vs-HIT indexer top-k SET per absolute position.

    The whole-tensor [CMPIDX] md5 is only comparable at equal ``ntok`` and the
    per-block ``blkx`` hashes the sampled row, so neither answers "did the top-k
    SET change at the SAME absolute position / stage".  This report uses the
    backend per-position ``kidx=[pos:i,i,..|..]`` field:

      * requests are split by the [C4POS] ``first`` label (``first==0`` MISS,
        ``first>0`` HIT), falling back to the [IDXIN] ``sps`` rule / largest
        ``ntok`` -- the SOURCE used is printed;
      * at every (layer, absolute position) reached by BOTH a MISS and a HIT,
        the MISS top-k SET is compared with the HIT top-k SET;
      * ``TOPK_DIFF`` reports the symmetric-difference offenders (the few keys
        that swapped) -- the predicted ``==kth`` tie-break signature;
      * a two-MISS control verifies two MISS requests agree at a shared position
        (MISS vs MISS MUST be identical) and WARNS when they do not.
    """
    print("\n== [CMPIDX-ALIGNED] indexer top-k SET per absolute position ==")
    if not cmp_rows:
        print("  [CMPIDX-ALIGNED] no [CMPIDX] rows")
        return
    if not any("kidx" in r for r in cmp_rows):
        print("  [CMPIDX-ALIGNED] [CMPIDX] has no per-position `kidx=` field -> "
              "cannot compare top-k SETs. Re-run with the backend field "
              "(DSV4_DUMP_CMPIDX=<layer>): it emits "
              "`kidx=[pos:i,i,..|..]` per absolute position.")
        print(CMPIDX_ALIGNED_RUNBOOK)
        return
    cmp_reqs = _c4pos_requests(cmp_rows)
    source, miss_idx, hit_idx, note = _cmp_aligned_roles(
        cmp_reqs, c4pos_rows, idxin_rows)
    print(f"  [CMPIDX-ALIGNED] split_source={source} miss_req={miss_idx} "
          f"hit_req={hit_idx}")
    print(f"  [CMPIDX-ALIGNED] rules: {note}")
    if not miss_idx or not hit_idx:
        print("  [CMPIDX-ALIGNED] need >=1 MISS (first==0) and >=1 HIT "
              "(first>0) with DIFFERENT labels; none")
        print(CMPIDX_ALIGNED_RUNBOOK)
        return

    miss_maps = [_cmpidx_by_layer(cmp_reqs[i]) for i in miss_idx]
    hit_maps = [_cmpidx_by_layer(cmp_reqs[i]) for i in hit_idx]

    def _hit_pos(ly, pos):
        for hm in hit_maps:
            if ly in hm and pos in hm[ly]:
                return hm[ly][pos]
        return None

    results = []  # (layer, pos, mode, miss_set, hit_set, delta, same)
    for ly in sorted(miss_maps[0]):
        for pos in sorted(miss_maps[0][ly]):
            mset, mmode = miss_maps[0][ly][pos]
            h = _hit_pos(ly, pos)
            if h is None:
                continue
            ms, hs = sorted(set(mset)), sorted(set(h[0]))
            delta = sorted(set(ms) ^ set(hs))
            results.append((ly, pos, mmode, ms, hs, delta, ms == hs))
    diffs = [r for r in results if not r[6]]
    print(f"  [CMPIDX-ALIGNED] positions_compared={len(results)} "
          f"same={len(results) - len(diffs)} diff={len(diffs)}")
    ordered = diffs + [r for r in results if r[6]]
    for ly, pos, mmode, ms, hs, delta, same in ordered[:MAX_ALIGNED_SHOWN]:
        verdict = "TOPK_SAME" if same else "TOPK_DIFF"
        print(f"  [CMPIDX-ALIGNED] layer={ly} pos={pos} mode={mmode} "
              f"SET_miss={_cmpidx_set_hash(ms)}({len(ms)}) "
              f"SET_hit={_cmpidx_set_hash(hs)}({len(hs)}) "
              f"delta={len(delta)} => {verdict}")
        if not same:
            print(f"     offenders(symdiff)={delta[:MAX_OFFENDER_SHOWN]}"
                  + (" ..." if len(delta) > MAX_OFFENDER_SHOWN else ""))
    if len(ordered) > MAX_ALIGNED_SHOWN:
        print(f"  ... {len(ordered) - MAX_ALIGNED_SHOWN} more positions")
    print(f"  [CMPIDX-ALIGNED] VERDICT "
          f"{'TOPK_SAME' if not diffs else 'TOPK_DIFF'}")

    # Two-MISS control: two MISS requests must agree at every shared position.
    if len(miss_maps) >= 2:
        base = miss_maps[0]
        ctl_pos = ctl_same = ctl_diff = 0
        first_bad = None
        for ly in sorted(base):
            for pos in sorted(base[ly]):
                if not all(ly in m and pos in m[ly] for m in miss_maps[1:]):
                    continue
                ctl_pos += 1
                sets = {tuple(sorted(m[ly][pos][0])) for m in miss_maps}
                if len(sets) == 1:
                    ctl_same += 1
                else:
                    ctl_diff += 1
                    if first_bad is None:
                        first_bad = (ly, pos)
        if ctl_pos == 0:
            print("  [CMPIDX-ALIGNED] two-MISS control: no position shared by "
                  "both MISS requests (cannot sanity-check)")
        elif ctl_diff == 0:
            print(f"  [CMPIDX-ALIGNED] two-MISS control: IDENTICAL at all "
                  f"{ctl_pos} shared position(s) (sanity OK)")
        else:
            print(f"  [CMPIDX-ALIGNED] WARN two-MISS control: {ctl_diff}/"
                  f"{ctl_pos} shared position(s) DIFFER between two MISS "
                  f"requests (first layer={first_bad[0]} pos={first_bad[1]})")
    print(CMPIDX_ALIGNED_RUNBOOK)


def report_cmpidx_ordered(cmp_rows, c4pos_rows, idxin_rows):
    """``[CMPIDX-ORDERED-ALIGNED]`` MISS-vs-HIT indexer top-k SEQUENCE per pos.

    The sparse-attn kernel ``kv_quant_sparse_attn_sharedkv`` consumes
    ``cmp_sparse_indices`` as an ORDERED list (a sequential gather in
    ``GetRealCmpS2Idx``/``ProcessSparseKv``), while the indexer's output ORDER
    after the cross-core (LD) merge is workspace-order (s1-dependent), NOT
    canonical.  The existing ``[CMPIDX-ALIGNED]`` compares ``sorted({...})`` -- a
    SET -- so it CANNOT see an order permutation.  This report compares the
    SEQUENCE (the backend ``kord=[pos:i,i,..|..]`` field, STORED order) at every
    (layer, absolute position) reached by BOTH a MISS and a HIT:

      * requests are split by the [C4POS] ``first`` label (``first==0`` MISS,
        ``first>0`` HIT), falling back to the [IDXIN] ``sps`` rule / largest
        ``ntok`` -- the SOURCE used is printed;
      * ``ORDER_SAME`` -> the stored order matches MISS-vs-HIT; escalate to the
        metadata/attention probes (the mismatch is not in the top-k order);
      * ``ORDER_DIFF`` while ``SET_miss == SET_hit`` -> confirms the
        ordered-consumption + s1-dependent cross-core workspace-order seed (R1):
        the SAME key SET is emitted in a different order;
      * a two-MISS control verifies two MISS requests are SEQUENCE-identical at a
        shared position (identical full prefills share s1) and WARNS otherwise.
    """
    print("\n== [CMPIDX-ORDERED-ALIGNED] indexer top-k SEQUENCE per position ==")
    if not cmp_rows:
        print("  [CMPIDX-ORDERED-ALIGNED] no [CMPIDX] rows")
        return
    if not any("kord" in r for r in cmp_rows):
        print("  [CMPIDX-ORDERED-ALIGNED] [CMPIDX] has no per-position `kord=` "
              "field -> cannot compare top-k SEQUENCEs. Re-run with the backend "
              "field (DSV4_DUMP_CMPIDX=<layer>): it emits "
              "`kord=[pos:i,i,..|..]` per absolute position (STORED order).")
        print(CMPIDX_ORDERED_RUNBOOK)
        return
    cmp_reqs = _c4pos_requests(cmp_rows)
    source, miss_idx, hit_idx, note = _cmp_aligned_roles(
        cmp_reqs, c4pos_rows, idxin_rows)
    print(f"  [CMPIDX-ORDERED-ALIGNED] split_source={source} "
          f"miss_req={miss_idx} hit_req={hit_idx}")
    print(f"  [CMPIDX-ORDERED-ALIGNED] rules: {note}")
    if not miss_idx or not hit_idx:
        print("  [CMPIDX-ORDERED-ALIGNED] need >=1 MISS (first==0) and >=1 HIT "
              "(first>0) with DIFFERENT labels; none")
        print(CMPIDX_ORDERED_RUNBOOK)
        return

    miss_maps = [_cmpidx_by_layer_order(cmp_reqs[i]) for i in miss_idx]
    hit_maps = [_cmpidx_by_layer_order(cmp_reqs[i]) for i in hit_idx]

    def _hit_pos(ly, pos):
        for hm in hit_maps:
            if ly in hm and pos in hm[ly]:
                return hm[ly][pos]
        return None

    results = []  # (layer, pos, mode, mseq, hseq, diff_off, set_same, same)
    for ly in sorted(miss_maps[0]):
        for pos in sorted(miss_maps[0][ly]):
            mseq, mmode = miss_maps[0][ly][pos]
            h = _hit_pos(ly, pos)
            if h is None:
                continue
            hseq = h[0]
            same = mseq == hseq
            set_same = set(mseq) == set(hseq)
            results.append((ly, pos, mmode, mseq, hseq,
                            _seq_first_diff(mseq, hseq), set_same, same))
    diffs = [r for r in results if not r[7]]
    set_same_all = all(r[6] for r in results) if results else None
    print(f"  [CMPIDX-ORDERED-ALIGNED] positions_compared={len(results)} "
          f"same={len(results) - len(diffs)} diff={len(diffs)}")
    ordered = diffs + [r for r in results if r[7]]
    for ly, pos, mmode, mseq, hseq, doff, set_same, same in (
            ordered[:MAX_ALIGNED_SHOWN]):
        verdict = "ORDER_SAME" if same else "ORDER_DIFF"
        print(f"  [CMPIDX-ORDERED-ALIGNED] layer={ly} pos={pos} mode={mmode} "
              f"SEQ_miss={_cmpidx_seq_hash(mseq)}({len(mseq)}) "
              f"SEQ_hit={_cmpidx_seq_hash(hseq)}({len(hseq)}) "
              f"SET_{'same' if set_same else 'diff'} => {verdict}")
        if not same and doff is not None:
            print(f"     first_diff offset={doff[0]} miss={doff[1]} "
                  f"hit={doff[2]}  swapped_miss={mseq} -> hit={hseq}")
    if len(ordered) > MAX_ALIGNED_SHOWN:
        print(f"  ... {len(ordered) - MAX_ALIGNED_SHOWN} more positions")
    print(f"  [CMPIDX-ORDERED-ALIGNED] VERDICT "
          f"{'ORDER_SAME' if not diffs else 'ORDER_DIFF'}")

    # Two-MISS control: two MISS requests must be SEQUENCE-identical at every
    # shared position (the indexer is deterministic; identical full prefills
    # share s1, so the workspace order must match too).
    if len(miss_maps) >= 2:
        base = miss_maps[0]
        ctl_pos = ctl_same = ctl_diff = 0
        first_bad = None
        for ly in sorted(base):
            for pos in sorted(base[ly]):
                if not all(ly in m and pos in m[ly] for m in miss_maps[1:]):
                    continue
                ctl_pos += 1
                seqs = {tuple(m[ly][pos][0]) for m in miss_maps}
                if len(seqs) == 1:
                    ctl_same += 1
                else:
                    ctl_diff += 1
                    if first_bad is None:
                        first_bad = (ly, pos)
        if ctl_pos == 0:
            print("  [CMPIDX-ORDERED-ALIGNED] two-MISS control: no position "
                  "shared by both MISS requests (cannot sanity-check)")
        elif ctl_diff == 0:
            print(f"  [CMPIDX-ORDERED-ALIGNED] two-MISS control: SEQUENCE-"
                  f"IDENTICAL at all {ctl_pos} shared position(s) (sanity OK)")
        else:
            print(f"  [CMPIDX-ORDERED-ALIGNED] WARN two-MISS control: {ctl_diff}/"
                  f"{ctl_pos} shared position(s) DIFFER between two MISS "
                  f"requests (first layer={first_bad[0]} pos={first_bad[1]})")

    # Verdict table: ORDER_DIFF with SET_miss == SET_hit confirms the
    # ordered-consumption + s1-dependent cross-core workspace-order seed (R1).
    if not diffs:
        reading = ("ORDER_SAME => the top-k ORDER matches MISS-vs-HIT; escalate "
                   "to the metadata/attention probes")
    elif set_same_all:
        reading = ("ORDER_DIFF + SET_miss==SET_hit => confirms the ordered-"
                   "consumption + s1-dependent cross-core workspace-order seed "
                   "(R1)")
    else:
        reading = ("ORDER_DIFF with a SET difference too => the top-k SETS "
                   "themselves differ; fix the selection before the order")
    print(f"  [CMPIDX-ORDERED-ALIGNED] READING: {reading}")
    print(CMPIDX_ORDERED_RUNBOOK)


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


def _seq_max(v):
    """Max element of a ``(min,max,numel)`` seq-length tuple, or -1."""
    m = SEQ_TUP.search(str(v or ""))
    return int(m.group(2)) if m else -1


def _idxin_stream_sig(r):
    """Boundary signature of one [IDXIN] row: ``(mode, slq, slk)``."""
    return (r.get("mode", "?"), r.get("slq", "?"), r.get("slk", "?"))


def _idxin_requests_by_stream(rows):
    """No-[CMPIDX] fallback split of the [IDXIN] stream into requests.

    Each request emits its own [IDXIN] lines with a CONSTANT ``(mode, slq,
    slk)`` signature, so a new request starts whenever that signature changes
    (the ``(mode,slq,slk)`` stream resets/repeats).  The layer index also
    RESETS (drops) on a new request, which additionally separates two adjacent
    requests that happen to share the signature (e.g. repeated identical HIT
    curls).  MISS is later picked as the request with the LARGEST prefill
    ``slq`` (see ``_idxin_miss_index``) -- no [CMPIDX] needed.
    """
    reqs = []
    prev_sig = None
    prev_ly = None
    for r in rows:
        sig = _idxin_stream_sig(r)
        ly = _int(r, "layer")
        if (not reqs or sig != prev_sig
                or (prev_ly is not None and 0 <= ly < prev_ly)):
            reqs.append([])
        reqs[-1].append(r)
        prev_sig = sig
        prev_ly = ly
    return reqs


def _idxin_chat_split(rows):
    """Group [IDXIN] rows by the ``/v1/chat/completions`` boundary before them.

    ``parse`` stamps every row with ``_req`` = the number of chat/completions
    boundary lines seen so far, so the request opened by the k-th boundary owns
    the rows with ``_req == k`` (the user's "5th request" == index 5).  EMPTY
    buckets are kept, so the returned list index EQUALS the boundary ordinal and
    ``req{i}`` output matches the curl order.  Returns ``(groups, n_boundaries)``.
    """
    by = defaultdict(list)
    top = 0
    for r in rows:
        k = _int(r, "_req", 0)
        by[k].append(r)
        top = max(top, k)
    return [by.get(i, []) for i in range(top + 1)], top


def _idxin_split_source(rows):
    """Name the segmenter ``_idxin_requests`` uses: chat/completions|lastpos|stream."""
    if any(_int(r, "_req", 0) > 0 for r in rows):
        return "chat/completions"
    if any(_int(r, "lastpos") >= 0 for r in rows):
        return "lastpos"
    return "stream"


def _idxin_split_info(rows):
    """``(source, split, n_req, miss_req, hit_req)`` for the ``[IDXIN-SPLIT]`` line.

    ``source`` is the MISS/HIT rule (``sps`` or ``largest-slq``); ``split`` the
    segmenter (``chat/completions``/``lastpos``/``stream``); ``n_req`` the request
    count; ``miss_req``/``hit_req`` the classified request indices.
    """
    split = _idxin_split_source(rows)
    reqs = _idxin_requests(rows)
    n_req = (max((_int(r, "_req", 0) for r in rows), default=0)
             if split == "chat/completions" else len(reqs))
    if len(reqs) < 2:
        return "none", split, n_req, [], []
    _miss_i, miss_set, hit_idx, source, _note = _idxin_roles(reqs)
    return source, split, n_req, miss_set, hit_idx


def report_idxin_split(rows):
    """One ``[IDXIN-SPLIT]`` banner: MISS/HIT rule, split, and req indices."""
    if not rows:
        return
    source, split, n_req, miss_set, hit_idx = _idxin_split_info(rows)
    print(f"  [IDXIN-SPLIT] source={source} split={split} requests={n_req} "
          f"miss_req={miss_set} hit_req={hit_idx}")


def _idxin_requests(rows):
    """Split [IDXIN] rows into requests.

    Prefers, in order:

      1. ``/v1/chat/completions`` HTTP boundaries (``_req``, stamped by
         ``parse``) -- the strongest signal: the user's real request boundaries;
      2. the order-based ``lastpos`` split (logs WITH [CMPIDX]);
      3. the self-contained ``(mode,slq,slk)`` stream split.
    """
    if any(_int(r, "_req", 0) > 0 for r in rows):
        groups, _n = _idxin_chat_split(rows)
        return groups
    if any(_int(r, "lastpos") >= 0 for r in rows):
        reqs = []
        last_lp = None
        for r in rows:
            lp = _int(r, "lastpos")
            if not reqs or (last_lp is not None and lp < last_lp):
                reqs.append([])
            reqs[-1].append(r)
            last_lp = lp
        return reqs
    return _idxin_requests_by_stream(rows)


def _idxin_miss_index(reqs):
    """Index of the cache-MISS request = largest prefill ``slq`` (best effort)."""
    scores = [max((_seq_max(r.get("slq")) for r in g), default=-1) for g in reqs]
    if not scores or max(scores) <= 0:
        return 0
    return scores.index(max(scores))


def _idxin_req_len(rows):
    """``(max prefill slq, max slk)`` of one request's [IDXIN] rows (or -1)."""
    slq = max((_seq_max(r.get("slq")) for r in rows), default=-1)
    slk = max((_seq_max(r.get("slk")) for r in rows), default=-1)
    return slq, slk


def _idxin_same_len_hits(reqs, miss_i):
    """HIT request indices whose prompt length matches the MISS (slq/slk).

    The MISS is the largest-prefill request, so a HIT that reports the SAME
    ``slq`` (prompt length) -- or the same ``slk`` (context length, the suffix
    continuation) -- is the cache-hit of the SAME sequence, i.e. the only fair
    knorm comparison.  Picking the FIRST HITs instead pairs the MISS with
    unrelated short prompts and yields a false ``CONTENT/WINDOW``.

    Returns ``(hit_indices, miss_slq, miss_slk)``; ``hit_indices`` is empty when
    no HIT shares the MISS length (caller then falls back to all HITs).
    """
    miss_slq, miss_slk = _idxin_req_len(reqs[miss_i])
    if miss_slq <= 0 and miss_slk <= 0:
        return [], miss_slq, miss_slk
    hits = []
    for j, g in enumerate(reqs):
        if j == miss_i:
            continue
        h_slq, h_slk = _idxin_req_len(g)
        if h_slq <= 0 and h_slk <= 0:
            continue
        if (h_slq == miss_slq or h_slk == miss_slq
                or h_slq == miss_slk or h_slk == miss_slk):
            hits.append(j)
    return hits, miss_slq, miss_slk


def _idxin_req_sps(group):
    """Start position (min ``sps``) of one request's rows, or None.

    The full prefill of the MISS starts at position 0, so its ``sps`` is 0 even
    when the request later decodes to higher positions; a suffix HIT never sees
    position 0.
    """
    vals = [s for s in (_rope_sps(r.get("ropephase")) for r in group)
            if s is not None]
    return min(vals) if vals else None


def _idxin_sps_roles(reqs):
    """``(miss_reqs, hit_reqs)`` by start position: ``sps==0`` MISS, ``sps>0`` HIT.

    Requests without a parseable ``sps`` are left unclassified (neither list).
    """
    miss, hit = [], []
    for i, g in enumerate(reqs):
        s = _idxin_req_sps(g)
        if s is None:
            continue
        (miss if s == 0 else hit).append(i)
    return miss, hit


def _idxin_roles(reqs):
    """``(miss_i, miss_set, hit_idx, source, note)`` -- MISS/HIT classification.

    PREFERS the ``sps`` rule (``sps==0`` -> MISS, ``sps>0`` -> HIT) -- the true
    prefill start position -- over the largest-prefill ``slq`` heuristic and over
    the /v1/chat/completions hit_idx.  When no request carries an ``sps`` (older
    probe) it falls back to the largest-``slq`` MISS plus its same-slq HIT(s).
    ``miss_i`` is the canonical (first) MISS.
    """
    miss, hit = _idxin_sps_roles(reqs)
    if miss:
        note = f"sps: sps==0 -> MISS {miss}, sps>0 -> HIT {hit}"
        return miss[0], miss, hit, "sps", note
    miss_i = _idxin_miss_index(reqs)
    same_hits, _slq, _slk = _idxin_same_len_hits(reqs, miss_i)
    if same_hits:
        note = f"same-slq/slk HIT(s)={same_hits}"
        hits = same_hits
    else:
        note = "no same-slq HIT -> fallback to all HITs (lengths differ)"
        hits = [j for j in range(len(reqs)) if j != miss_i and reqs[j]]
    return miss_i, [miss_i], hits, "largest-slq", note


def _distinct_km(group):
    """Distinct ``(klog, kpg)`` tuples of one request, sorted."""
    return sorted({(r.get("klog", "?"), r.get("kpg", "?")) for r in group}, key=str)


def report_idxin_requests(rows):
    """Per request (MISS vs each HIT), the DISTINCT ``(klog, kpg)`` values.

    ``klog`` is the md5 of the LOGICAL index-K content the request reads;
    ``kpg`` lists the PHYSICAL read-page ids that md5 was computed from.  A
    MISS and a HIT that read the same logical content from the same physical
    page print identical pairs.
    """
    if not rows:
        return
    report_idxin_split(rows)
    reqs = _idxin_requests(rows)
    n_req = sum(1 for g in reqs if g)
    if n_req < 2:
        print("  [IDXIN-KM] <2 requests -> per-request MISS/HIT split skipped")
        return
    miss_i, miss_set, _hit_idx, role_src, _note = _idxin_roles(reqs)
    print(f"  [IDXIN-KM] MISS=req{miss_i} (source={role_src}); "
          f"{n_req - len(miss_set)} HIT request(s)")
    for i, g in enumerate(reqs[:MAX_REQ_SHOWN]):
        if not g:
            continue
        role = "MISS" if i in miss_set else "HIT "
        km = _distinct_km(g)
        print(f"     req{i} {role} probes={len(g)} distinct(klog,kpg)={len(km)}")
        for klog, kpg in km[:MAX_DIFF_SHOWN]:
            print(f"        klog={klog} kpg={kpg}")
        if len(km) > MAX_DIFF_SHOWN:
            print(f"        ... {len(km) - MAX_DIFF_SHOWN} more")


def _km_classify(miss_set, hit_set):
    """Binary verdict for one probe: compare MISS vs HIT ``(klog,kpg)`` sets."""
    if miss_set == hit_set:
        return "SAME at these probes"
    if {k for k, _ in miss_set} != {k for k, _ in hit_set}:
        return "COMPRESSOR-OUTPUT differs"
    return "READ-PAGE misaligned"


def _idxin_layer_verdict(rows):
    """(per_layer, miss_i, nreqs) for the [IDXIN] MISS-vs-HIT klog comparison.

    ``per_layer[ly]`` = ``{same, comp, page, first_comp, first_page}``; ``comp``
    counts ``(layer,lastpos)`` probes where the MISS logical index-K md5
    (``klog``) differs from a HIT -> compressor-output divergence.  Shared by
    ``report_idxin_verdict`` (which prints it) and ``report_idxin_rope`` (which
    feeds the ROPE-PHASE verdict).
    """
    reqs = _idxin_requests(rows)
    miss_i = _idxin_miss_index(reqs)
    probes = defaultdict(lambda: defaultdict(set))
    for i, g in enumerate(reqs):
        for r in g:
            key = (_int(r, "layer"), _int(r, "lastpos"))
            probes[key][i].add((r.get("klog", "?"), r.get("kpg", "?")))
    per_layer = defaultdict(
        lambda: {"same": 0, "comp": 0, "page": 0,
                 "first_comp": None, "first_page": None})
    for (_ly, lp), byreq in probes.items():
        if miss_i not in byreq:
            continue
        hit_sets = [s for j, s in byreq.items() if j != miss_i]
        if not hit_sets:
            continue
        verds = [_km_classify(byreq[miss_i], s) for s in hit_sets]
        agg = per_layer[_ly]
        if any(v == "COMPRESSOR-OUTPUT differs" for v in verds):
            agg["comp"] += 1
            if agg["first_comp"] is None:
                agg["first_comp"] = lp
        elif all(v == "SAME at these probes" for v in verds):
            agg["same"] += 1
        else:
            agg["page"] += 1
            if agg["first_page"] is None:
                agg["first_page"] = lp
    return per_layer, miss_i, len(reqs)


def _idxin_comp_differs(rows):
    """True/False when the index-K content (``klog``) diverges MISS-vs-HIT.

    ``None`` when there is no MISS/HIT split to judge.
    """
    per_layer, _miss_i, nreqs = _idxin_layer_verdict(rows)
    if nreqs < 2:
        return None
    return any(a["comp"] for a in per_layer.values())


def report_idxin_verdict(rows):
    """One ``[IDXIN-VERDICT]`` line per layer: MISS-vs-HIT ``(klog,kpg)``.

    Compares, at every ``(layer,lastpos)`` probe that BOTH the MISS request and
    at least one HIT request reached, the distinct ``(klog,kpg)`` sets:

      * equal                          -> ``SAME at these probes``
      * ``klog`` differs               -> ``COMPRESSOR-OUTPUT differs`` (the
                                          written index-K content differs)
      * ``klog`` same, ``kpg`` differs -> ``READ-PAGE misaligned`` (same content
                                          md5 read from a different page)
    """
    if not rows:
        return
    per_layer, _miss_i, nreqs = _idxin_layer_verdict(rows)
    if nreqs < 2:
        print("[IDXIN-VERDICT] need >=2 requests (MISS + HIT); none")
        return
    if not per_layer:
        print("[IDXIN-VERDICT] no overlapping (layer,lastpos) probes "
              "between MISS and HIT")
        return
    for _ly in sorted(per_layer):
        a = per_layer[_ly]
        if a["comp"]:
            verdict = "COMPRESSOR-OUTPUT differs"
        elif a["page"]:
            verdict = "READ-PAGE misaligned"
        else:
            verdict = "SAME at these probes"
        if a["first_comp"] is not None:
            extra = f" first_compressor_diff_lastpos={a['first_comp']}"
        elif a["first_page"] is not None:
            extra = f" first_readpage_diff_lastpos={a['first_page']}"
        else:
            extra = ""
        print(f"[IDXIN-VERDICT] layer={_ly} same={a['same']} "
              f"compressor_output_diff={a['comp']} readpage_diff={a['page']} "
              f"=> {verdict}{extra}")


def _short(v, n=64):
    """``str(v)`` truncated to ``n`` chars (with an ellipsis when cut)."""
    s = str(v)
    return s if len(s) <= n else s[:n - 3] + "..."


def _knorm_mean(v):
    """Leading float of ``knorm=<mean>:<n>:<rd>`` (or None when unparseable)."""
    if not v:
        return None
    try:
        return float(str(v).split(":", 1)[0])
    except (TypeError, ValueError):
        return None


def _rope_shift(v):
    """First ``shift=[..]`` entry of a ``ropephase=`` value (or None)."""
    if not v:
        return None
    m = ROPE_SHIFT.search(str(v))
    if not m:
        return None
    first = m.group(1).split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def _rope_sps(v):
    """First ``sps=[..]`` (start position) entry of a ``ropephase=`` value."""
    if not v:
        return None
    m = ROPE_SPS.search(str(v))
    if not m:
        return None
    first = m.group(1).split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def _idxin_knorm_by_block(group):
    """``{abs_shift: knorm_mean}`` for one request's [IDXIN] rows.

    Each row carries ``knorm=<mean>:<npairs>:<ropeDim>`` (the RoPE-invariant
    magnitude of the logical index-K slab) AND a ``ropephase`` whose ``shift``
    is the ABSOLUTE compressed block index -- so the row pairs its block index
    with that block's magnitude.  That pair IS the per-block signature; the
    whole-request value-SET is NOT comparable (the MISS's extra blocks would
    fake a CONTENT/WINDOW).  Rows without a parseable shift/knorm are skipped.
    """
    out = {}
    for r in group:
        sh = _rope_shift(r.get("ropephase"))
        kn = _knorm_mean(r.get("knorm"))
        if sh is not None and kn is not None:
            out[sh] = kn
    return out


def _knorm_block_verdict(miss_map, hit_map, tol):
    """``(verdict, rel, shared, miss_only, hit_only, offenders)`` per shared block.

    RoPE is orthonormal, so the per-pair magnitude of a given ABSOLUTE block is
    INVARIANT under the phase: the MISS and the matching HIT must print the same
    ``knorm`` for every block present in BOTH.  Only SHARED blocks are compared
    (a block present in just one request cannot be judged).  ``rel`` is the max
    relative gap over the shared blocks; ``offenders`` lists the worst
    ``(shift, miss, hit)`` triples over ``tol`` (largest gap first, capped).
    """
    miss_k, hit_k = set(miss_map), set(hit_map)
    shared = sorted(miss_k & hit_k)
    miss_only, hit_only = sorted(miss_k - hit_k), sorted(hit_k - miss_k)
    if not shared:
        return "NO-SHARED", None, shared, miss_only, hit_only, []
    rels = []
    for s in shared:
        a, b = miss_map[s], hit_map[s]
        denom = max(abs(a), abs(b), 1e-12)
        rels.append((abs(a - b) / denom, s, a, b))
    rel = max(r for r, _s, _a, _b in rels)
    offenders = [(s, a, b) for r, s, a, b in sorted(rels, key=lambda t: -t[0])
                 if r > tol][:MAX_OFFENDER_SHOWN]
    return ("CONTENT/WINDOW" if rel > tol else "SAME",
            rel, shared, miss_only, hit_only, offenders)


def _knorm_maps_agree(base, other, tol):
    """True when two ``{abs_shift: knorm}`` maps agree on every SHARED block."""
    for s in set(base) & set(other):
        a, b = base[s], other[s]
        if abs(a - b) / max(abs(a), abs(b), 1e-12) > tol:
            return False
    return True


def report_idxin_rope(rows, knorm_tol=0.05):
    """``[IDXIN-ROPE]`` MISS-vs-HIT RoPE-phase verdict (one line + per request).

    The A5 compressor anchors its RoPE table offset on the CALL-LOCAL compressed
    counter (``padScIdx = dealedScCnt``), so the same absolute index-K block is
    rotated by a call-origin phase.  RoPE is orthonormal, so the per-pair L2
    magnitude (``knorm``) is phase-INVARIANT:

      * ``knorm`` differs (rel > ``knorm_tol``) -> ``CONTENT/WINDOW`` (the
        index-K content/window itself differs, not just its rotation)
      * ``knorm`` equal & ``klog`` differs       -> ``ROPE-PHASE`` (the sole
        remaining difference is the RoPE phase the kernel applies)

    MISS/HIT classification: a request whose (prefill) start position ``sps`` is
    0 is the cache-MISS (a full prefill from position 0); an ``sps>0`` request is
    a cache-HIT (a suffix prefill).  This ``sps`` rule is preferred over the
    largest-prefill ``slq`` heuristic and over the /v1/chat/completions hit_idx.
    Several identical MISS curls are allowed: the first MISS is the reference and
    the others are checked for self-consistency.

    Comparison is PER-BLOCK: each request's ``knorm`` values are paired with
    their ABSOLUTE block index (the ``ropephase`` ``shift``) and ONLY the blocks
    present in BOTH the MISS and the HIT are compared.  RoPE is orthonormal, so a
    shared block's magnitude must match within ``knorm_tol``; the MISS's extra
    (or the HIT's extra) blocks are reported but NOT judged.  With no shared
    block the verdict is ``NO-SHARED-BLOCKS``.

    Requests are split by the ``/v1/chat/completions`` boundary when present
    (see ``_idxin_requests``), so ``req{i}`` matches the curl order and the
    ``[IDXIN-SPLIT]`` banner lets the user confirm which request is the HIT.

    Also prints the ``shift`` = ``abs - loc`` (the phase error: 0 for the MISS,
    ``start_pos // cmp_ratio`` for a suffix HIT).  Additive / read-only.
    """
    if not rows:
        return
    report_idxin_split(rows)
    reqs = _idxin_requests(rows)
    n_req = sum(1 for g in reqs if g)
    if n_req < 2:
        print("  [IDXIN-ROPE] <2 requests -> MISS/HIT RoPE split skipped")
        return
    miss_i, miss_set, hit_idx, role_src, role_note = _idxin_roles(reqs)
    miss_slq, miss_slk = _idxin_req_len(reqs[miss_i])
    hit_len_txt = ",".join(str(s) for s in sorted({_idxin_req_len(reqs[j])[0]
                                                   for j in hit_idx})) or "-"
    knorms, shifts, phases = [], [], []
    for g in reqs:
        knorms.append(sorted({r.get("knorm") for r in g if r.get("knorm")}))
        shifts.append(sorted({_rope_shift(r.get("ropephase")) for r in g
                              if _rope_shift(r.get("ropephase")) is not None}))
        phases.append(sorted({r.get("ropephase") for r in g if r.get("ropephase")}))

    print(f"  [IDXIN-ROPE] MISS=req{miss_i} (source={role_src}); "
          f"{len(miss_set)} MISS / {len(hit_idx)} HIT request(s)  "
          f"knorm_tol={knorm_tol}")
    print(f"  [IDXIN-ROPE] pairing: {role_note}")
    print(f"  [IDXIN-ROPE] miss_req={miss_set} hit_req={hit_idx} "
          f"miss_slq={miss_slq} hit_slq={hit_len_txt}")
    for i, _g in enumerate(reqs[:MAX_REQ_SHOWN]):
        if not _g:
            continue
        role = "MISS" if i in miss_set else "HIT "
        kmeans = [_knorm_mean(v) for v in knorms[i]]
        kmean_txt = ",".join(
            "?" if m is None else f"{m:.4g}" for m in kmeans[:MAX_DIFF_SHOWN])
        ph_txt = ";".join(_short(v) for v in phases[i][:2]) or "-"
        print(f"     req{i} {role} knorm[{len(knorms[i])}]={kmean_txt or '-'} "
              f"shift={shifts[i] or '-'} ropephase[{len(phases[i])}]={ph_txt}")

    # Per-BLOCK comparison: {abs_shift: knorm} for the MISS and the HIT(s),
    # compared ONLY on the blocks present in BOTH.  The MISS's (or the HIT's)
    # EXTRA blocks must not be judged -- a whole-request value-SET compare lets
    # them fake a CONTENT/WINDOW.  Identical MISS curls are also checked for
    # self-consistency against the first MISS.
    miss_maps = [_idxin_knorm_by_block(reqs[j]) for j in miss_set]
    miss_consistent = all(_knorm_maps_agree(miss_maps[0], m, knorm_tol)
                          for m in miss_maps[1:])
    miss_map = miss_maps[0]
    hit_map = {}
    for j in hit_idx:
        hit_map.update(_idxin_knorm_by_block(reqs[j]))
    knorm_cmp, rel, shared, miss_only, hit_only, offenders = (
        _knorm_block_verdict(miss_map, hit_map, knorm_tol))
    klog_diff = _idxin_comp_differs(rows)
    if knorm_cmp == "CONTENT/WINDOW":
        verdict = "CONTENT/WINDOW"
    elif knorm_cmp == "NO-SHARED":
        verdict = "NO-SHARED-BLOCKS"
    elif klog_diff:
        verdict = "ROPE-PHASE"
    elif klog_diff is None:
        verdict = "UNKNOWN"
    else:
        verdict = "SAME"
    rel_txt = "na" if rel is None else f"{rel:.3e}"
    print(f"[IDXIN-ROPE] miss_req={miss_set} hit_req={hit_idx} "
          f"miss_consistent={miss_consistent} n_shared={len(shared)} "
          f"n_miss_only={len(miss_only)} n_hit_only={len(hit_only)} "
          f"maxrel={rel_txt} knorm_same={knorm_cmp == 'SAME'} "
          f"klog_diff={klog_diff} => {verdict}")
    if offenders:
        off_txt = "  ".join(
            f"shift={s}:miss={a:.4g}->hit={b:.4g}" for s, a, b in offenders)
        print(f"[IDXIN-ROPE] differing shared blocks (>{knorm_tol}): {off_txt}")


QK_ALIGNED_RUNBOOK = (
    "[QK-ALIGNED-RUNBOOK] query vs index-K at P* (per absolute position/block):\n"
    "  DSV4_DUMP_IDXIN=<layer> DSV4_DUMP_IDXK=<layer> DSV4_DUMP_C4POS=1 "
    "DSV4_DUMP_MIN_POS=17590 DSV4_DUMP_MAX_POS=17605 "
    "<serve> ; <curl miss> ; <curl hit> ; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_qk.log\n"
    "  qpos=[pos:md5|..] = query= operand (q_quant) per ABSOLUTE position; "
    "[IDXK] blkx=[blk:md5] = index-K pool page per ABSOLUTE 128-token block"
)


def _idxin_qpos(rec):
    """Parse ``qpos=[pos:md5|pos:md5|..]`` into ``{abs_pos: md5}``.

    ``qpos`` is the per-ABSOLUTE-POSITION hash of the indexer op's ``query=``
    operand (``q_quant``), emitted at the op-input point.  It is the ONLY query
    signal that aligns a MISS and a HIT at the SAME DECODE step: ``qblk`` keys by
    128-token block, and at decode block 137 spans 128 positions, so a MISS and a
    HIT at the divergence position P*=17598 would land in the SAME group as every
    other decode step.  Empty for probes without the field.
    """
    v = rec.get("qpos")
    if not v:
        return {}
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    out = {}
    for entry in s.split("|"):
        entry = entry.strip()
        if ":" not in entry:
            continue
        p, _, h = entry.partition(":")
        try:
            out[int(p.strip())] = h.strip()
        except ValueError:
            continue
    return out


def _qpos_maps(group):
    """``{layer: {abs_pos: q_md5}}`` of one request's [IDXIN] rows."""
    out = defaultdict(dict)
    for r in group:
        ly = _int(r, "layer")
        for pos, h in _idxin_qpos(r).items():
            out[ly].setdefault(pos, h)
    return out


def _idxk_aligned_div(idxk_rows):
    """``[(layer, block, lastpos, hashes)]`` where 2+ requests disagree at the
    SAME decode step.

    Keyed by ``(layer, block, lastpos)`` so the compare is at the SAME absolute
    decode position, NOT across the evolving page: within one request a
    ``lastpos`` has exactly one hash for a block, so >1 distinct hash among
    requests is a REAL MISS-vs-HIT index-K pool divergence (a per-block-only
    grouping would false-positive on the partially-written tail page).
    """
    groups = defaultdict(lambda: defaultdict(set))  # (ly,blk,lastpos) -> {req: hashes}
    for r in idxk_rows:
        ly = _int(r, "layer")
        lp = _int(r, "lastpos", -1)
        req = _int(r, "_req", -1)
        for b, h in _parse_blkx(r, "blkx").items():
            groups[(ly, b, lp)][req].add(h)
    out = []
    for (ly, b, lp), byreq in groups.items():
        if len(byreq) < 2:
            continue
        allh = set().union(*byreq.values())
        if len(allh) > 1:
            out.append((ly, b, lp, sorted(allh)))
    out.sort()
    return out


def report_qk_aligned(idxin_rows, idxk_rows=(), c4pos_rows=()):
    """``[QK-ALIGNED]`` query vs index-K at P* -- the decisive q-vs-K split.

    The indexer op ``npu_quant_lightning_indexer`` is fed TWO operands
    (``ascend_dsv4_backend.py:1898-1915``):
      * (a) QUERY -- ``query=q_quant``: this report reads the ``qpos=[pos:md5]``
        field (per ABSOLUTE position) emitted at the op-input point, so MISS and
        HIT are compared at the SAME decode position P*;
      * (b) KEYS  -- ``key=<index-K pool>``: the [IDXIN] ``klog`` whole-logical-read
        content (``_idxin_comp_differs``, keyed by ``(layer,lastpos)``) plus the
        [IDXK] ``blkx=[blk:md5]`` per-absolute-block pool divergence, aligned by
        ``(layer,block,lastpos)`` so the evolving tail page cannot false-positive.

    MISS/HIT roles reuse the [IDXIN] ``sps`` rule (``_idxin_roles``).  The query
    side also runs a two-MISS control (self-control): two MISS requests must
    agree at every shared position.

    Decision tree (printed as ``[QK-ALIGNED] DECISION``):
      * ``Q_DIFF``  -> divergence is the QUERY path (q proj/quant/RoPE);
      * ``K_DIFF``  -> upstream compressor / state at P* (trace the compressor
        state read at P*, ``_build_explicit_state_block_table`` /
        ``translate_from_swa_loc_to_state_loc``);
      * both SAME   -> the top-k score/reduction inside the indexer is
        shape-dependent (probe the score / top-k ORDER, cf.
        ``[CMPIDX-ORDERED-ALIGNED]``).
    """
    print("\n== [QK-ALIGNED] query (qpos) vs index-K (klog/blkx) at P* ==")
    if not idxin_rows:
        print("  [QK-ALIGNED] no [IDXIN] rows -> run with DSV4_DUMP_IDXIN=<layer>")
        print(QK_ALIGNED_RUNBOOK)
        return
    if not any("qpos" in r for r in idxin_rows):
        print("  [QK-ALIGNED] [IDXIN] has no `qpos=` field (probe without the "
              "windowed query hash) -> cannot judge the QUERY side; re-run with "
              "DSV4_DUMP_MIN_POS/MAX_POS set (the field is window-gated)")
        print(QK_ALIGNED_RUNBOOK)
        return
    reqs = _idxin_requests(idxin_rows)
    n_req = sum(1 for g in reqs if g)
    if n_req < 2:
        print("  [QK-ALIGNED] <2 requests -> no MISS/HIT split")
        print(QK_ALIGNED_RUNBOOK)
        return
    miss_i, miss_set, hit_idx, role_src, note = _idxin_roles(reqs)
    report_idxin_split(idxin_rows)
    print(f"  [QK-ALIGNED] miss_req={miss_set} hit_req={hit_idx} "
          f"source={role_src}  ({note})")

    # (a) QUERY: per absolute position.
    miss_q = _qpos_maps(reqs[miss_i])
    hit_qs = [_qpos_maps(reqs[j]) for j in hit_idx]
    q_cmp = q_same = q_diff = 0
    q_offenders = []
    for ly in sorted(miss_q):
        for pos in sorted(miss_q[ly]):
            for hqm in hit_qs:
                if ly in hqm and pos in hqm[ly]:
                    q_cmp += 1
                    if miss_q[ly][pos] == hqm[ly][pos]:
                        q_same += 1
                    else:
                        q_diff += 1
                        q_offenders.append(
                            (ly, pos, miss_q[ly][pos], hqm[ly][pos]))
    q_verdict = ("Q_DIFF" if q_diff else
                 ("Q_SAME" if q_cmp else "Q_NO_SHARED"))
    print(f"  [QK-ALIGNED] QUERY(qpos): compared={q_cmp} same={q_same} "
          f"diff={q_diff} => {q_verdict}")
    for ly, pos, a, b in q_offenders[:MAX_OFFENDER_SHOWN]:
        print(f"     [Q] layer={ly} pos={pos} miss={a} hit={b}")
    if q_offenders:
        _firstq = min(q_offenders, key=lambda t: t[1])
        print(f"  [QK-ALIGNED] FIRST query diff at layer={_firstq[0]} "
              f"pos={_firstq[1]}")

    # two-MISS control (self-control): two MISS requests must agree on qpos.
    if len(miss_set) >= 2:
        base = _qpos_maps(reqs[miss_set[0]])
        ctl = ctl_same = ctl_diff = 0
        for m in miss_set[1:]:
            o = _qpos_maps(reqs[m])
            for ly in base:
                for pos in base[ly]:
                    if ly in o and pos in o[ly]:
                        ctl += 1
                        if base[ly][pos] == o[ly][pos]:
                            ctl_same += 1
                        else:
                            ctl_diff += 1
        if ctl == 0:
            print("  [QK-ALIGNED] two-MISS control: no shared qpos position")
        elif ctl_diff == 0:
            print(f"  [QK-ALIGNED] two-MISS control: IDENTICAL at all {ctl} "
                  f"shared qpos position(s) (sanity OK)")
        else:
            print(f"  [QK-ALIGNED] WARN two-MISS control: {ctl_diff}/{ctl} "
                  f"shared qpos position(s) DIFFER between two MISS requests")

    # (b) KEYS: whole-read content (klog) + per-block pool divergence ([IDXK]).
    klog_diff = _idxin_comp_differs(idxin_rows)
    kblk_diff = _idxk_aligned_div(idxk_rows) if idxk_rows else []
    if kblk_diff or klog_diff:
        k_verdict = "K_DIFF"
    elif klog_diff is False:
        k_verdict = "K_SAME"
    else:
        k_verdict = "K_UNKNOWN"
    _kblk_show = [f"L{ly}/blk{b}/lp{lp}" for ly, b, lp, _h in kblk_diff]
    print(f"  [QK-ALIGNED] KEYS: klog_diff={klog_diff} "
          f"blkx_aligned_diff={len(kblk_diff)} "
          f"{_kblk_show[:MAX_OFFENDER_SHOWN]} => {k_verdict}")

    # DECISION.
    print("  [QK-ALIGNED] DECISION:")
    if q_verdict == "Q_DIFF":
        print("     QUERY-PATH: q differs at P* -> divergence is the QUERY "
              "(q proj/quant/RoPE); index-K is at most secondary")
    elif k_verdict == "K_DIFF":
        print("     COMPRESSOR/STATE: index-K differs at P* -> upstream "
              "compressor/state; trace the state read at P* "
              "(_build_explicit_state_block_table / "
              "translate_from_swa_loc_to_state_loc)")
    elif q_verdict == "Q_SAME" and k_verdict == "K_SAME":
        print("     BOTH-SAME: neither op input differs -> the indexer's "
              "score/top-k reduction is shape-dependent; probe the score/ORDER "
              "next (see [CMPIDX-ORDERED-ALIGNED])")
    else:
        print(f"     INCONCLUSIVE: q={q_verdict} k={k_verdict} -> enable BOTH "
              f"DSV4_DUMP_IDXIN and DSV4_DUMP_IDXK and set the window "
              f"(DSV4_DUMP_MIN_POS/MAX_POS)")
    print(QK_ALIGNED_RUNBOOK)


C4POS_RUNBOOK = (
    "[C4POS-RUNBOOK] DSV4_DUMP_C4POS=1 bash start_test.sh > /tmp/dsv4_c4pos.log 2>&1 & "
    "bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh; "
    "grep -n '\\[C4POS\\]' /tmp/dsv4_c4pos.log; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_c4pos.log"
)


def _c4pos_ints(rec):
    """Parse the compact ``pos=[a,b,c]`` list of one [C4POS] row into ints."""
    v = rec.get("pos")
    if not v:
        return []
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


def _c4pos_requests(rows):
    """Split [C4POS] rows into requests (chat/completions, else layer-reset).

    The strongest boundary is the ``/v1/chat/completions`` line (``_req`` stamped
    by ``parse``); with none present we fall back to the layer index RESET, which
    happens exactly once per request (cmp layers emit in ascending order).
    """
    if any(_int(r, "_req", 0) > 0 for r in rows):
        return _idxin_chat_split(rows)[0]
    reqs = []
    prev_ly = None
    for r in rows:
        ly = _int(r, "layer")
        if not reqs or (prev_ly is not None and 0 <= ly < prev_ly):
            reqs.append([])
        reqs[-1].append(r)
        prev_ly = ly
    return reqs


def _c4pos_prefill_maps(group):
    """``{layer: (first, ratio, pos_list)}`` of the PREFILL row per layer.

    A request emits one [C4POS] row per compressor call (one prefill + every
    decode step); the prefill row is the one with the LONGEST compact ``pos``
    list, so for each layer keep the max-length row.
    """
    out = {}
    for r in group:
        pos = _c4pos_ints(r)
        if not pos:
            continue
        ly = _int(r, "layer")
        if ly not in out or len(pos) > len(out[ly][2]):
            out[ly] = (_int(r, "first"), _int(r, "ratio", 4), pos)
    return out


def _c4pos_value_maps(pos_list, ratio):
    """``{abs_block: value}`` keyed by the ABSOLUTE compressed block.

    The table value IS the absolute RoPE position, so ``block = value // ratio``.
    A MISS and a HIT that anchor the SAME absolute block must carry the SAME
    value; anything else is a host table bug.
    """
    if ratio <= 0:
        return {}
    return {v // ratio: v for v in pos_list}


def report_c4pos(rows):
    """``[C4POS-VERDICT]`` MISS-vs-HIT per ABSOLUTE compressed block.

    The host builds ``positions_cmp_padding_c{ratio}`` as a COMPACT per-call
    table already anchored at absolute positions (entry j = absolute token
    ``(first_k+j)*ratio``) and the kernel indexes it with a call-local
    ``padScIdx``; so the RoPE value for a given ABSOLUTE block must be identical
    on a MISS and a HIT:

      * equal on every shared block -> ``phase_same`` (compressor RoPE provably
        identical; any remaining divergence is CONTENT / output-slot, not phase)
      * any shared block differs    -> ``phase_diff`` (host metadata bug)
    """
    print("\n== [C4POS] positions_cmp_padding_c4 (absolute-block parity) ==")
    if not rows:
        print("  [C4POS-VERDICT] no [C4POS] rows (dump gate off?)")
        print(C4POS_RUNBOOK)
        return
    reqs = _c4pos_requests(rows)
    sides = [_c4pos_prefill_maps(g) for g in reqs]
    miss_idx, hit_idx = [], []
    for i, m in enumerate(sides):
        if not m:
            continue
        # MISS = a full prefill whose compact table starts at absolute block 0.
        if min(first for first, _r, _p in m.values()) == 0:
            miss_idx.append(i)
        else:
            hit_idx.append(i)
    if not miss_idx or not hit_idx:
        print(f"  [C4POS-VERDICT] miss_req={miss_idx} hit_req={hit_idx} "
              f"-> need >=1 MISS (first==0) and >=1 HIT (first>0); none")
        print(C4POS_RUNBOOK)
        return
    miss_layers = sides[miss_idx[0]]
    shared_total = miss_only_total = hit_only_total = 0
    diff_layers, per_layer = [], []
    for ly in sorted(miss_layers):
        _miss_first, ratio, miss_pos = miss_layers[ly]
        miss_map = _c4pos_value_maps(miss_pos, ratio)
        shared = miss_only = hit_only = 0
        offenders = []
        for j in hit_idx:
            hm = sides[j].get(ly)
            if hm is None:
                continue
            hit_map = _c4pos_value_maps(hm[2], ratio)
            sh = set(miss_map) & set(hit_map)
            shared += len(sh)
            miss_only += len(set(miss_map) - set(hit_map))
            hit_only += len(set(hit_map) - set(miss_map))
            for b in sorted(sh):
                if miss_map[b] != hit_map[b]:
                    offenders.append((b, miss_map[b], hit_map[b]))
        if offenders:
            diff_layers.append(ly)
        per_layer.append((ly, offenders))
        shared_total += shared
        miss_only_total += miss_only
        hit_only_total += hit_only
    verdict = "phase_diff" if diff_layers else "phase_same"
    print(f"  [C4POS-VERDICT] miss_req={miss_idx} hit_req={hit_idx} "
          f"layers={len(miss_layers)} n_shared={shared_total} "
          f"n_miss_only={miss_only_total} n_hit_only={hit_only_total} "
          f"=> {verdict}")
    if diff_layers:
        print(f"  [C4POS-VERDICT] phase-diff layers={diff_layers}")
        for ly, off in per_layer:
            if off:
                off_txt = "  ".join(
                    f"block{b}:miss={a}->hit={c}"
                    for b, a, c in off[:MAX_OFFENDER_SHOWN])
                print(f"     layer={ly} {off_txt}")
    else:
        print("  -> every shared absolute block has the SAME value MISS vs HIT: "
              "compressor RoPE is identical; look at CONTENT/output-slot, not "
              "phase")
    print(C4POS_RUNBOOK)


WSTAGES = ("S0", "S1", "S2", "S3", "S4", "S5")
MAX_WSTAGE_SHOWN = 20
# First-differing write-stage -> root-cause verdict (from the S253/T6 probe).
WSTAGE_VERDICT = {
    "S0": "UPSTREAM-INPUT (layer input differs)",
    "S1": "A-GEMM (npu_quant_matmul KV projection M-dependent split-K)",
    "S2": "kv_norm",
    "S3": "C-ROPE (inplace_partial_rotary_mul)",
    "S4": "writer-assembly",
    "S5": "E-fp8-epilog (kv_compress_epilog)",
}
WSTAGE_ALL_SAME = (
    "ALL-SAME (write clean -> downstream attention/read or input side)")

WSTAGE_RUNBOOK = (
    "[WSTAGE-RUNBOOK] write-side stage sweep (MISS vs HIT per absolute position):\n"
    "  DSV4_DUMP_WSTAGE=1 DSV4_DUMP_C4POS=1 "
    "DSV4_DUMP_MIN_POS=<win_lo> DSV4_DUMP_MAX_POS=<win_hi> "
    "<serve> ; <curl miss> ; <curl hit> ; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_wstage.log\n"
    "  [WSTAGE] layer=<L> pos=<p> stage=S<i> md5=<h> (i in 0..5: S0 pre-GEMM, "
    "S1 post-GEMM kv, S2 post-norm, S3 post-rope, S4 pre-epilog bf16, S5 stored "
    "fp8); the parser is keyed by ABSOLUTE position (never a whole-log compare)"
)


def _wstage_maps(group):
    """``{layer: {abs_pos: {stage: md5}}}`` of one request's [WSTAGE] rows.

    Keyed by ABSOLUTE position (``pos=``), NEVER a whole-log compare: the MISS
    (full prefill) and the HIT (suffix prefill) have different ``ntok``, so only
    the absolute position aligns the two write-side stage sweeps.  The FIRST row
    carrying a ``(pos, stage)`` wins; rows missing ``layer``/``pos``/``stage``/
    ``md5`` or carrying an unknown stage are skipped.
    """
    out = defaultdict(dict)
    for r in group:
        ly = _int(r, "layer")
        pos = _int(r, "pos", None)
        stage = r.get("stage", "")
        md5 = r.get("md5")
        if pos is None or not md5 or stage not in WSTAGES:
            continue
        out.setdefault(ly, {}).setdefault(pos, {}).setdefault(stage, md5)
    return out


def _wstage_stage_str(maps, ly, pos):
    """``S0:a,S1:b,...`` per-stage distinct md5s for one (layer,pos) side."""
    parts = []
    for st in WSTAGES:
        vals = sorted({m.get(ly, {}).get(pos, {}).get(st) for m in maps
                       if m.get(ly, {}).get(pos, {}).get(st) is not None})
        if vals:
            parts.append(f"{st}:{'/'.join(vals)}")
    return ",".join(parts)


def report_wstage(rows, c4pos_rows=(), idxin_rows=()):
    """``[WSTAGE-VERDICT]`` MISS-vs-HIT write-side stage sweep per abs position.

    The S253/T6 probe emits, one line per position per stage::

        [WSTAGE] layer=<L> pos=<p> stage=S<i> md5=<h>   (i in 0..5)

    with S0 pre-GEMM, S1 post-GEMM kv, S2 post-norm, S3 post-rope, S4 pre-epilog
    bf16, S5 stored fp8.  This report ALIGNS the MISS and the HIT at the SAME
    ABSOLUTE position (never a whole-log compare: the two requests have different
    ``ntok``) and, per shared ``(layer, abs_pos)``, finds the FIRST stage
    ``S0..S5`` whose md5 differs (MISS vs HIT):

      * offending positions print ``layer=<L> pos=<p> first_diff_stage=S<i>``
        with each side's per-stage md5 run (capped);
      * the summary prints the MODE of ``first_diff_stage`` over the offending
        positions, mapped to the T6 verdict::

            S1 -> A-GEMM     S2 -> kv_norm      S3 -> C-ROPE
            S4 -> writer-assembly               S5 -> E-fp8-epilog
            none -> ALL-SAME (write clean)      S0 -> UPSTREAM-INPUT

    Requests are split by the SAME rule the rest of the analyzer uses: the
    [C4POS] ``first==0`` / ``first>0`` label, else the [IDXIN] ``sps==0`` /
    ``sps>0`` rule, else largest-``ntok`` (the SOURCE is printed).  A two-MISS
    self-control verifies two MISS requests are IDENTICAL at every shared
    ``(layer, pos, stage)``; on MISMATCH it WARNS and marks the verdict
    UNRELIABLE.  When ``[WSTAGE]`` rows are absent the report prints exactly what
    is needed (env + window) and NEVER a false PASS/FAIL.
    """
    print("\n== [WSTAGE-VERDICT] write-side stage sweep (MISS vs HIT) ==")
    if not rows:
        print("  [WSTAGE-VERDICT] no [WSTAGE] rows -> need "
              "DSV4_DUMP_WSTAGE=1 and a WINDOW "
              "(DSV4_DUMP_MIN_POS + DSV4_DUMP_MAX_POS) around the position")
        print(WSTAGE_RUNBOOK)
        return
    if not any("stage" in r for r in rows):
        print("  [WSTAGE-VERDICT] [WSTAGE] rows carry no `stage=S<i>` field -> "
              "cannot sweep the write stages; re-run the S253/T6 probe")
        print(WSTAGE_RUNBOOK)
        return
    reqs = _c4pos_requests(rows)
    source, miss_idx, hit_idx, note = _cmp_aligned_roles(
        reqs, c4pos_rows, idxin_rows)
    print(f"  [WSTAGE-VERDICT] split_source={source} miss_req={miss_idx} "
          f"hit_req={hit_idx}")
    print(f"  [WSTAGE-VERDICT] rules: {note}")
    if not miss_idx or not hit_idx:
        print("  [WSTAGE-VERDICT] need >=1 MISS (first==0) and >=1 HIT "
              "(first>0); none")
        print(WSTAGE_RUNBOOK)
        return
    miss_maps = [_wstage_maps(reqs[i]) for i in miss_idx]
    hit_maps = [_wstage_maps(reqs[i]) for i in hit_idx]

    # --- two-MISS self-control: two MISS requests must be IDENTICAL ---
    unreliable = False
    if len(miss_maps) < 2:
        print("  [WSTAGE-VERDICT] two-MISS SKIP (need >=2 MISS requests; got "
              f"{len(miss_maps)})")
    else:
        base = miss_maps[0]
        ctl_shared = ctl_diff = 0
        first_bad2 = None
        for ly in sorted(base):
            for pos in sorted(base[ly]):
                if not all(ly in m and pos in m[ly] for m in miss_maps[1:]):
                    continue
                ctl_shared += 1
                for st in WSTAGES:
                    vals = {m[ly][pos].get(st) for m in miss_maps}
                    if len(vals) > 1:
                        ctl_diff += 1
                        if first_bad2 is None:
                            first_bad2 = (ly, pos, st)
                        break
        if ctl_shared == 0:
            print("  [WSTAGE-VERDICT] two-MISS SKIP (no (layer,pos) shared by "
                  "both MISS requests)")
        elif ctl_diff == 0:
            print(f"  [WSTAGE-VERDICT] two-MISS IDENTICAL "
                  f"(at all {ctl_shared} shared position(s))")
        else:
            unreliable = True
            print(f"  [WSTAGE-VERDICT] WARNING two-MISS MISMATCH "
                  f"({ctl_diff}/{ctl_shared} shared position(s) differ between "
                  f"two MISS requests; first layer={first_bad2[0]} "
                  f"pos={first_bad2[1]} stage={first_bad2[2]}) -> verdict "
                  f"UNRELIABLE")

    # union of (layer, abs_pos) over the MISS side and the HIT side.
    miss_keys = {(ly, pos) for m in miss_maps for ly in m for pos in m[ly]}
    hit_keys = {(ly, pos) for h in hit_maps for ly in h for pos in h[ly]}
    shared = sorted(miss_keys & hit_keys)
    if not shared:
        print("  [WSTAGE-VERDICT] no (layer,pos) reached by BOTH a MISS and a "
              "HIT -> cannot judge (need the SAME absolute window in both)")
        print(WSTAGE_RUNBOOK)
        return

    offenders = []
    modes = defaultdict(int)
    n_same = 0
    for ly, pos in shared:
        first = None
        for st in WSTAGES:
            hashes = set()
            for m in miss_maps:
                v = m.get(ly, {}).get(pos, {}).get(st)
                if v is not None:
                    hashes.add(v)
            for h in hit_maps:
                v = h.get(ly, {}).get(pos, {}).get(st)
                if v is not None:
                    hashes.add(v)
            if len(hashes) > 1:
                first = st
                break
        if first is None:
            n_same += 1
        else:
            modes[first] += 1
            offenders.append((ly, pos, first))

    for ly, pos, first in offenders[:MAX_WSTAGE_SHOWN]:
        print(f"  [WSTAGE-VERDICT] layer={ly} pos={pos} "
              f"first_diff_stage={first} "
              f"miss={_wstage_stage_str(miss_maps, ly, pos)} "
              f"hit={_wstage_stage_str(hit_maps, ly, pos)}")
    if len(offenders) > MAX_WSTAGE_SHOWN:
        print(f"  ... {len(offenders) - MAX_WSTAGE_SHOWN} more positions")

    if modes:
        mode_stage = sorted(modes,
                            key=lambda s: (-modes[s], WSTAGES.index(s)))[0]
        verdict = WSTAGE_VERDICT.get(mode_stage, "UNKNOWN")
        mode_txt = mode_stage
    else:
        verdict = WSTAGE_ALL_SAME
        mode_txt = "none"
    suffix = " [UNRELIABLE]" if unreliable else ""
    print(f"  [WSTAGE-VERDICT] S0..S5: first_diff_stage MODE = {mode_txt} "
          f"(=> {verdict}){suffix}")
    print(f"  [WSTAGE-VERDICT] positions_compared={len(shared)} "
          f"same={n_same} diff={len(offenders)}")
    print(WSTAGE_RUNBOOK)


C4DEC_RUNBOOK = (
    "[C4DEC-RUNBOOK] windowed probe (compressor produced idxK vs stored pool):\n"
    "  DSV4_DUMP_C4DEC=2 DSV4_DUMP_IDXK=2 DSV4_DUMP_IDXIN=2 "
    "DSV4_DUMP_LAYERS=2 DSV4_DUMP_MIN_POS=17280 DSV4_DUMP_MAX_POS=17540 "
    "SGLANG_DSV4_AIC_NUM=<real> <serve> ; <curl miss> ; <curl hit> ; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/dsv4_c4dec.log\n"
    "  verdicts: OUTPUT-SLOT (produced same, stored differs) | "
    "CONTENT:K-SPLIT (kbase differs) | "
    "CONTENT:READ-MISALIGN (kbase same, rdcarry differs) | SAME"
)
# The two probe sides live in DIFFERENT key spaces and must be translated before
# the join:
#   * ``[C4DEC] idxK`` keys are RATIO-compressed block ids (the probe emits the
#     hardcoded tail blocks 4382/4383 = ratio-4 blocks);
#   * ``[IDXK] blkx`` keys are c4-PAGE columns (absolute token block j, entry j
#     <-> position j*128).
# A ratio-``r`` block ``b`` covers tokens [b*r, (b+1)*r); a page ``p`` covers
# [p*128, (p+1)*128); hence page = b*r // 128 (= b//32 for r=4).  Without this
# translation the join looked up stored page 4383 (absent) and missed real
# divergences such as page 136.
C4_PAGE_TOKENS = 128   # a stored c4 page spans 128 absolute tokens
C4_DEFAULT_RATIO = 4   # c4 compressor ratio when the probe omits ratio=



def _c4dec_start_min(rec):
    """Min of the ``start=[..]`` list (the call's start position / ``sps``).

    ``0`` marks a full prefill (cache-MISS); ``>0`` a suffix prefill (cache-HIT).
    """
    v = rec.get("start")
    if not v:
        return None
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            continue
    return min(vals) if vals else None


def _c4dec_idxk(rec):
    """Parse ``idxK=[n_raw=..,n_keep=..,<block>:<hash>:v0=..:sum=..:am=.., ..]``.

    Returns ``(by_block, meta)`` where ``by_block`` = ``{abs_block: md5}`` (the
    compressor's PRODUCED index-K) and ``meta`` = ``{n_raw, n_keep}``.
    """
    v = rec.get("idxK")
    if not v:
        return {}, {}
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    by_block, meta = {}, {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if "=" in parts[0]:
            k, _, val = tok.partition("=")
            meta[k] = val
            continue
        if len(parts) >= 2:
            try:
                by_block[int(parts[0])] = parts[1]
            except ValueError:
                continue
    return by_block, meta


def _c4dec_rdcarry(rec):
    """Parse ``rdcarry=[<pos>=<sl>:<h>, ..]`` into ``{pos: "<sl>:<h>"}``."""
    v = rec.get("rdcarry")
    if not v:
        return {}
    s = str(v).strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if "=" not in tok:
            continue
        k, _, val = tok.partition("=")
        out[k.strip()] = val.strip()
    return out


def _c4dec_split(rows):
    """Split [C4DEC]/[IDXK] rows into requests (chat boundary, else layer reset)."""
    return _c4pos_requests(rows)


def _c4dec_roles(c4_reqs, idxin_rows):
    """``(miss_idx, hit_idx, source)`` for the [C4DEC] requests.

    MISS/HIT is decided by the REQUEST BOUNDARY rule the [IDXIN]/[C4POS] sections
    use -- the ``/v1/chat/completions`` split plus the ``sps`` rule (``sps==0`` ->
    MISS, ``sps>0`` -> HIT) -- NOT by [C4DEC]'s own ``start==[]``.  The [IDXIN]
    rows are split by the SAME ``/v1/chat/completions`` boundaries as the [C4DEC]
    rows, so their request indices align; the [IDXIN] sps classification is
    mapped straight onto the [C4DEC] request indices.  This is REQUIRED because
    the widened [C4DEC] gate makes the MISS request's emitted call a
    window-overlapping CHUNK whose ``start`` list is 0-FREE, so the old
    ``start==0`` test mislabelled the MISS as HIT (``miss_req=[]``).  The
    [C4DEC] ``start`` list is used ONLY as a secondary fallback, never as the
    sole MISS test.
    """
    if idxin_rows and _idxin_split_source(idxin_rows) == "chat/completions":
        ireqs = _idxin_requests(idxin_rows)
        if len(ireqs) == len(c4_reqs):
            _miss_i, miss_set, hit_idx, source, _note = _idxin_roles(ireqs)
            if miss_set or hit_idx:
                return list(miss_set), list(hit_idx), f"idxin:{source}"
    # Secondary hint only (no usable [IDXIN]): the [C4DEC] start==0 test.
    miss, hit = [], []
    for i, g in enumerate(c4_reqs):
        starts = [s for s in (_c4dec_start_min(r) for r in g) if s is not None]
        if not starts:
            continue
        (miss if min(starts) == 0 else hit).append(i)
    return miss, hit, "c4dec-start"


def _c4dec_ratio(rec):
    """Compressor ratio of one [C4DEC] row (``ratio=`` field, else 4).

    The [C4DEC] emit does not print ``ratio=`` today, so the c4 default (4) is
    used; a future probe that adds the field is honored automatically.
    """
    try:
        r = int(rec.get("ratio"))
    except (TypeError, ValueError):
        return C4_DEFAULT_RATIO
    return r if r > 0 else C4_DEFAULT_RATIO


def _c4dec_block_page(block, ratio):
    """ABSOLUTE stored-page id a produced ratio-``ratio`` block ``b`` lands in.

    ``[C4DEC] idxK`` keys are ratio-compressed block ids; ``[IDXK] blkx`` keys are
    c4-page columns.  A ratio-``r`` block covers tokens ``[b*r, (b+1)*r)`` and a
    page covers ``[p*128, (p+1)*128)``, so ``page = block*ratio // 128``
    (ratio=4 -> ``block // 32``: 4382/4383 -> page 136).
    """
    return (block * ratio) // C4_PAGE_TOKENS


def _c4dec_produced_maps(group):
    """``{layer: (idxK_by_block, kbase, rdcarry_by_pos, ratio)}`` one request."""
    out = {}
    for r in group:
        if not _is_prefill(r):
            continue
        idxk, _meta = _c4dec_idxk(r)
        if not idxk:
            continue
        out[_int(r, "layer")] = (
            idxk, r.get("kbase", ""), _c4dec_rdcarry(r), _c4dec_ratio(r))
    return out


def _c4dec_stored_maps(group):
    """``{layer: {abs_block: md5}}`` of the stored [IDXK] pool content."""
    out = {}
    for r in group:
        if not _is_prefill(r):
            continue
        blk = _parse_blkx(r, "blkx")
        if not blk:
            continue
        out[_int(r, "layer")] = blk
    return out


def report_c4dec(rows, idxk_rows=(), idxin_rows=()):
    """``[C4DEC-VERDICT]`` produced idxK vs stored [IDXK] blkx, per ABSOLUTE page.

    produced = the compressor's PRODUCED compressed KV ([C4DEC] ``idxK``);
    stored   = the POOL content per absolute page ([IDXK] ``blkx[j]``).

    The two sides use DIFFERENT key spaces: produced idxK keys are ratio-4
    COMPRESSED-BLOCK ids (the probe emits the hardcoded tail blocks 4382/4383),
    while stored blkx keys are c4-PAGE columns.  The produced keys are therefore
    TRANSLATED to the stored-page space (``page = block*ratio // 128``, i.e.
    ``block//32`` for ratio=4: 4382/4383 -> page 136) BEFORE the join; without
    this the join looked up stored page 4383 (absent) and missed page 136.

    Requests are split by the ``/v1/chat/completions`` boundary and MISS/HIT is
    classified by the REQUEST-BOUNDARY rule the [IDXIN] section uses (``sps==0``
    -> MISS, ``sps>0`` -> HIT, via ``_c4dec_roles``); the [C4DEC] ``start==0``
    test is only a secondary fallback.  Per shared (layer, page):

      * produced MISS==HIT but stored MISS!=HIT  -> OUTPUT-SLOT
        (scatter / c4_loc misplacement: the right value lands in the wrong slot)
      * produced MISS!=HIT                       -> CONTENT, sub-split below
      * else (all identical)                     -> SAME

    ``n_shared`` is counted over the TRANSLATED (page) keys.  The stored side is
    taken from the SAME ``[IDXK]`` pool the ``[IDXK]`` section groups: any shared
    page that section found divergent (``divergent=`` / ``first_block=``) is
    REUSED (``_blkx_divergent``) as a stored divergence, so ``stored_diff`` cannot
    miss a page such as 136 even when the per-request stored maps do not line up
    with the ``[C4DEC]`` request indices.  CONTENT sub-split
    via [C4DEC] ``kbase`` / ``rdcarry``:
      kbase differs                          -> CONTENT:K-SPLIT
      kbase same + rdcarry differs           -> CONTENT:READ-MISALIGN (ReadState)
      both same                              -> producer-internal (CONTENT)
    """
    print("\n== [C4DEC] produced idxK vs stored [IDXK] blkx (absolute page) ==")
    if not rows:
        print("  [C4DEC-VERDICT] no [C4DEC] rows (dump gate off?)")
        print(C4DEC_RUNBOOK)
        return
    c4_reqs = _c4dec_split(rows)
    miss_idx, hit_idx, role_src = _c4dec_roles(c4_reqs, idxin_rows)
    if not miss_idx or not hit_idx:
        print(f"  [C4DEC-VERDICT] miss_req={miss_idx} hit_req={hit_idx} "
              f"(source={role_src}) "
              f"-> need >=1 MISS and >=1 HIT; none")
        print(C4DEC_RUNBOOK)
        return

    idxk_groups = _c4dec_split(idxk_rows) if idxk_rows else []
    # REUSE the [IDXK] per-page divergence instead of a SECOND, independent
    # keying of the stored pool.  ``_blkx_divergent`` applies the exact criterion
    # ``report_blkx`` prints as ``divergent=`` (a ``(layer, page)`` group with >1
    # distinct hash); the stored side is that SAME pool, so a page the [IDXK]
    # section already found divergent (e.g. page 136) MUST make stored_diff True
    # even when the per-request stored maps do not line up with the [C4DEC]
    # request indices.
    idxk_div = _blkx_divergent(idxk_rows, "blkx", keep=_is_prefill)

    def _stored(i):
        return (_c4dec_stored_maps(idxk_groups[i])
                if 0 <= i < len(idxk_groups) else {})

    def _produced_pages(idxk, ratio):
        """Produced ``{page: {md5, ...}}`` keyed by the stored-page space.

        Several produced blocks may land in the SAME page (4382 and 4383 both ->
        page 136), so the per-page produced signature is the SET of their hashes.
        ``ratio`` comes from the [C4DEC] row (default 4); a future probe with more
        ``n_keep`` blocks is handled because every produced block is translated.
        """
        pages = defaultdict(set)
        for b, h in idxk.items():
            pages[_c4dec_block_page(b, ratio)].add(h)
        return pages

    miss_prod = _c4dec_produced_maps(c4_reqs[miss_idx[0]])
    stored_miss = _stored(miss_idx[0])
    hit_prod = [_c4dec_produced_maps(c4_reqs[i]) for i in hit_idx]
    hit_stored = [_stored(i) for i in hit_idx]

    n_shared = 0
    produced_same = True
    stored_diff = False
    kbase_diff = False
    rdcarry_diff = False
    rows_out = []
    for ly in sorted(miss_prod):
        pm_idxk, pm_kbase, pm_rd, pm_ratio = miss_prod[ly]
        pm_pages = _produced_pages(pm_idxk, pm_ratio)
        sm_map = stored_miss.get(ly, {})
        for hj, hm in enumerate(hit_prod):
            if ly not in hm:
                continue
            ph_idxk, ph_kbase, ph_rd, ph_ratio = hm[ly]
            ph_pages = _produced_pages(ph_idxk, ph_ratio)
            if pm_kbase != ph_kbase:
                kbase_diff = True
            if pm_rd != ph_rd:
                rdcarry_diff = True
            sh_map = hit_stored[hj].get(ly, {})
            for page in sorted(set(pm_pages) & set(ph_pages)):
                n_shared += 1
                p_same = pm_pages[page] == ph_pages[page]
                if not p_same:
                    produced_same = False
                s_m = sm_map.get(page)
                s_h = sh_map.get(page)
                # A shared page the [IDXK] section found divergent is a STORED
                # divergence by construction -- take it from idxk_div, not from
                # the (possibly index-misaligned) per-request stored maps, and
                # make both hashes visible in the row below.
                div_hashes = idxk_div.get((ly, page))
                if div_hashes is not None:
                    stored_diff = True
                    if s_m is None:
                        s_m = div_hashes
                    if s_h is None:
                        s_h = div_hashes
                elif s_m is not None and s_h is not None and s_m != s_h:
                    stored_diff = True
                rows_out.append((ly, page, sorted(pm_pages[page]),
                                 sorted(ph_pages[page]), s_m, s_h, p_same,
                                 div_hashes))

    if not produced_same:
        if kbase_diff:
            verdict = "CONTENT:K-SPLIT"
        elif rdcarry_diff:
            verdict = "CONTENT:READ-MISALIGN"
        else:
            verdict = "CONTENT"
    elif stored_diff:
        verdict = "OUTPUT-SLOT"
    else:
        verdict = "SAME"

    print(f"  [C4DEC-VERDICT] miss_req={miss_idx} hit_req={hit_idx} "
          f"(source={role_src}) "
          f"n_shared={n_shared} produced_same={produced_same} "
          f"stored_diff={stored_diff} kbase_diff={kbase_diff} "
          f"rdcarry_diff={rdcarry_diff} => {verdict}")

    shown = 0
    for (ly, page, pm, ph, s_m, s_h, p_same, div_hashes) in rows_out:
        if verdict == "OUTPUT-SLOT":
            # Show the reused [IDXK] divergent page too -- it is the evidence
            # that the stored side differs, even absent an aligned per-side map.
            keep = div_hashes is not None or (
                p_same and s_m is not None and s_h is not None and s_m != s_h)
        elif verdict.startswith("CONTENT"):
            keep = not p_same
        else:
            keep = False
        if not keep:
            continue
        print(f"  [C4DEC-VERDICT] layer={ly} page={page} prod_miss={pm} "
              f"prod_hit={ph} stored_miss={s_m} stored_hit={s_h}")
        shown += 1
        if shown >= MAX_OFFENDER_SHOWN:
            break
    print(C4DEC_RUNBOOK)


def _parse_kstateloc(recs):
    """Split raw [KSTATELOC] rows into (write op=W, read op=R) numeric dicts."""
    writes, reads = [], []
    for rec in recs:
        raw = rec.get("_raw", "")
        m = KSTATELOC_W.search(raw)
        if m:
            writes.append({
                "b": int(m.group(1)), "seq": int(m.group(2)),
                "col": int(m.group(3)), "sl": int(m.group(4)),
                "blk": int(m.group(5)), "row": int(m.group(6)),
                "out": int(m.group(7)),
            })
            continue
        m = KSTATELOC_R.search(raw)
        if m:
            reads.append({
                "b": int(m.group(1)), "seq": int(m.group(2)),
                "col": int(m.group(3)), "sl": int(m.group(4)),
                "blk": int(m.group(5)), "row": int(m.group(6)),
            })
    return writes, reads


def _kstateloc_focus(writes):
    """``(lo, hi)`` window to show: a known tail window, else the first block."""
    seqs = sorted({x["seq"] for x in writes})
    if not seqs:
        return None
    for lo, hi in C4STP_NUM_WINDOWS:
        if any(lo <= s < hi for s in seqs):
            return (lo, hi)
    return (seqs[0], seqs[0] + 24)


def report_kstateloc(rows):
    """Print ``out=`` (op=W) / stateLoc (op=R) so write-loc can be judged.

    The write output anchor ``out`` is the compressed position the compressor
    committed.  It must ADVANCE (roughly once per ``cmp_ratio`` decode steps)
    as the request grows; a FROZEN ``out`` across advancing write positions is
    the write-loc bug.  ``op=R`` stateLoc is printed for the same positions so
    the read side can be compared against the write.
    """
    if not rows:
        return
    writes, reads = _parse_kstateloc(rows)
    if not writes and not reads:
        print("\n== [KSTATELOC] no rows (dump gate off) ==")
        return
    print("\n== [KSTATELOC] compressor state write(op=W)/read(op=R) loc ==")
    if writes:
        seqs = sorted({x["seq"] for x in writes})
        print(f"  [KSTATELOC-W] calls={len(writes)} distinct_seq={len(seqs)} "
              f"seq_range=[{seqs[0]}..{seqs[-1]}] "
              f"out_vals={sorted({x['out'] for x in writes})}")
    if reads:
        seqs = sorted({x["seq"] for x in reads})
        print(f"  [KSTATELOC-R] calls={len(reads)} distinct_seq={len(seqs)} "
              f"seq_range=[{seqs[0]}..{seqs[-1]}]")
    win = _kstateloc_focus(writes) if writes else None
    if win is not None:
        lo, hi = win
        print(f"  [KSTATELOC-W] focus window seq in [{lo},{hi}):")
        shown = 0
        for x in writes:
            if lo <= x["seq"] < hi:
                print(f"     seq={x['seq']} out={x['out']} sl={x['sl']} "
                      f"blk={x['blk']} row={x['row']} col={x['col']}")
                shown += 1
                if shown >= 60:
                    print("     ... (truncated)")
                    break
        rmatch = [x for x in reads if lo <= x["seq"] < hi]
        if rmatch:
            print(f"  [KSTATELOC-R] same positions seq in [{lo},{hi}):")
            for x in rmatch[:60]:
                print(f"     seq={x['seq']} sl={x['sl']} blk={x['blk']} "
                      f"row={x['row']} col={x['col']}")
        else:
            print("  [KSTATELOC-R] none for focus window (op=R not emitted "
                  "at these positions)")
    if writes:
        _kstateloc_advance(writes)


def _kstateloc_advance(writes):
    """Does ``out`` advance per write call, or stay frozen (write-loc bug)?"""
    per_seq = defaultdict(set)
    for x in writes:
        per_seq[x["seq"]].add(x["out"])
    multi = sorted(s for s, v in per_seq.items() if len(v) > 1)
    ladder = sorted((s, next(iter(per_seq[s])))
                    for s in per_seq if len(per_seq[s]) == 1)
    advances = sum(1 for i in range(1, len(ladder))
                   if ladder[i][1] != ladder[i - 1][1])
    steps = max(len(ladder) - 1, 0)
    distinct_out = len({x["out"] for x in writes})
    if multi:
        verdict = "WRITE-LOC inconsistent (same seq -> multiple out)"
    elif steps and advances == 0:
        verdict = "WRITE-LOC FROZEN (out never advances)"
    else:
        verdict = f"WRITE-LOC advances ({advances}/{steps} steps)"
    multi_txt = ",".join(str(s) for s in multi[:20]) or "none"
    print(f"  [KSTATELOC-W] VERDICT {verdict}  multi_out_seq=[{multi_txt}] "
          f"distinct_out={distinct_out} distinct_seq={len(per_seq)}")


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


C4STP_RUNBOOK = (
    "[C4STP-RUNBOOK] DSV4_DUMP_C4ST=2 bash start_test.sh > /tmp/c4st.log 2>&1 & "
    "bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh; bash curl.sh; "
    "python scripts/analyze_dsv4_cmpidx.py /tmp/c4st.log"
)
C4STP_CARRY_POS = tuple(range(16380, 16384))
C4STP_BLOCK136_STEPS = (17532, 17533, 17534, 17535)
C4STP_NUM_MAX_POS = 12
C4STP_NUM_WINDOWS = ((16380, 16384), (17516, 17536))


def _c4stp_norm(rec):
    """Normalize one raw [C4STP] field-dict, or None when it is unusable.

    Needs ``tag`` in {pre,post}, a non-empty ``md5`` and integer ``pos``/``sloc``.
    ``layer``/``idx``/``start`` default when the probe line omits them, so partial
    or older dumps still parse.
    """
    tag = rec.get("tag")
    if tag not in ("pre", "post"):
        return None
    md5 = rec.get("md5")
    if not md5 or md5 == "empty":
        return None
    pos = _int(rec, "pos", None)
    sloc = _int(rec, "sloc", None)
    if pos is None or sloc is None:
        return None
    return {
        "layer": _int(rec, "layer", -1),
        "idx": rec.get("idx", ""),
        "start": _c4st_start(rec),
        "pos": pos,
        "sloc": sloc,
        "md5": md5,
        "tag": tag,
        "v0": _float(rec, "v0"),
        "sum": _float(rec, "sum"),
        "absmax": _float(rec, "absmax"),
    }


def _c4stp_low_slots(slocs):
    """Lower half of two DISJOINT ``sloc`` regions, else None.

    ``sloc`` is a state-slot address: the MISS (freshly computed) and HIT (reused
    prefix) streams live in separate slot regions, so the boundary is the widest
    gap.  A single stream scans a contiguous slot run whose gaps are all small, so
    the boundary only counts when the widest gap clearly dominates the rest --
    otherwise None and the caller falls back to ``start``.
    """
    uniq = sorted(set(slocs))
    if len(uniq) < 2:
        return None
    gaps = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    top = max(gaps)
    rest = max([g for g in gaps if g != top] or [0])
    if top < 2 or top < 2 * rest:
        return None
    return set(uniq[: gaps.index(top) + 1])


def _c4stp_sides(rows):
    """Split one (idx,tag) pool into (miss_records, hit_records).

    Rows are clustered by ``start`` and each cluster is cut at the widest ``sloc``
    gap into a low (MISS) and high (HIT) region: this separates the two streams
    even when they share a ``start``.  Only when NO cluster shows a gap do we fall
    back to the smallest/largest ``start`` -- never rely on start alone.
    """
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["start"]].append(r)
    miss, hit, split = [], [], False
    for group in buckets.values():
        low = _c4stp_low_slots([r["sloc"] for r in group])
        if low is None:
            continue
        split = True
        for r in group:
            (miss if r["sloc"] in low else hit).append(r)
    if split:
        return miss, hit
    starts = sorted({r["start"] for r in rows if r["start"] is not None})
    if len(starts) >= 2:
        for r in rows:
            if r["start"] == starts[0]:
                miss.append(r)
            elif r["start"] == starts[-1]:
                hit.append(r)
    return miss, hit


def _c4stp_by_pos(records):
    """{pos: md5} for one side (first md5 wins on a duplicate pos)."""
    out = {}
    for r in records:
        out.setdefault(r["pos"], r["md5"])
    return out


def _c4stp_region(md5_map, positions):
    """Hex16 over the per-position md5s present in ``positions`` (or None)."""
    parts = [f"{p}:{md5_map[p]}" for p in positions if p in md5_map]
    if not parts:
        return None
    return hashlib.md5("|".join(parts).encode("ascii")).hexdigest()[:16]


def _c4stp_cmp(miss_map, hit_map, pos):
    """SAME/DIFF for ``pos`` when both sides carry it, else None."""
    a, b = miss_map.get(pos), hit_map.get(pos)
    if a is None or b is None:
        return None
    return "SAME" if a == b else "DIFF"


def _c4stp_num_report(miss_recs, hit_recs):
    """Compact numeric MISS-vs-HIT comparison of the c4-indexer state.

    Additive to ``verdict_c4stp``: for the positions present in BOTH streams it
    diffs the ``v0``/``sum``/``absmax`` float fields and prints ONE block --
    header + at most 12 per-position lines (boundary windows first) + the MAG
    summary + a single landing VERDICT.  Prints ``[C4STP-MAG] no numeric data``
    when neither side carries numeric fields.
    """

    def _by_pos(records):
        out = {}
        for r in records:
            slot = out.setdefault(
                r["pos"], {"v0": None, "sum": None, "absmax": None})
            for k in ("v0", "sum", "absmax"):
                if slot[k] is None and r.get(k) is not None:
                    slot[k] = r[k]
        return {p: s for p, s in out.items()
                if any(v is not None for v in s.values())}

    def _sub(a, b):
        return None if a is None or b is None else a - b

    def _efmt(v):
        return f"{v:.3e}" if v is not None else "None"

    def _afmt(v):
        return f"{abs(v):.3e}" if v is not None else "None"

    def _prio(p):
        for lo, hi in C4STP_NUM_WINDOWS:
            if lo <= p < hi:
                return (0, p)
        return (1, p)

    miss, hit = _by_pos(miss_recs), _by_pos(hit_recs)
    shared = sorted(set(miss) & set(hit))
    if not shared:
        print("[C4STP-MAG] no numeric data")
        return

    diffs, nzero = [], 0
    max_dsum = max_dabs = max_dv0 = 0.0
    for pos in shared:
        m, h = miss[pos], hit[pos]
        dsum = _sub(m["sum"], h["sum"])
        dabs = _sub(m["absmax"], h["absmax"])
        dv0 = _sub(m["v0"], h["v0"])
        if dsum is not None:
            max_dsum = max(max_dsum, abs(dsum))
        if dabs is not None:
            max_dabs = max(max_dabs, abs(dabs))
        if dv0 is not None:
            max_dv0 = max(max_dv0, abs(dv0))
        present = [d for d in (dsum, dabs, dv0) if d is not None]
        if present and any(d != 0.0 for d in present):
            diffs.append((pos, m, h, dsum, dabs, dv0))
        else:
            nzero += 1

    print("===== [C4STP-NUM] miss-vs-hit numeric (paste this) =====")
    if diffs:
        diffs.sort(key=lambda t: _prio(t[0]))
        for pos, m, h, dsum, dabs, dv0 in diffs[:C4STP_NUM_MAX_POS]:
            print(f"pos={pos} miss_sum={_efmt(m['sum'])} "
                  f"hit_sum={_efmt(h['sum'])} |dsum|={_afmt(dsum)} "
                  f"|dabsmax|={_afmt(dabs)} |dv0|={_afmt(dv0)}")
    else:
        print(f"pos={min(shared, key=_prio)} identical")
    print(f"[C4STP-MAG] npos={len(shared)} nzero={nzero} "
          f"max|dsum|={max_dsum:.3e} max|dabsmax|={max_dabs:.3e} "
          f"max|dv0|={max_dv0:.3e}")
    if max_dsum == 0.0 and max_dabs == 0.0 and max_dv0 == 0.0:
        verdict = "IDENTICAL"
    elif 0.0 < max_dabs < 1e-3:
        verdict = "NUMERIC"
    else:
        verdict = "LOGIC"
    print(f"[C4STP-MAG] VERDICT={verdict}")
    print("===== end =====")


def verdict_c4stp(recs):
    """Per-position [C4STP] MISS-vs-HIT verdict + one binary-rule line.

    Splits the records into the two streams (see ``_c4stp_sides``), reports the
    carried boundary (pos 16380..16383) and the block-136 commit steps, then
    prints ONE rule line mapping the divergence to its root cause::

        carry DIFF                         -> READ
        first block136 step with content=DIFF -> WRITE (post-only probe included)
        carry/block136 data but no diff    -> NON_STATE
        pre streams already DIFF (no data) -> UPSTREAM

    Additive only: it never touches the [C4ST] report above.
    """
    rows = [n for n in (_c4stp_norm(r) for r in recs) if n is not None]
    if not rows:
        print("[C4STP-BINRULE] no C4STP data")
        print("[C4STP-MAG] no numeric data")
        print(C4STP_RUNBOOK)
        return
    # Focus on the indexer state (idx=1) when the probe emitted it, like [C4ST].
    if any(r["idx"] == "1" for r in rows):
        rows = [r for r in rows if r["idx"] == "1"]

    miss_by_tag, hit_by_tag = {}, {}
    miss_recs_by_tag, hit_recs_by_tag = {}, {}
    for tag in ("pre", "post"):
        pool = [r for r in rows if r["tag"] == tag]
        miss, hit = _c4stp_sides(pool) if pool else ([], [])
        miss_by_tag[tag] = _c4stp_by_pos(miss)
        hit_by_tag[tag] = _c4stp_by_pos(hit)
        miss_recs_by_tag[tag] = miss
        hit_recs_by_tag[tag] = hit

    # The tag that actually pairs positions (prefer post = committed state).
    def _pairs(tag):
        return len(set(miss_by_tag[tag]) & set(hit_by_tag[tag]))

    tag = max(("post", "pre"), key=_pairs)
    miss_map, hit_map = miss_by_tag[tag], hit_by_tag[tag]

    carry = "N/A"
    if any(p in miss_map and p in hit_map for p in C4STP_CARRY_POS):
        m = _c4stp_region(miss_map, C4STP_CARRY_POS)
        h = _c4stp_region(hit_map, C4STP_CARRY_POS)
        carry = "SAME" if m == h else "DIFF"
        print(f"[C4STP-VERDICT] carry pos=<16380..16383> miss={m} hit={h} "
              f"content={carry}")

    for step in C4STP_BLOCK136_STEPS:
        if step in miss_map and step in hit_map:
            m, h = miss_map[step], hit_map[step]
            content = "SAME" if m == h else "DIFF"
            print(f"[C4STP-VERDICT] block136 step={step} miss={m} hit={h} "
                  f"content={content}")

    pre_m, pre_h = miss_by_tag["pre"], hit_by_tag["pre"]

    # First block136 step (sorted) whose paired MISS-vs-HIT content diverges.
    # The COMMIT-ROW probe only emits ``tag=post``, so this must not depend on a
    # pre/post pair at the same step.
    first_diff_step, block136_seen = None, False
    for step in sorted(C4STP_BLOCK136_STEPS):
        cmp = _c4stp_cmp(miss_map, hit_map, step)
        if cmp is not None:
            block136_seen = True
        if cmp == "DIFF" and first_diff_step is None:
            first_diff_step = step

    # UPSTREAM probe: both ``pre`` streams already diverged at a block136 step.
    pre_diff = any(_c4stp_cmp(pre_m, pre_h, step) == "DIFF"
                   for step in sorted(C4STP_BLOCK136_STEPS))

    has_state_data = carry != "N/A" or block136_seen
    if carry == "DIFF":
        verdict = "READ"
    elif first_diff_step is not None:
        verdict = "WRITE"
    elif has_state_data:
        verdict = "NON_STATE"
    elif pre_diff:
        verdict = "UPSTREAM"
    else:
        verdict = "NON_STATE"
    first_txt = "NONE" if first_diff_step is None else str(first_diff_step)
    print(f"[C4STP-BINRULE] carry={carry} "
          f"first_commit_diff_step={first_txt} verdict={verdict}")
    _c4stp_num_report(miss_recs_by_tag[tag], hit_recs_by_tag[tag])
    print(C4STP_RUNBOOK)


def main(path):
    recs = parse(path)
    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA", "C4ST", "C4STP", "KSTATELOC", "C4POS", "C4DEC", "WSTAGE"):
        print(f"parsed {tag}: {len(recs.get(tag, []))} lines")
    if recs.get("CMPIDX"):
        cmp_reqs = segment(recs["CMPIDX"])
        _ntoks = [(rq["ntok0"] or 0) for rq in cmp_reqs]
        _mx = max(_ntoks) if _ntoks else 0
        # MISS = ALL requests at the largest prefill ntok. The harness sends N
        # identical full-prefill MISS then 1 suffix HIT; the old single-max rule
        # marked only ONE MISS and labeled the other N-1 MISS as HIT.
        miss_set = {i for i, n in enumerate(_ntoks) if n == _mx}
        hit_set = {i for i in range(len(cmp_reqs))} - miss_set
        _note = f"ALL largest prefill ntok={_mx}"
        if not hit_set:
            # ntok identical across ALL requests (or missing) -> harness
            # convention: the LAST request is the suffix HIT, earlier are MISS.
            miss_set = set(range(len(cmp_reqs) - 1))
            hit_set = {len(cmp_reqs) - 1}
            _note = ("ntok cannot separate requests -> harness convention "
                     "(LAST=HIT)")
        print("\n== requests (classified by [CMPIDX] prefill ntok) ==")
        for i, rq in enumerate(cmp_reqs[:MAX_REQ_SHOWN]):
            role = "MISS" if i in miss_set else "HIT "
            print(f"  req{i:<2} {role} prefill_ntok={rq['ntok0']}"
                  f"  steps={len(rq['steps'])}")
        if len(cmp_reqs) > MAX_REQ_SHOWN:
            print(f"  ... {len(cmp_reqs) - MAX_REQ_SHOWN} more requests")
        print(f"\nMISS = req{sorted(miss_set)} ({_note}); each HIT compared to them.")
        miss_i = min(miss_set) if miss_set else 0
    else:
        # No [CMPIDX]: ntok is unavailable, so use the harness convention --
        # N full-prefill MISS followed by ONE suffix HIT => the LAST request is
        # the HIT, every earlier request is a MISS.  (The old "req0 = MISS" rule
        # mislabeled req1..req{N-1} -- which are ALSO MISS -- as HIT.)
        cmp_reqs = None
        _nreq = 0
        for _t in ("IDXK", "C4POS", "LHID", "C4DEC", "WSTAGE", "XIN", "LIMETA",
                   "OSHAPE", "C4KV", "C128KV"):
            if recs.get(_t):
                _nreq = len(segment(recs[_t]))
                break
        miss_i = 0
        hit_set = {_nreq - 1} if _nreq >= 1 else set()
        print(f"\nno [CMPIDX] lines; split by harness convention: "
              f"req0..req{max(0, _nreq - 2)} MISS (same-side), "
              f"req{max(0, _nreq - 1)} HIT   [{_nreq} requests].")

    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA", "C4ST", "KSTATELOC", "C4POS", "C4DEC", "WSTAGE"):
        rows = recs.get(tag, [])
        if tag == "WSTAGE":
            # Always-emit: real verdict, or the no-data runbook when the dump
            # gate produced no rows. Needs the [C4POS]/[IDXIN] rows for the
            # MISS/HIT split (same rule as the rest of the analyzer).
            report_wstage(rows, recs.get("C4POS", []), recs.get("IDXIN", []))
            continue
        if tag == "C4POS":
            # Emit this section ALWAYS (real verdict, or the no-data fallback
            # when the dump gate produced no rows), so the [C4POS] block is
            # visible on the executed main() path even with zero rows.
            report_c4pos(rows)
            continue
        if tag == "C4DEC":
            # Same always-emit rule: needs the [IDXK] rows for the stored side.
            report_c4dec(rows, recs.get("IDXK", []), recs.get("IDXIN", []))
            continue
        if not rows:
            continue
        if tag == "KSTATELOC":
            report_kstateloc(rows)
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
            report_uniq(rows, "IDXIN.keymeta",
                        ("klog", "kpg", "kraw", "meta", "bt"))
            report_idxin_requests(rows)
            report_idxin_verdict(rows)
            print("\n== [IDXIN-ROPE] RoPE-phase vs content/window (knorm invariant) ==")
            report_idxin_rope(rows)
            print("\n== [QK-ALIGNED] query vs index-K at P* (decisive q-vs-K split) ==")
            report_qk_aligned(rows, recs.get("IDXK", []), recs.get("C4POS", []))
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
            print("  ---- [LHID] per-absolute-position single row ----")
            report_lhid_pos(rows)
            report_lhid_accept(rows, recs.get("C4POS", []),
                               recs.get("IDXIN", []))
            print("  ---- [LHID] per-c128-block ----")
            report_blkx(rows, "LHID")
            continue
        if tag == "XIN":
            print("\n== [XIN] c4-indexer input x (order-independent) ==")
            report_xin_pos(rows, recs.get("C4POS", []), recs.get("IDXIN", []))
            # fall through to the whole-tensor segment compare below as well
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
            report_cmpidx_div(rows, cmp_reqs, miss_i)
            report_cmpidx_aligned(
                rows, recs.get("C4POS", []), recs.get("IDXIN", []))
            report_cmpidx_ordered(
                rows, recs.get("C4POS", []), recs.get("IDXIN", []))
            # fall through to the whole-tensor segment compare below as well
        if tag == "IDXK":
            print("\n== [IDXK] c4 index-K (per-logical-page, prefill) ==")
            report_blkx(rows, "IDXK", keep=_is_prefill)
            report_idxk_aligned(rows)
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
            _role = "HIT " if (not hit_set or i in hit_set) else "MISS"
            print(f"  MISS(req{miss_i}) vs {_role}(req{i}): lastpos={n}"
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
    verdict_c4stp(recs.get("C4STP", []))


def _selftest():
    """Exercise the new ``kpg=`` / ``out=`` parsing + verdict branches.

    Builds a synthetic log with one SAME probe (layer 5) and one READ-PAGE
    divergent probe (layer 7, same klog / different kpg) plus [KSTATELOC] rows,
    then asserts both the raw parse and the printed verdict strings.  ALSO builds
    two synthetic logs WITHOUT any [CMPIDX] (the ">2 requests" fallback path) and
    asserts the ``(mode,slq,slk)`` stream split reaches both the ROPE-PHASE and
    the CONTENT/WINDOW [IDXIN-ROPE] branches.  Finally, TWO same-slq-pairing
    cases: C1 has a same-slq HIT (knorm ~equal) and asserts the MISS is paired
    with THAT HIT (ROPE-PHASE, not a false CONTENT/WINDOW from the first short
    HITs); C2 has NO same-slq HIT and asserts the fallback is stated.  FINALLY,
    D1 builds a synthetic log with FIVE ``/v1/chat/completions`` boundary lines
    and asserts the boundary split + the ``sps`` rule label requests 1..4 MISS
    and 5 HIT.  E1/E2 build per-BLOCK ``knorm`` and assert the comparison judges
    ONLY shared blocks: E1 (shared equal, one extra MISS block) -> ROPE-PHASE; E2
    (a shared block differs) -> CONTENT/WINDOW.  F1 is the real shape: five
    requests where 0..3 are MISS (``sps=0``) and 4 is HIT (``sps=16384``); it
    asserts ``[IDXIN-SPLIT] source=sps ... miss_req=[0, 1, 2, 3] hit_req=[4]``,
    a self-consistent MISS set, and a ROPE-PHASE shared-block verdict.
    """
    import contextlib
    import io
    import os
    import tempfile

    kpg_a = "n=8,first=[3, 5, 7, 9],last=[11, 13, 15, 17]"
    kpg_b = "n=8,first=[1, 2, 3, 4],last=[5, 6, 7, 8]"

    def _idxin(ly, slq, klog, kpg, knorm="1.0:100:64", shift=0, npb=100,
               slk=17523, with_sps=True, qpos=None):
        sps = f"sps=[{shift * 4}]," if with_sps else ""
        q = f" qpos={qpos}" if qpos is not None else ""
        return (
            f"[IDXIN] layer={ly} mode=1 qblk=[0:aa] wblk=[0:bb] "
            f"slq=({slq},{slq},1) slk=({slk},{slk},1) klog={klog} kpg={kpg} "
            f"knorm={knorm} ropephase=r=4,{sps}shift=[{shift}],"
            f"npb=[{npb}],loc=[0..{npb - 1}],abs=[{shift}..{shift + npb - 1}] "
            f"kraw=kk meta=mm bt=tt qshape=(1,2,3) wshape=(1,2,3) "
            f"kshape=(4,5,6){q}"
        )

    lines = [
        # MISS request (prefill slq=17523) spans layers 5/7/9, lastpos 100 then 101.
        "[CMPIDX] layer=5 lastpos=100 ntok=17523 md5=ffff",
        _idxin(5, 17523, "aaaa", kpg_a),
        "[CMPIDX] layer=7 lastpos=100 ntok=17523 md5=ffff",
        _idxin(7, 17523, "zzzz", kpg_a),
        "[CMPIDX] layer=9 lastpos=100 ntok=17523 md5=ffff",
        _idxin(9, 17523, "mmmm", kpg_a),
        "[CMPIDX] layer=5 lastpos=101 ntok=17523 md5=ffff",
        _idxin(5, 17523, "aaaa", kpg_a, shift=4096),
        "[CMPIDX] layer=7 lastpos=101 ntok=17523 md5=ffff",
        _idxin(7, 17523, "zzzz", kpg_a, shift=4096),
        # HIT request (suffix prefill slq=1139): lastpos drops to 100 -> new req.
        #   layer 5: same (klog,kpg) as MISS               -> SAME
        #   layer 7: same klog, DIFFERENT kpg              -> READ-PAGE misaligned
        #   layer 9: DIFFERENT klog, same knorm (invariant) -> ROPE-PHASE
        "[CMPIDX] layer=5 lastpos=100 ntok=1139 md5=ffff",
        _idxin(5, 1139, "aaaa", kpg_a, shift=4096),
        "[CMPIDX] layer=7 lastpos=100 ntok=1139 md5=ffff",
        _idxin(7, 1139, "zzzz", kpg_b, shift=4096),
        "[CMPIDX] layer=9 lastpos=100 ntok=1139 md5=ffff",
        _idxin(9, 1139, "hhhh", kpg_a, shift=4096),
        # [KSTATELOC] write/read loc
        "[KSTATELOC] op=W 0 17520 -12 40 0 40 out=17520",
        "[KSTATELOC] op=W 0 17521 -11 41 0 41 out=17520",
        "[KSTATELOC] op=W 0 17524 -8 44 0 44 out=17524",
        "[KSTATELOC] op=R 0 17520 -12 40 0 40",
    ]
    fd, path = tempfile.mkstemp(suffix=".log", prefix="cmpidx_selftest_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        recs = parse(path)
        idxin = recs.get("IDXIN", [])
        ks = recs.get("KSTATELOC", [])
        writes, reads = _parse_kstateloc(ks)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_idxin_requests(idxin)
            report_idxin_verdict(idxin)
            report_idxin_rope(idxin)
            report_kstateloc(ks)
        out = buf.getvalue()

        # --- no-[CMPIDX] fallback: split [IDXIN] purely from its own stream ---
        # B1: NO sps -> largest-slq fallback; same knorm on the SHARED block
        #     (shift 0) + differing klog -> ROPE-PHASE.
        nocmp_rope = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="1.0:100:64", shift=0,
                   with_sps=False),
            _idxin(5, 1139, "cccc", kpg_a, knorm="1.0:100:64", shift=0,
                   with_sps=False),
        ]
        # B2: shared block (shift 0) knorm differs (rel=0.5 > tol=0.05)
        #     -> CONTENT/WINDOW.
        nocmp_content = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="1.0:100:64", shift=0,
                   with_sps=False),
            _idxin(5, 1139, "aaaa", kpg_a, knorm="2.0:100:64", shift=0,
                   with_sps=False),
        ]
        # C1: NO sps -> same-slq fallback.  A same-slq HIT EXISTS (knorm ~equal,
        #     198.935 vs 199.0 on the SAME shared block) -> must pair the MISS
        #     with THAT HIT and print ROPE-PHASE -- NOT with the first, short-slq
        #     HITs (knorm 6.7/12.7) which would fake CONTENT/WINDOW.
        same_slq = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0,
                   with_sps=False),
            _idxin(5, 10, "xx", kpg_a, knorm="6.71568:100:64", shift=1, slk=10,
                   with_sps=False),
            _idxin(5, 12, "yy", kpg_a, knorm="12.7265:100:64", shift=1, slk=12,
                   with_sps=False),
            _idxin(5, 17523, "bbbb", kpg_a, knorm="199.0:100:64",
                   shift=0, slk=17523, with_sps=False),
        ]
        # C2: NO same-slq HIT -> fall back to all HITs AND SAY SO; shared block
        #     knorm equal -> still ROPE-PHASE (the fallback is stated).
        no_same_slq = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0,
                   with_sps=False),
            _idxin(5, 10, "xx", kpg_a, knorm="198.935:100:64", shift=0, slk=10,
                   with_sps=False),
        ]
        # E1: per-BLOCK knorm with the sps rule.  Shared shifts {4096,4097} are
        #     EQUAL; the MISS (sps=0) also carries an extra block (shift 0,
        #     12.73) the HIT lacks -> the extra block must NOT be judged
        #     => ROPE-PHASE (a whole-request value-SET compare would fake
        #     CONTENT/WINDOW).
        per_shift_rope = [
            _idxin(2, 17523, "aaaa", kpg_a, knorm="12.73:1:64", shift=0),
            _idxin(3, 17523, "aaaa", kpg_a, knorm="197.7:1:64", shift=4096),
            _idxin(4, 17523, "aaaa", kpg_a, knorm="199.0:1:64", shift=4097),
            _idxin(2, 17523, "bbbb", kpg_a, knorm="197.7:1:64", shift=4096),
            _idxin(3, 17523, "bbbb", kpg_a, knorm="199.0:1:64", shift=4097),
        ]
        # E2: per-BLOCK knorm.  Shared blocks (4096/4097) DIFFER -> CONTENT/WINDOW.
        per_shift_content = [
            _idxin(2, 17523, "aaaa", kpg_a, knorm="12.73:1:64", shift=0),
            _idxin(3, 17523, "aaaa", kpg_a, knorm="197.7:1:64", shift=4096),
            _idxin(4, 17523, "aaaa", kpg_a, knorm="199.0:1:64", shift=4097),
            _idxin(2, 17523, "bbbb", kpg_a, knorm="12.73:1:64", shift=4096),
            _idxin(3, 17523, "bbbb", kpg_a, knorm="250.0:1:64", shift=4097),
        ]
        # F1: the REAL shape -- the harness fires 5 identical requests; req0..3
        #     MISS (sps=0), req4 HIT (sps=16384).  The sps rule must label 0-3
        #     MISS / 4 HIT and the shared block (shift 4096) must give ROPE-PHASE.
        per_sps = [
            _idxin(0, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(1, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            _idxin(0, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(1, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            _idxin(0, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(1, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            _idxin(0, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(1, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            _idxin(0, 17523, "bbbb", kpg_a, knorm="199.0:100:64", shift=4096),
            _idxin(1, 17523, "bbbb", kpg_a, knorm="199.0:100:64", shift=4096),
        ]
        # D1: /v1/chat/completions HTTP boundaries + the sps rule.  req1..4 are
        #     identical MISSes (sps=0) and req5 the HIT (sps=16384) sharing
        #     block 4096 -> ROPE-PHASE.
        CHAT = ('INFO:     127.0.0.1:0 - "POST /v1/chat/completions '
                'HTTP/1.1" 200 OK')
        chat_split = [
            CHAT,  # request 1: MISS
            _idxin(2, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(3, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            CHAT,  # request 2: MISS
            _idxin(2, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(3, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            CHAT,  # request 3: MISS
            _idxin(2, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(3, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            CHAT,  # request 4: MISS
            _idxin(2, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(3, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=4096),
            CHAT,  # request 5: HIT (sps=16384), shares block 4096
            _idxin(2, 17523, "bbbb", kpg_a, knorm="199.0:100:64",
                   shift=4096, slk=17523),
        ]
        # G1: [C4POS] absolute-block parity.  MISS anchor 0 (blocks 0..7), HIT
        #     anchor 16 = block 4 (blocks 4..9); shared blocks 4..7 carry the
        #     SAME value -> phase_same.
        g1_pos = [
            CHAT,
            "[C4POS] layer=2 idx=1 n=12 ratio=4 nz=8 first=0 last=0 "
            "md5=aaa sum=112 pos=[0,4,8,12,16,20,24,28]",
            CHAT,
            "[C4POS] layer=2 idx=1 n=10 ratio=4 nz=6 first=16 last=0 "
            "md5=bbb sum=156 pos=[16,20,24,28,32,36]",
        ]
        # G2: same anchors, but the HIT's block-7 value is 31 (not 28) -> a
        #     SHARED block differs -> phase_diff.
        g2_pos = [
            CHAT,
            "[C4POS] layer=2 idx=1 n=12 ratio=4 nz=8 first=0 last=0 "
            "md5=aaa sum=112 pos=[0,4,8,12,16,20,24,28]",
            CHAT,
            "[C4POS] layer=2 idx=1 n=10 ratio=4 nz=6 first=16 last=0 "
            "md5=bbb sum=156 pos=[16,20,24,31]",
        ]

        def _run_nocmp(src_lines):
            """Parse + report a synthetic no-[CMPIDX] log; return (rows, out)."""
            fd2, p2 = tempfile.mkstemp(suffix=".log", prefix="cmpidx_selftest_")
            try:
                with os.fdopen(fd2, "w", encoding="utf-8") as fh2:
                    fh2.write("\n".join(src_lines) + "\n")
                irows = parse(p2).get("IDXIN", [])
                buf2 = io.StringIO()
                with contextlib.redirect_stdout(buf2):
                    report_idxin_requests(irows)
                    report_idxin_verdict(irows)
                    report_idxin_rope(irows)
                return irows, buf2.getvalue()
            finally:
                try:
                    os.remove(p2)
                except OSError:
                    pass

        def _run_c4pos(src_lines):
            """Parse + report a synthetic [C4POS] log; return (rows, out)."""
            fd3, p3 = tempfile.mkstemp(suffix=".log", prefix="cmpidx_selftest_")
            try:
                with os.fdopen(fd3, "w", encoding="utf-8") as fh3:
                    fh3.write("\n".join(src_lines) + "\n")
                crows = parse(p3).get("C4POS", [])
                buf3 = io.StringIO()
                with contextlib.redirect_stdout(buf3):
                    report_c4pos(crows)
                return crows, buf3.getvalue()
            finally:
                try:
                    os.remove(p3)
                except OSError:
                    pass

        def _run_main(src_lines):
            """Write a synthetic log and run the REAL ``main()``; return stdout.

            Unlike ``_run_c4pos`` (which calls ``report_c4pos`` directly), this
            exercises the executed ``main()`` control flow, so it catches the
            case where the [C4POS] section is defined but never reached.
            """
            fd4, p4 = tempfile.mkstemp(suffix=".log", prefix="cmpidx_selftest_")
            try:
                with os.fdopen(fd4, "w", encoding="utf-8") as fh4:
                    fh4.write("\n".join(src_lines) + "\n")
                buf4 = io.StringIO()
                with contextlib.redirect_stdout(buf4):
                    main(p4)
                return buf4.getvalue()
            finally:
                try:
                    os.remove(p4)
                except OSError:
                    pass

        # G3: the EXACT reported repro - two [C4POS] rows (one MISS anchor 0,
        #     one HIT anchor 16384) through the REAL main() path.  Before the
        #     fix main() printed only the "parsed C4POS: 2 lines" counter.
        repro_pos = [
            "[C4POS] layer=2 idx=1 n=5 ratio=4 nz=3 first=0 last=8 "
            "md5=abc sum=12 pos=[0,4,8]",
            "[C4POS] layer=2 idx=1 n=2 ratio=4 nz=2 first=16384 last=16388 "
            "md5=def sum=9 pos=[16384,16388]",
        ]
        repro_out = _run_main(repro_pos)
        # G4: a log with NO [C4POS] rows must still print the section + the
        #     "no [C4POS] rows (dump gate off?)" fallback on the main() path.
        no_c4pos_out = _run_main(["[CMPIDX] layer=1 lastpos=1 ntok=10 md5=xx"])
        # G5: the phase_same fixture through main() -> a REAL verdict (not the
        #     no-data line) reaches stdout.
        g5_out = _run_main(g1_pos)

        # --- [C4DEC] produced idxK vs stored [IDXK] blkx (3 verdict branches) ---
        _KB_A = ("mBaseNum=137/kBaseNum=1 raw_tok=17523 headDim=128"
                 "(attr=None,state=64) dBasicBlockNum=2 state_row_dim=64 "
                 "thresh=274 usedCoreNum=unset")
        _KB_B = ("mBaseNum=137/kBaseNum=2 raw_tok=17523 headDim=128"
                 "(attr=None,state=64) dBasicBlockNum=2 state_row_dim=64 "
                 "thresh=274 usedCoreNum=unset")
        _RC_A = "[17528=40:aaaa,17529=41:bbbb]"
        _RC_B = "[17528=41:cccc,17529=42:dddd]"

        def _c4dec(ly, start, kbase, rd, blk_a, blk_b):
            # Produced idxK keys are the probe's hardcoded ratio-4 tail blocks
            # 4382/4383 (BOTH land in stored page 136 after the translation).
            idxk = (f"[n_raw=2,n_keep=2,"
                    f"4382:{blk_a}:v0=1.0e+00:sum=2.0e+00:am=1.0e+00,"
                    f"4383:{blk_b}:v0=1.0e+00:sum=2.0e+00:am=1.0e+00]")
            return (f"[C4DEC] layer={ly} idx=1 start=[{start}] "
                    f"kbase={kbase} rdcarry={rd} idxK={idxk}")

        def _idxk_line(ly, h136, h137):
            # Stored [IDXK] keys are c4-PAGE columns (136 = tokens 17408..17535).
            blkx = f"['136:{h136}', '137:{h137}']"
            return (f"[IDXK] layer={ly} mode=EXTEND logical=ll "
                    f"blkx={blkx} ptab=['136:7', '137:8']")

        def _parse_src(src_lines):
            fd5, p5 = tempfile.mkstemp(suffix=".log", prefix="cmpidx_selftest_")
            try:
                with os.fdopen(fd5, "w", encoding="utf-8") as fh5:
                    fh5.write("\n".join(src_lines) + "\n")
                return parse(p5)
            finally:
                try:
                    os.remove(p5)
                except OSError:
                    pass

        _chat = "/v1/chat/completions"
        # H1 OUTPUT-SLOT: produced MISS==HIT, stored MISS!=HIT at page 136.
        c4dec_slot = [
            _chat,
            _c4dec(2, 0, _KB_A, _RC_A, "aaaa", "bbbb"),
            _idxk_line(2, "p111", "p222"),
            _chat,
            _c4dec(2, 17536, _KB_A, _RC_A, "aaaa", "bbbb"),
            _idxk_line(2, "p333", "p222"),
        ]
        # H2 CONTENT:K-SPLIT: produced differs + kbase differs.
        c4dec_ksplit = [
            _chat,
            _c4dec(2, 0, _KB_A, _RC_A, "aaaa", "bbbb"),
            _idxk_line(2, "p111", "p222"),
            _chat,
            _c4dec(2, 17536, _KB_B, _RC_A, "cccc", "bbbb"),
            _idxk_line(2, "p111", "p222"),
        ]
        # H3 CONTENT:READ-MISALIGN: produced differs, kbase same, rdcarry differs.
        c4dec_rmis = [
            _chat,
            _c4dec(2, 0, _KB_A, _RC_A, "aaaa", "bbbb"),
            _idxk_line(2, "p111", "p222"),
            _chat,
            _c4dec(2, 17536, _KB_A, _RC_B, "cccc", "bbbb"),
            _idxk_line(2, "p111", "p222"),
        ]
        # H5 (page-136 mapping): produced block 4383 -> page 136.  The two
        #     requests produce the SAME 4382/4383 hashes but stored page 136
        #     differs -> OUTPUT-SLOT.  BEFORE the block->page translation the join
        #     looked up stored page 4383 (absent) so stored_diff stayed False and
        #     the verdict was wrongly SAME.
        c4dec_page136 = [
            _chat,
            _c4dec(2, 0, _KB_A, _RC_A, "same4382", "same4383"),
            _idxk_line(2, "p111", "p222"),
            _chat,
            _c4dec(2, 17536, _KB_A, _RC_A, "same4382", "same4383"),
            _idxk_line(2, "p333", "p222"),
        ]
        # H6 (produced differs on page 136): same 4383 key, DIFFERENT produced
        #     hash, kbase+rdcarry identical -> CONTENT (producer-internal).
        c4dec_prod_diff = [
            _chat,
            _c4dec(2, 0, _KB_A, _RC_A, "aaaa", "same4383"),
            _idxk_line(2, "p111", "p222"),
            _chat,
            _c4dec(2, 17536, _KB_A, _RC_A, "cccc", "same4383"),
            _idxk_line(2, "p111", "p222"),
        ]
        c4slot_out = _run_main(c4dec_slot)
        c4ksplit_out = _run_main(c4dec_ksplit)
        c4rmis_out = _run_main(c4dec_rmis)
        c4page136_out = _run_main(c4dec_page136)
        c4proddiff_out = _run_main(c4dec_prod_diff)
        # H7 ([IDXK] reuse): produced page 136 is IDENTICAL MISS vs HIT, but the
        #     [IDXK] POOL itself carries TWO hashes for page 136 -- exactly what
        #     ``report_blkx`` prints as ``divergent=1 first_block=136``.  The
        #     per-request stored maps do NOT line up with the [C4DEC] indices
        #     (both [IDXK] rows sit in req 1, so the hit req's stored map is
        #     empty), so the OLD keying missed it -> wrongly SAME.  The stored
        #     side must REUSE the [IDXK] divergence -> OUTPUT-SLOT, with page
        #     136's two hashes in the verdict row.
        c4dec_idxk_div = [
            _chat,
            _c4dec(2, 0, _KB_A, _RC_A, "same4382", "same4383"),
            _idxk_line(2, "79bc7de27691ef57", "p222"),
            _idxk_line(2, "be64ada86e32b778", "p222"),
            _chat,
            _c4dec(2, 16384, _KB_A, _RC_A, "same4382", "same4383"),
        ]
        # H8 (no divergence): same produced page 136 and the [IDXK] pool agrees on
        #     ONE hash for page 136 -> stored_diff stays False -> SAME (SAME only
        #     when NO shared page diverges).
        c4dec_no_div = [
            _chat,
            _c4dec(2, 0, _KB_A, _RC_A, "same4382", "same4383"),
            _idxk_line(2, "p111", "p222"),
            _idxk_line(2, "p111", "p222"),
            _chat,
            _c4dec(2, 16384, _KB_A, _RC_A, "same4382", "same4383"),
        ]
        c4idxkdiv_out = _run_main(c4dec_idxk_div)
        c4nodiv_out = _run_main(c4dec_no_div)        # H4: the EXACT reported repro -- FIVE ``/v1/chat/completions`` requests
        #     (boundary line AFTER each request's rows -> 0-based indices); req0..3
        #     are cache-MISS (sps=0) and req4 the cache-HIT (sps=16384).  The
        #     widened [C4DEC] gate emits the MISS calls from a window-OVERLAPPING
        #     chunk, so ALL five [C4DEC] start lists are 0-FREE.  The split MUST
        #     follow the request boundary + sps rule -> miss_req=[0, 1, 2, 3],
        #     hit_req=[4]; the OLD start==0 test gave miss_req=[].
        c4dec_startfree = []
        for _r in range(5):
            _shift = 0 if _r < 4 else 16384
            c4dec_startfree += [
                _idxin(2, 17523, "aaaa", kpg_a,
                       knorm="198.935:100:64", shift=_shift),
                _c4dec(2, 16384, _KB_A, _RC_A, "aaaa", "bbbb"),
                _idxk_line(2, "p111", "p222"),
                _chat,
            ]
        h4_out = _run_main(c4dec_startfree)
        _slot_c4 = _parse_src(c4dec_slot).get("C4DEC", [])

        b1_rows, b1_out = _run_nocmp(nocmp_rope)
        b2_rows, b2_out = _run_nocmp(nocmp_content)
        c1_rows, c1_out = _run_nocmp(same_slq)
        c2_rows, c2_out = _run_nocmp(no_same_slq)
        d1_rows, d1_out = _run_nocmp(chat_split)
        e1_rows, e1_out = _run_nocmp(per_shift_rope)
        e2_rows, e2_out = _run_nocmp(per_shift_content)
        f1_rows, f1_out = _run_nocmp(per_sps)
        g1_rows, g1_out = _run_c4pos(g1_pos)
        g2_rows, g2_out = _run_c4pos(g2_pos)
        b1_reqs = _idxin_requests(b1_rows)
        b2_reqs = _idxin_requests(b2_rows)

        # --- [CMPIDX-ALIGNED] per-absolute-position top-k SET (I1/I2) ---
        # I1: two MISS (first==0) + one HIT (first==16384); IDENTICAL top-k SETs
        #     at the shared positions -> TOPK_SAME (+ two-MISS control OK).
        # I2: the HIT swaps key 9 -> 999 at pos 17522 -> TOPK_DIFF with the
        #     symmetric-difference offenders.
        def _cmpidx_line(ly, mode, ntok, kidx, kord=None):
            extra = f" kord={kord}" if kord is not None else ""
            return (f"[CMPIDX] layer={ly} mode={mode} ntok={ntok} "
                    f"lastpos=17522 md5=zz tail=[] blkx=[] kidx={kidx}{extra}")

        def _c4pos_first(first, ly=2):
            return (f"[C4POS] layer={ly} idx=1 n=8 ratio=4 nz=8 first={first} "
                    f"last=0 md5=aa sum=1 pos=[0,4,8,12,16,20,24,28]")

        _kidx_ok = "[17520:1,2,3|17521:4,5,6|17522:7,8,9]"
        _kidx_bad = "[17520:1,2,3|17521:4,5,6|17522:7,8,999]"
        cmpidx_same = [
            CHAT, _c4pos_first(0), _cmpidx_line(2, 1, 17523, _kidx_ok),
            CHAT, _c4pos_first(0), _cmpidx_line(2, 1, 17523, _kidx_ok),
            CHAT, _c4pos_first(16384), _cmpidx_line(2, 1, 1139, _kidx_ok),
        ]
        cmpidx_diff = [
            CHAT, _c4pos_first(0), _cmpidx_line(2, 1, 17523, _kidx_ok),
            CHAT, _c4pos_first(0), _cmpidx_line(2, 1, 17523, _kidx_ok),
            CHAT, _c4pos_first(16384), _cmpidx_line(2, 1, 1139, _kidx_bad),
        ]
        ci_same_out = _run_main(cmpidx_same)
        ci_diff_out = _run_main(cmpidx_diff)

        # J1/J2: [CMPIDX-ORDERED-ALIGNED] top-k SEQUENCE.  J1 emits identical
        #     kord -> ORDER_SAME; J2 emits the SAME SET in a DIFFERENT ORDER
        #     (7,8,9 -> 8,7,9) -> ORDER_DIFF with SET_miss==SET_hit (R1).
        _kord_ok = "[17520:1,2,3|17521:4,5,6|17522:7,8,9]"
        _kord_perm = "[17520:1,2,3|17521:4,5,6|17522:8,7,9]"
        cmpidx_ord_same = [
            CHAT, _c4pos_first(0),
            _cmpidx_line(2, 1, 17523, _kidx_ok, _kord_ok),
            CHAT, _c4pos_first(0),
            _cmpidx_line(2, 1, 17523, _kidx_ok, _kord_ok),
            CHAT, _c4pos_first(16384),
            _cmpidx_line(2, 1, 1139, _kidx_ok, _kord_ok),
        ]
        cmpidx_ord_diff = [
            CHAT, _c4pos_first(0),
            _cmpidx_line(2, 1, 17523, _kidx_ok, _kord_ok),
            CHAT, _c4pos_first(0),
            _cmpidx_line(2, 1, 17523, _kidx_ok, _kord_ok),
            CHAT, _c4pos_first(16384),
            _cmpidx_line(2, 1, 1139, _kidx_ok, _kord_perm),
        ]
        ci_ord_same_out = _run_main(cmpidx_ord_same)
        ci_ord_diff_out = _run_main(cmpidx_ord_diff)

        # --- [QK-ALIGNED] query vs index-K at P* (Q1/Q2) ---
        # Q1: two MISS (sps=0) + one HIT (sps=16384) share qpos=[17598:qq] AND
        #     klog -> Q_SAME + K_SAME -> DECISION BOTH-SAME (indexer-internal).
        # Q2: the HIT's qpos differs (qq->rr) -> Q_DIFF -> DECISION QUERY-PATH.
        def _qk_line(ly, slq, klog, shift, qhash):
            return _idxin(ly, slq, klog, kpg_a, shift=shift,
                          qpos=f"[17598:{qhash}]")

        qk_same = [
            CHAT, _c4pos_first(0), _qk_line(2, 17523, "aaaa", 0, "qq"),
            CHAT, _c4pos_first(0), _qk_line(2, 17523, "aaaa", 0, "qq"),
            CHAT, _c4pos_first(16384), _qk_line(2, 1139, "aaaa", 4096, "qq"),
        ]
        qk_diff = [
            CHAT, _c4pos_first(0), _qk_line(2, 17523, "aaaa", 0, "qq"),
            CHAT, _c4pos_first(0), _qk_line(2, 17523, "aaaa", 0, "qq"),
            CHAT, _c4pos_first(16384), _qk_line(2, 1139, "aaaa", 4096, "rr"),
        ]
        qk_same_out = _run_main(qk_same)
        qk_diff_out = _run_main(qk_diff)

        # --- [LHID] per-absolute-position single-row (S252) ---
        # Two requests emit the SAME layer's hidden at absolute position 17597:
        # layers 1/2 identical, layer 3 differs -> the first divergent layer.
        def _lhid_line(ly, hpos):
            return (f"[LHID] layer={ly} lastpos=17597 ntok=1 md5=full "
                    f"blkx=[] hpos={hpos}")

        lhid_diff = [
            _lhid_line(1, "[17597:aa]"),
            _lhid_line(2, "[17597:bb]"),
            _lhid_line(3, "[17597:cc]"),
            _lhid_line(3, "[17597:dd]"),
        ]
        lhid_same = [
            _lhid_line(1, "[17597:aa]"),
            _lhid_line(2, "[17597:bb]"),
            _lhid_line(3, "[17597:cc]"),
            _lhid_line(3, "[17597:cc]"),
        ]
        lhid_out = _run_main(lhid_diff)
        lhid_same_out = _run_main(lhid_same)

        # --- [XIN-POS] per-absolute-position indexer input x hash (S253) ---
        # Two MISS (first==0) + one HIT (first==16384) of layer 0 share xpos at
        # pos 17598: X1 all equal -> X_SAME (+ two-MISS control OK); X2 the HIT
        # differs at 17598 (ab -> ef) -> X_DIFF at the SAME absolute position.
        def _xin_line(ly, ntok, xpos):
            return (f"[XIN] layer={ly} lastpos=17598 ntok={ntok} "
                    f"shape=({ntok}, 128) md5=full xpos={xpos}")

        _xp_ok = "[17598:ab|17599:cd]"
        _xp_bad = "[17598:ef|17599:cd]"
        xin_same = [
            CHAT, _c4pos_first(0, 0), _xin_line(0, 17523, _xp_ok),
            CHAT, _c4pos_first(0, 0), _xin_line(0, 17523, _xp_ok),
            CHAT, _c4pos_first(16384, 0), _xin_line(0, 1139, _xp_ok),
        ]
        xin_diff = [
            CHAT, _c4pos_first(0, 0), _xin_line(0, 17523, _xp_ok),
            CHAT, _c4pos_first(0, 0), _xin_line(0, 17523, _xp_ok),
            CHAT, _c4pos_first(16384, 0), _xin_line(0, 1139, _xp_bad),
        ]
        xin_out = _run_main(xin_same)
        xin_diff_out = _run_main(xin_diff)

        # --- [LHID-ACCEPT] MISS-vs-HIT fix gate (S1) + two-MISS control (S2) ---
        # K1: two MISS (first==0) agree at 17597, the HIT (first==16384)
        #     DIFFERS -> S1 divergent=1 => FAIL (the RED gate on today's logs);
        #     S2 two-MISS stays IDENTICAL.
        # K2: the HIT matches the MISS hash -> S1 divergent=0 => PASS.
        # K3: the two MISS requests DIFFER (aa vs bb) with a HIT present ->
        #     S2 two-MISS MISMATCH (proves the S2 control is NON-VACUOUS).
        # K4: no `hpos=` field -> S1 CANNOT-VERIFY, never a false PASS.
        lhid_acc_fail = [
            CHAT, _c4pos_first(0), _lhid_line(1, "[17597:aa]"),
            CHAT, _c4pos_first(0), _lhid_line(1, "[17597:aa]"),
            CHAT, _c4pos_first(16384), _lhid_line(1, "[17597:bb]"),
        ]
        lhid_acc_pass = [
            CHAT, _c4pos_first(0), _lhid_line(1, "[17597:aa]"),
            CHAT, _c4pos_first(0), _lhid_line(1, "[17597:aa]"),
            CHAT, _c4pos_first(16384), _lhid_line(1, "[17597:aa]"),
        ]
        lhid_acc_2miss = [
            CHAT, _c4pos_first(0), _lhid_line(1, "[17597:aa]"),
            CHAT, _c4pos_first(0), _lhid_line(1, "[17597:bb]"),
            CHAT, _c4pos_first(16384), _lhid_line(1, "[17597:aa]"),
        ]
        la_fail_out = _run_main(lhid_acc_fail)
        la_pass_out = _run_main(lhid_acc_pass)
        la_2miss_out = _run_main(lhid_acc_2miss)
        la_nohpos_out = _run_main([
            CHAT, _c4pos_first(0),
            "[LHID] layer=1 lastpos=17597 ntok=1 md5=full blkx=[]",
        ])

        # --- [WSTAGE-VERDICT] write-side stage sweep (6 fixtures a-f) ---
        # One line per position per stage:
        #   [WSTAGE] layer=<L> pos=<p> stage=S<i> md5=<h>   (i in 0..5)
        def _wstage(ly, pos, stage, md5):
            return f"[WSTAGE] layer={ly} pos={pos} stage={stage} md5={md5}"

        def _wstage_sweep(ly, pos, hashes):
            """S0..S5 lines for one (layer,pos); ``hashes`` = 6 md5 values."""
            return [_wstage(ly, pos, st, h)
                    for st, h in zip(WSTAGES, hashes)]

        W_LAYER, W_POS = 2, 17522
        _base = ["a", "b", "c", "d", "e", "f"]

        def _wstage_log(miss_hashes, hit_hashes, miss2=None):
            """CHAT/C4POS-framed WSTAGE log: MISS (+ optional 2nd MISS) + HIT."""
            lines = [CHAT, _c4pos_first(0)]
            lines += _wstage_sweep(W_LAYER, W_POS, miss_hashes)
            if miss2 is not None:
                lines += [CHAT, _c4pos_first(0)]
                lines += _wstage_sweep(W_LAYER, W_POS, miss2)
            lines += [CHAT, _c4pos_first(16384)]
            lines += _wstage_sweep(W_LAYER, W_POS, hit_hashes)
            return lines

        # (a) S1 diff, S0 same -> A-GEMM.
        wa_log = _wstage_log(_base, ["a", "X", "c", "d", "e", "f"])
        # (b) S3 diff with S0-S2 same -> C-ROPE.
        wb_log = _wstage_log(_base, ["a", "b", "c", "X", "e", "f"])
        # (c) S5 diff with S0-S4 same -> E-fp8-epilog.
        wc_log = _wstage_log(_base, ["a", "b", "c", "d", "e", "X"])
        # (d) all stages same (2 MISS + HIT) -> ALL-SAME + two-MISS IDENTICAL.
        wd_log = _wstage_log(_base, _base, miss2=_base)
        # (e) S0 diff -> UPSTREAM-INPUT.
        we_log = _wstage_log(_base, ["X", "b", "c", "d", "e", "f"])
        # (f) two-MISS mismatch at S1 (MISS2 S1=Z) with HIT == MISS1 -> UNRELIABLE.
        wf_log = _wstage_log(_base, _base,
                             miss2=["a", "Z", "c", "d", "e", "f"])
        wa_out = _run_main(wa_log)
        wb_out = _run_main(wb_log)
        wc_out = _run_main(wc_log)
        wd_out = _run_main(wd_log)
        we_out = _run_main(we_log)
        wf_out = _run_main(wf_log)
        # absent [WSTAGE] rows -> runbook, never a false verdict.
        w_none_out = _run_main([CHAT, _c4pos_first(0)])
        w_parse = _parse_src(wa_log).get("WSTAGE", [])

        checks = [
            ("kpg full value parsed",
             bool(idxin) and idxin[0].get("kpg") == kpg_a),
            ("klog parsed", bool(idxin) and idxin[0].get("klog") == "aaaa"),
            ("knorm parsed",
             bool(idxin) and idxin[0].get("knorm") == "1.0:100:64"),
            ("ropephase shift parsed (miss=0, hit=4096)",
             bool(idxin) and _rope_shift(idxin[0].get("ropephase")) == 0
             and any(_rope_shift(r.get("ropephase")) == 4096 for r in idxin)),
            ("kraw/meta/bt kept",
             bool(idxin) and idxin[0].get("kraw") == "kk"
             and idxin[0].get("meta") == "mm" and idxin[0].get("bt") == "tt"),
            ("KSTATELOC op=W count/out",
             len(writes) == 3 and writes[0]["out"] == 17520
             and writes[2]["out"] == 17524),
            ("KSTATELOC op=W seq/sl",
             writes[0]["seq"] == 17520 and writes[0]["sl"] == 40),
            ("KSTATELOC op=R parsed",
             len(reads) == 1 and reads[0]["seq"] == 17520
             and reads[0]["sl"] == 40),
            ("[IDXIN-VERDICT] SAME branch",
             "[IDXIN-VERDICT] layer=5" in out
             and "SAME at these probes" in out),
            ("[IDXIN-VERDICT] READ-PAGE branch",
             "[IDXIN-VERDICT] layer=7" in out
             and "READ-PAGE misaligned" in out),
            ("[IDXIN-ROPE] ROPE-PHASE verdict",
             "[IDXIN-ROPE] " in out and "=> ROPE-PHASE" in out),
            ("[KSTATELOC-W] out sequence printed",
             "out=17520" in out and "out=17524" in out),
            ("no-CMPIDX fallback split >=2 requests",
             len(b1_reqs) >= 2 and len(b2_reqs) >= 2),
            ("no-CMPIDX fallback MISS = largest slq (req0)",
             _idxin_miss_index(b1_reqs) == 0),
            ("[IDXIN-KM] runs on fallback (not <2 skipped)",
             "[IDXIN-KM] MISS=req0" in b1_out
             and "<2 requests" not in b1_out),
            ("[IDXIN-ROPE] fallback ROPE-PHASE branch",
             "=> ROPE-PHASE" in b1_out),
            ("[IDXIN-ROPE] fallback CONTENT/WINDOW branch",
             "=> CONTENT/WINDOW" in b2_out),
            ("[IDXIN-ROPE] same-slq HIT paired (not the first HITs)",
             "same-slq/slk HIT(s)=[3]" in c1_out
             and "hit_slq=17523" in c1_out
             and "miss_slq=17523" in c1_out),
            ("[IDXIN-ROPE] same-slq HIT -> ROPE-PHASE (fix)",
             "n_shared=1" in c1_out and "=> ROPE-PHASE" in c1_out
             and "=> CONTENT/WINDOW" not in c1_out),
            ("[IDXIN-ROPE] no-same-slq fallback is STATED",
             "no same-slq HIT" in c2_out),
            ("[IDXIN-ROPE] no-same-slq fallback still ROPE-PHASE",
             "=> ROPE-PHASE" in c2_out),
            ("[IDXIN-SPLIT] sps rule labels chat reqs (miss 1-4, hit 5)",
             "source=sps" in d1_out
             and "miss_req=[1, 2, 3, 4]" in d1_out
             and "hit_req=[5]" in d1_out),
            ("[IDXIN-SPLIT] info: sps + chat/completions + 5 reqs",
             _idxin_split_info(d1_rows)
             == ("sps", "chat/completions", 5, [1, 2, 3, 4], [5])),
            ("[IDXIN-ROPE] chat-split MISS vs HIT -> ROPE-PHASE",
             "miss_req=[1, 2, 3, 4]" in d1_out and "hit_req=[5]" in d1_out
             and "=> ROPE-PHASE" in d1_out
             and "=> CONTENT/WINDOW" not in d1_out),
            ("[IDXIN-SPLIT] non-chat logs keep lastpos/stream source",
             _idxin_split_source(idxin) == "lastpos"
             and _idxin_split_source(b1_rows) == "stream"),
            ("[IDXIN-ROPE] per-block: extra MISS block NOT judged -> ROPE-PHASE",
             "n_shared=2" in e1_out and "n_miss_only=1" in e1_out
             and "n_hit_only=0" in e1_out
             and "=> ROPE-PHASE" in e1_out
             and "=> CONTENT/WINDOW" not in e1_out),
            ("[IDXIN-ROPE] per-block: shared block differs -> CONTENT/WINDOW",
             "n_shared=2" in e2_out and "=> CONTENT/WINDOW" in e2_out),
            ("[IDXIN-ROPE] per-block offending block printed",
             "differing shared blocks" in e2_out and "shift=4096:" in e2_out),
            ("[IDXIN-SPLIT] sps rule: 4 MISS (sps=0) + 1 HIT (sps=16384)",
             "source=sps" in f1_out and "miss_req=[0, 1, 2, 3]" in f1_out
             and "hit_req=[4]" in f1_out),
            ("[IDXIN-SPLIT] info: sps + stream + miss 0-3 / hit 4",
             _idxin_split_info(f1_rows)
             == ("sps", "stream", 5, [0, 1, 2, 3], [4])),
            ("[IDXIN-ROPE] sps MISS set self-consistent",
             "miss_consistent=True" in f1_out),
            ("[IDXIN-ROPE] sps shared block -> ROPE-PHASE",
             "n_shared=1" in f1_out and "=> ROPE-PHASE" in f1_out
             and "=> CONTENT/WINDOW" not in f1_out),
            ("[IDXIN-ROPE] per-request knorm+ropephase printed",
             "knorm[" in b1_out and "ropephase[" in b1_out),
            ("[C4POS] pos= parsed per row",
             bool(g1_rows) and _c4pos_ints(g1_rows[0])[:3] == [0, 4, 8]),
            ("[C4POS] prefill row keeps the longest pos list",
             _c4pos_ints(g1_rows[0]) == [0, 4, 8, 12, 16, 20, 24, 28]),
            ("[C4POS-VERDICT] same absolute blocks -> phase_same",
             "=> phase_same" in g1_out and "n_shared=" in g1_out
             and "n_miss_only=" in g1_out),
            ("[C4POS-VERDICT] differing absolute block -> phase_diff",
             "=> phase_diff" in g2_out and "block7:" in g2_out),
            ("[C4POS-RUNBOOK] printed",
             "[C4POS-RUNBOOK]" in g1_out and "DSV4_DUMP_C4POS=1" in g1_out),
            ("[C4POS] real main() emits header+verdict on the 2-row repro",
             "== [C4POS]" in repro_out and "[C4POS-VERDICT]" in repro_out),
            ("[C4POS] real main() prints a phase_same verdict",
             "=> phase_same" in g5_out),
            ("[C4POS] real main() prints no-data fallback when no rows",
             "== [C4POS]" in no_c4pos_out
             and "no [C4POS] rows (dump gate off?)" in no_c4pos_out),
            ("[C4DEC] parse: full kbase + 2 idxK blocks + start==0",
             bool(_slot_c4)
             and _slot_c4[0].get("kbase") == _KB_A
             and len(_c4dec_idxk(_slot_c4[0])[0]) == 2
             and _c4dec_start_min(_slot_c4[0]) == 0),
            ("[C4DEC] produced==produced but stored differs -> OUTPUT-SLOT",
             "== [C4DEC]" in c4slot_out
             and "=> OUTPUT-SLOT" in c4slot_out
             and "produced_same=True" in c4slot_out
             and "stored_diff=True" in c4slot_out
             and "page=136" in c4slot_out),
            ("[C4DEC] produced differs + kbase differs -> CONTENT:K-SPLIT",
             "=> CONTENT:K-SPLIT" in c4ksplit_out
             and "produced_same=False" in c4ksplit_out
             and "kbase_diff=True" in c4ksplit_out),
            ("[C4DEC] produced differs, kbase same, rdcarry differs"
             " -> CONTENT:READ-MISALIGN",
             "=> CONTENT:READ-MISALIGN" in c4rmis_out
             and "produced_same=False" in c4rmis_out
             and "kbase_diff=False" in c4rmis_out
             and "rdcarry_diff=True" in c4rmis_out),
            ("[C4DEC] block 4383 -> page 136: same produced + stored differs"
             " -> OUTPUT-SLOT",
             "=> OUTPUT-SLOT" in c4page136_out
             and "n_shared=1" in c4page136_out
             and "page=136" in c4page136_out
             and "stored_diff=True" in c4page136_out),
            ("[C4DEC] produced differs on page 136 -> CONTENT",
             "=> CONTENT" in c4proddiff_out
             and "=> CONTENT:K-SPLIT" not in c4proddiff_out
             and "=> CONTENT:READ-MISALIGN" not in c4proddiff_out
             and "produced_same=False" in c4proddiff_out),
            ("[C4DEC] start-free MISS classified by request boundary"
             " -> miss 0-3 / hit 4",
             "[C4DEC-VERDICT] miss_req=[0, 1, 2, 3] hit_req=[4]" in h4_out
             and "miss_req=[]" not in h4_out
             and "source=idxin:sps" in h4_out),
            ("[C4DEC] [IDXK] divergent page 136 REUSED -> OUTPUT-SLOT"
             " (produced_same, stored_diff)",
             "== [C4DEC]" in c4idxkdiv_out
             and "=> OUTPUT-SLOT" in c4idxkdiv_out
             and "produced_same=True" in c4idxkdiv_out
             and "stored_diff=True" in c4idxkdiv_out
             and "page=136" in c4idxkdiv_out
             and "79bc7de27691ef57" in c4idxkdiv_out
             and "be64ada86e32b778" in c4idxkdiv_out
             and "=> SAME" not in c4idxkdiv_out),
            ("[C4DEC] no [IDXK] divergence -> SAME",
             "== [C4DEC]" in c4nodiv_out
             and "=> SAME" in c4nodiv_out
             and "stored_diff=False" in c4nodiv_out
             and "=> OUTPUT-SLOT" not in c4nodiv_out),
            ("[CMPIDX-ALIGNED] kidx parsed per absolute position",
             _cmpidx_kidx({"kidx": _kidx_bad})[17522] == [7, 8, 999]),
            ("[CMPIDX-ALIGNED] TOPK_SAME + two-MISS control OK",
             "split_source=c4pos" in ci_same_out
             and "=> TOPK_SAME" in ci_same_out
             and "two-MISS control: IDENTICAL" in ci_same_out
             and "=> TOPK_DIFF" not in ci_same_out),
            ("[CMPIDX-ALIGNED] TOPK_DIFF + offenders",
             "=> TOPK_DIFF" in ci_diff_out
             and "offenders(symdiff)=[9, 999]" in ci_diff_out
             and "split_source=c4pos" in ci_diff_out),
            ("[CMPIDX-ALIGNED] miss/hit roles are disjoint (no MISS-vs-MISS pair)",
             "[CMPIDX-ALIGNED] split_source=c4pos miss_req=[1, 2] "
             "hit_req=[3]" in ci_same_out),
            ("[CMPIDX-ORDERED-ALIGNED] kord parsed in STORED order",
             _cmpidx_kord({"kord": _kord_perm})[17522] == [8, 7, 9]),
            ("[CMPIDX-ORDERED-ALIGNED] ORDER_SAME + two-MISS control OK",
             "split_source=c4pos" in ci_ord_same_out
             and "=> ORDER_SAME" in ci_ord_same_out
             and "two-MISS control: SEQUENCE-IDENTICAL" in ci_ord_same_out
             and "escalate to the metadata/attention probes" in ci_ord_same_out
             and "=> ORDER_DIFF" not in ci_ord_same_out),
             ("[CMPIDX-ORDERED-ALIGNED] ORDER_DIFF + SET same (R1)",
              "=> ORDER_DIFF" in ci_ord_diff_out
              and "SET_same" in ci_ord_diff_out
              and "first_diff offset=0 miss=7 hit=8" in ci_ord_diff_out
              and "swapped_miss=[7, 8, 9] -> hit=[8, 7, 9]" in ci_ord_diff_out
              and "ordered-consumption" in ci_ord_diff_out),
            ("[QK-ALIGNED] qpos parsed per absolute position",
             _idxin_qpos({"qpos": "[17598:qq|17599:rr]"})
             == {17598: "qq", 17599: "rr"}),
            ("[QK-ALIGNED] Q_SAME + K_SAME -> DECISION BOTH-SAME",
             "[QK-ALIGNED] QUERY(qpos): " in qk_same_out
             and "=> Q_SAME" in qk_same_out
             and "=> K_SAME" in qk_same_out
             and "BOTH-SAME" in qk_same_out
             and "QUERY-PATH" not in qk_same_out),
            ("[QK-ALIGNED] two-MISS qpos control OK",
             "two-MISS control: IDENTICAL" in qk_same_out),
            ("[QK-ALIGNED] qpos diff -> Q_DIFF -> DECISION QUERY-PATH",
             "=> Q_DIFF" in qk_diff_out
             and "=> Q_SAME" not in qk_diff_out
             and "QUERY-PATH" in qk_diff_out),
            ("[QK-ALIGNED] index-K alignment field emitted",
             "blkx_aligned_diff=" in qk_same_out),
            ("[LHID-POS] hpos parsed per absolute position",
             _lhid_hpos({"hpos": "[17597:aa|17598:bb]"})
             == {17597: "aa", 17598: "bb"}),
            ("[LHID-POS] first divergent layer at first pos reported",
             "[LHID-POS]" in lhid_out
             and "FIRST divergent position=17597" in lhid_out
             and "layers there = [3]" in lhid_out
             and "layer=3" in lhid_out),
            ("[LHID-POS] identical layers -> no divergence",
             "no per-position divergence" in lhid_same_out),
            ("[XIN-POS] xpos parsed per absolute position",
             _xin_xpos({"xpos": "[17598:ab|17599:cd]"})
             == {17598: "ab", 17599: "cd"}),
            ("[XIN-POS] X_SAME + two-MISS control OK",
             "split_source=c4pos" in xin_out
             and "=> X_SAME" in xin_out
             and "two-MISS control: IDENTICAL" in xin_out
             and "=> X_DIFF" not in xin_out),
            ("[XIN-POS] X_DIFF at the SAME absolute position",
             "=> X_DIFF" in xin_diff_out
             and "pos=17598" in xin_diff_out
             and "X_miss=ab" in xin_diff_out
             and "X_hit=ef" in xin_diff_out),
            ("[LHID-ACCEPT] S1 divergent>0 -> FAIL (RED gate, non-vacuous)",
             "[LHID-ACCEPT] S1 divergent=1 => FAIL" in la_fail_out
             and "S1 first divergence layer=1 pos=17597" in la_fail_out),
            ("[LHID-ACCEPT] S1 divergent=0 -> PASS (identical fixture)",
             "[LHID-ACCEPT] S1 divergent=0 => PASS" in la_pass_out
             and "S1 divergent=1 => FAIL" not in la_pass_out),
            ("[LHID-ACCEPT] S2 two-MISS IDENTICAL on the PASS fixture",
             "[LHID-ACCEPT] S2 two-MISS IDENTICAL" in la_pass_out),
            ("[LHID-ACCEPT] S2 two-MISS MISMATCH (control non-vacuous)",
             "[LHID-ACCEPT] S2 two-MISS MISMATCH" in la_2miss_out),
            ("[LHID-ACCEPT] S3 manual placeholder printed",
             "[LHID-ACCEPT] S3 curl#1..5 byte-equal: <manual>"
             in la_pass_out),
            ("[LHID-ACCEPT] no hpos -> CANNOT-VERIFY (no false PASS)",
             "S1 divergent=? => CANNOT-VERIFY" in la_nohpos_out
             and "S1 divergent=0 => PASS" not in la_nohpos_out),
            ("[WSTAGE] parser: 12 rows, keyed by (layer,abs_pos,stage)",
             len(w_parse) == 12
             and _wstage_maps(w_parse)[2][17522]["S0"] == "a"
             and _wstage_maps(w_parse)[2][17522]["S5"] == "f"),
            ("[WSTAGE-VERDICT] (a) S1 diff -> A-GEMM",
             "split_source=c4pos" in wa_out
             and "first_diff_stage=S1" in wa_out
             and "MODE = S1" in wa_out and "A-GEMM" in wa_out
             and "UPSTREAM-INPUT" not in wa_out),
            ("[WSTAGE-VERDICT] (b) S3 diff (S0-S2 same) -> C-ROPE",
             "first_diff_stage=S3" in wb_out
             and "MODE = S3" in wb_out and "C-ROPE" in wb_out),
            ("[WSTAGE-VERDICT] (c) S5 diff (S0-S4 same) -> E-fp8-epilog",
             "first_diff_stage=S5" in wc_out
             and "MODE = S5" in wc_out and "E-fp8-epilog" in wc_out),
            ("[WSTAGE-VERDICT] (d) all same -> ALL-SAME",
             "MODE = none" in wd_out and "ALL-SAME" in wd_out
             and "first_diff_stage=S" not in wd_out),
            ("[WSTAGE-VERDICT] (d) two-MISS IDENTICAL (control non-vacuous)",
             "two-MISS IDENTICAL" in wd_out),
            ("[WSTAGE-VERDICT] (e) S0 diff -> UPSTREAM-INPUT",
             "first_diff_stage=S0" in we_out
             and "MODE = S0" in we_out and "UPSTREAM-INPUT" in we_out),
            ("[WSTAGE-VERDICT] (f) two-MISS MISMATCH -> UNRELIABLE",
             "two-MISS MISMATCH" in wf_out
             and "UNRELIABLE" in wf_out),
            ("[WSTAGE-VERDICT] no [WSTAGE] rows -> runbook, no false verdict",
             "no [WSTAGE] rows" in w_none_out
             and "DSV4_DUMP_WSTAGE=1" in w_none_out
             and "ALL-SAME" not in w_none_out),
        ]
        failed = [name for name, ok in checks if not ok]
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if failed:
            print(f"SELFTEST FAIL ({len(failed)}): {', '.join(failed)}")
            return 1
        print("SELFTEST PASS")
        return 0
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    main(sys.argv[1])

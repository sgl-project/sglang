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
    r"|C4ST|C4STP|KSTATELOC)\]"
)
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
    """``(source, n_req, hit_idx)`` banner fields for the ``[IDXIN-SPLIT]`` line.

    ``source`` names the segmenter in use; ``n_req`` is the detected request
    count (the HTTP request count when chat-split); ``hit_idx`` is the same-slq
    HIT that [IDXIN-ROPE] compares against the MISS (its boundary ordinal when
    chat-split), or ``None`` when there is no MISS/HIT pair.
    """
    source = _idxin_split_source(rows)
    reqs = _idxin_requests(rows)
    if source == "chat/completions":
        n_req = max((_int(r, "_req", 0) for r in rows), default=0)
    else:
        n_req = len(reqs)
    if len(reqs) < 2:
        return source, n_req, None
    miss_i = _idxin_miss_index(reqs)
    same_hits, _slq, _slk = _idxin_same_len_hits(reqs, miss_i)
    return source, n_req, (same_hits[0] if same_hits else None)


def report_idxin_split(rows):
    """One ``[IDXIN-SPLIT]`` banner naming the request-split source + HIT idx."""
    if not rows:
        return
    source, n_req, hit_i = _idxin_split_info(rows)
    hit_txt = "none" if hit_i is None else str(hit_i)
    print(f"  [IDXIN-SPLIT] source={source} requests={n_req} hit_idx={hit_txt}")


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
    miss_i = _idxin_miss_index(reqs)
    print(f"  [IDXIN-KM] MISS=req{miss_i} (largest prefill slq); "
          f"{n_req - 1} HIT request(s)")
    for i, g in enumerate(reqs[:MAX_REQ_SHOWN]):
        if not g:
            continue
        role = "MISS" if i == miss_i else "HIT "
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


def _knorm_verdict(miss_vals, hit_vals, tol):
    """(verdict, max_rel) comparing the MISS vs HIT ``knorm`` means.

    RoPE is orthonormal, so the per-pair magnitude is invariant: the means must
    match within ``tol`` (fp8 rounding) when the ONLY difference is the phase.
    A larger relative gap means the CONTENT/WINDOW differs.
    """
    if not miss_vals or not hit_vals:
        return "UNKNOWN", None
    rel = 0.0
    for a in miss_vals:
        for b in hit_vals:
            denom = max(abs(a), abs(b), 1e-12)
            rel = max(rel, abs(a - b) / denom)
    return ("CONTENT/WINDOW" if rel > tol else "SAME", rel)


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

    Pairing rule (the fix): the MISS (largest prefill) is compared against the
    HIT request(s) whose ``slq``/``slk`` MATCHES the MISS -- i.e. the cache-hit
    of the SAME sequence.  Pairing the MISS with the FIRST HITs (different, short
    prompts) forces a false ``CONTENT/WINDOW``.  When no HIT shares the length we
    fall back to all HITs and SAY SO.  A per-absolute-block knorm comparison is
    NOT derivable (``knorm`` is a whole-request mean); the whole-request same-slq
    comparison is used and that fact is printed.

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
    miss_i = _idxin_miss_index(reqs)
    knorms, shifts, phases = [], [], []
    for g in reqs:
        knorms.append(sorted({r.get("knorm") for r in g if r.get("knorm")}))
        shifts.append(sorted({_rope_shift(r.get("ropephase")) for r in g
                              if _rope_shift(r.get("ropephase")) is not None}))
        phases.append(sorted({r.get("ropephase") for r in g if r.get("ropephase")}))

    # Pair the MISS with the HIT(s) of the SAME prompt length (cache-hit of the
    # SAME sequence), NOT the first HITs.  Only that comparison is meaningful:
    # short unrelated HITs have a different window and would fake CONTENT/WINDOW.
    same_hits, miss_slq, miss_slk = _idxin_same_len_hits(reqs, miss_i)
    if same_hits:
        hit_idx = same_hits
        pairing_txt = f"same-slq/slk HIT(s)={same_hits}"
    else:
        hit_idx = [j for j in range(len(reqs)) if j != miss_i and reqs[j]]
        pairing_txt = "no same-slq HIT -> fallback to all HITs (lengths differ)"
    hit_len_txt = ",".join(str(s) for s in sorted({_idxin_req_len(reqs[j])[0]
                                                   for j in hit_idx}))

    print(f"  [IDXIN-ROPE] MISS=req{miss_i} (largest prefill slq={miss_slq}); "
          f"{n_req - 1} HIT request(s)  knorm_tol={knorm_tol}")
    print(f"  [IDXIN-ROPE] pairing: {pairing_txt}  miss_slk={miss_slk} "
          f"hit_slq={hit_len_txt}")
    print("  [IDXIN-ROPE] note: per-block knorm not derivable (knorm is a "
          "whole-request mean); using whole-request same-slq comparison")
    for i, _g in enumerate(reqs[:MAX_REQ_SHOWN]):
        if not _g:
            continue
        role = "MISS" if i == miss_i else "HIT "
        kmeans = [_knorm_mean(v) for v in knorms[i]]
        kmean_txt = ",".join(
            "?" if m is None else f"{m:.4g}" for m in kmeans[:MAX_DIFF_SHOWN])
        ph_txt = ";".join(_short(v) for v in phases[i][:2]) or "-"
        print(f"     req{i} {role} knorm[{len(knorms[i])}]={kmean_txt or '-'} "
              f"shift={shifts[i] or '-'} ropephase[{len(phases[i])}]={ph_txt}")

    miss_kn = [m for m in (_knorm_mean(v) for v in knorms[miss_i])
               if m is not None]
    hit_kn = []
    for j in hit_idx:
        hit_kn += [m for m in (_knorm_mean(v) for v in knorms[j])
                   if m is not None]
    knorm_cmp, rel = _knorm_verdict(miss_kn, hit_kn, knorm_tol)
    klog_diff = _idxin_comp_differs(rows)
    if knorm_cmp == "CONTENT/WINDOW":
        verdict = "CONTENT/WINDOW"
    elif klog_diff:
        verdict = "ROPE-PHASE"
    elif klog_diff is None:
        verdict = "UNKNOWN"
    else:
        verdict = "SAME"
    rel_txt = "na" if rel is None else f"{rel:.3e}"
    print(f"[IDXIN-ROPE] miss_slq={miss_slq} miss_knorm={miss_kn[:4]} "
          f"hit_slq={hit_len_txt} hit_knorm={hit_kn[:4]} "
          f"maxrel={rel_txt} knorm_same={knorm_cmp != 'CONTENT/WINDOW'} "
          f"klog_diff={klog_diff} => {verdict}")


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
    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA", "C4ST", "C4STP", "KSTATELOC"):
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

    for tag in ("CMPIDX", "IDXK", "C4KV", "C128KV", "C128X", "OSHAPE", "XIN", "LHID", "IDXIN", "LIMETA", "C4ST", "KSTATELOC"):
        rows = recs.get(tag, [])
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
            report_cmpidx_div(rows, cmp_reqs, miss_i)
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
    and asserts the boundary split picks request 5 as the HIT (``[IDXIN-SPLIT]
    source=chat/completions requests=5 hit_idx=5``) and yields ROPE-PHASE.
    """
    import contextlib
    import io
    import os
    import tempfile

    kpg_a = "n=8,first=[3, 5, 7, 9],last=[11, 13, 15, 17]"
    kpg_b = "n=8,first=[1, 2, 3, 4],last=[5, 6, 7, 8]"

    def _idxin(ly, slq, klog, kpg, knorm="1.0:100:64", shift=0, npb=100,
               slk=17523):
        return (
            f"[IDXIN] layer={ly} mode=1 qblk=[0:aa] wblk=[0:bb] "
            f"slq=({slq},{slq},1) slk=({slk},{slk},1) klog={klog} kpg={kpg} "
            f"knorm={knorm} ropephase=r=4,sps=[{shift * 4}],shift=[{shift}],"
            f"npb=[{npb}],loc=[0..{npb - 1}],abs=[{shift}..{shift + npb - 1}] "
            f"kraw=kk meta=mm bt=tt qshape=(1,2,3) wshape=(1,2,3) "
            f"kshape=(4,5,6)"
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
        _idxin(5, 17523, "aaaa", kpg_a),
        "[CMPIDX] layer=7 lastpos=101 ntok=17523 md5=ffff",
        _idxin(7, 17523, "zzzz", kpg_a),
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
        # B1: same knorm (RoPE-invariant) + differing klog -> ROPE-PHASE.
        nocmp_rope = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="1.0:100:64", shift=0),
            _idxin(7, 17523, "bbbb", kpg_a, knorm="1.0:100:64", shift=0),
            _idxin(5, 1139, "cccc", kpg_a, knorm="1.0:100:64", shift=4096),
            _idxin(7, 1139, "dddd", kpg_a, knorm="1.0:100:64", shift=4096),
        ]
        # B2: differing knorm (rel=0.5 > tol=0.05) -> CONTENT/WINDOW.
        nocmp_content = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="1.0:100:64", shift=0),
            _idxin(5, 1139, "aaaa", kpg_a, knorm="2.0:100:64", shift=4096),
        ]
        # C1: a same-slq HIT EXISTS (knorm ~equal, 198.935 vs 199.0) -> must pair
        #     the MISS with THAT HIT and print ROPE-PHASE -- NOT with the first,
        #     short-slq HITs (knorm 6.7/12.7) which would fake CONTENT/WINDOW.
        same_slq = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(5, 10, "xx", kpg_a, knorm="6.71568:100:64", shift=1, slk=10),
            _idxin(5, 12, "yy", kpg_a, knorm="12.7265:100:64", shift=1, slk=12),
            _idxin(5, 17523, "bbbb", kpg_a, knorm="199.0:100:64",
                   shift=4380, slk=17523),
        ]
        # C2: NO same-slq HIT -> fall back to all HITs AND SAY SO; knorm equal
        #     -> still ROPE-PHASE (the fallback is stated, not silent).
        no_same_slq = [
            _idxin(5, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            _idxin(5, 10, "xx", kpg_a, knorm="198.935:100:64", shift=1, slk=10),
        ]
        # D1: /v1/chat/completions HTTP boundaries -- the STRONGEST split signal.
        #     The 5th request is the HIT (same slq=17523 as the 4th=MISS); the
        #     boundary split must pick request 5 (not the earlier short ones) and
        #     print ROPE-PHASE.
        CHAT = ('INFO:     127.0.0.1:0 - "POST /v1/chat/completions '
                'HTTP/1.1" 200 OK')
        chat_split = [
            CHAT,  # request 1: short
            _idxin(5, 10, "s1", kpg_a, knorm="6.71568:100:64", shift=1, slk=10),
            CHAT,  # request 2: short
            _idxin(5, 12, "s2", kpg_a, knorm="12.7265:100:64", shift=1, slk=12),
            CHAT,  # request 3: short
            _idxin(5, 14, "s3", kpg_a, knorm="12.7265:100:64", shift=1, slk=14),
            CHAT,  # request 4: MISS prefill
            _idxin(5, 17523, "aaaa", kpg_a, knorm="198.935:100:64", shift=0),
            CHAT,  # request 5: HIT (same slq) -> the one to compare against
            _idxin(5, 17523, "bbbb", kpg_a, knorm="199.0:100:64",
                   shift=4380, slk=17523),
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

        b1_rows, b1_out = _run_nocmp(nocmp_rope)
        b2_rows, b2_out = _run_nocmp(nocmp_content)
        c1_rows, c1_out = _run_nocmp(same_slq)
        c2_rows, c2_out = _run_nocmp(no_same_slq)
        d1_rows, d1_out = _run_nocmp(chat_split)
        b1_reqs = _idxin_requests(b1_rows)
        b2_reqs = _idxin_requests(b2_rows)
        d1_reqs = _idxin_requests(d1_rows)

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
             "miss_knorm=[198.935]" in c1_out and "hit_knorm=[199" in c1_out
             and "=> ROPE-PHASE" in c1_out
             and "=> CONTENT/WINDOW" not in c1_out),
            ("[IDXIN-ROPE] no-same-slq fallback is STATED",
             "no same-slq HIT" in c2_out),
            ("[IDXIN-ROPE] no-same-slq fallback still ROPE-PHASE",
             "=> ROPE-PHASE" in c2_out),
            ("[IDXIN-SPLIT] chat/completions banner (5 requests, HIT=5)",
             "[IDXIN-SPLIT] source=chat/completions requests=5 hit_idx=5"
             in d1_out),
            ("chat boundary split picks request 5 as HIT",
             _idxin_split_info(d1_rows) == ("chat/completions", 5, 5)),
            ("chat boundary split MISS=req4 (prefill before the HIT)",
             _idxin_miss_index(d1_reqs) == 4),
            ("[IDXIN-ROPE] chat-split HIT -> ROPE-PHASE",
             "miss_slq=17523" in d1_out and "hit_slq=17523" in d1_out
             and "=> ROPE-PHASE" in d1_out
             and "=> CONTENT/WINDOW" not in d1_out),
            ("[IDXIN-SPLIT] non-chat logs keep lastpos/stream source",
             _idxin_split_source(idxin) == "lastpos"
             and _idxin_split_source(b1_rows) == "stream"),
            ("[IDXIN-ROPE] per-request knorm+ropephase printed",
             "knorm[" in b1_out and "ropephase[" in b1_out),
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

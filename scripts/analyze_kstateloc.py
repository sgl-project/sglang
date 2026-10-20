#!/usr/bin/env python3
"""Analyze DSV4 [KSTATELOC] / [XDIFF] to decide MISS-vs-HIT state addressing.

Probes (sglang forward_compress, ascend_dsv4_backend.py):
  [KSTATELOC]  env SGL_DSV4_STATELOC_DUMP=1  (kernel-side, compressor_block_vec.h)
  [XDIFF]      env SGL_DSV4_XDIFF=1          (op input x)

[KSTATELOC] has NO layer id, and different layers use different state pools, so a
same-position slot differs across layers LEGITIMATELY. To avoid false positives,
this tool groups lines into contiguous CALL BLOCKS (one block = one layer's op
call) and supports comparing TWO runs block-by-block (layer order is stable).

Usage:
  # single log: show the block(s) touching the window, per block
  python analyze_kstateloc.py /tmp/run.log
  # miss vs hit: two logs, aligned block-by-block
  python analyze_kstateloc.py /tmp/miss.log --log2 /tmp/hit.log
  # A/B from ONE log: is the miss!=hit root the COMPRESSOR (A) or the
  # c4-indexer tie-break (B)?  Compares [IDXIN].klog vs [CMPIDX].md5 per layer.
  python analyze_kstateloc.py /tmp/run.log --ab              # default layer 2
  python analyze_kstateloc.py /tmp/run.log --ab --layers all  # every layer
  python analyze_kstateloc.py --selftest
"""
import argparse
import collections
import re

RE_RW = re.compile(
    r"\[KSTATELOC\]\s+op=([RW])\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
)
RE_L = re.compile(r"\[KSTATELOC\]\s+op=L\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)")
RE_XDIFF = re.compile(
    r"\[XDIFF\]\s+layer=(\d+)\s+idx=(\d+)\s+start=\[([^\]]*)\]\s+ntok=(\d+)\s+"
    r"xshape=\(([^)]*)\)\s+xsum=([-\d.eE+]+)\s+xabsmax=([-\d.eE+]+)\s+xmd5=([0-9a-f]+)"
)


def _rw_from(ln):
    m = RE_RW.search(ln)
    if not m:
        return None
    op, b, p, col, sl, blk, row = m.groups()
    return (op, int(b), int(p), int(col), int(sl), int(blk), int(row))


def _xdiff_from(ln):
    m = RE_XDIFF.search(ln)
    if not m:
        return None
    layer, idx, start, ntok, shape, s, am, md5 = m.groups()
    return (int(layer), int(idx), start, int(ntok), shape, s, am, md5)


def parse(path):
    """Return (blocks, lefts, xdiff). blocks = list of lists of rw-tuples,
    split on any non-[KSTATELOC] line (a block == one contiguous op call)."""
    blocks, cur, lefts, xdiff = [], [], [], []
    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            if "[KSTATELOC]" in ln:
                rw = _rw_from(ln)
                if rw is not None:
                    cur.append(rw)
                    continue
                m = RE_L.search(ln)
                if m:
                    lefts.append(tuple(int(x) for x in m.groups()))
                    continue
                # unknown KSTATELOC line -> flush block
            else:
                xd = _xdiff_from(ln)
                if xd is not None:
                    xdiff.append(xd)
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    return blocks, lefts, xdiff


def _map_in_window(block, lo, hi):
    mp = collections.OrderedDict()
    for op, b, p, col, sl, blk, row in block:
        if lo <= p < hi:
            mp.setdefault(p, []).append((op, col, sl, blk, row))
    return mp


def single(path, lo, hi, verbose=False):
    blocks, lefts, xdiff = parse(path)
    agg = collections.defaultdict(list)  # pos -> [(tableColumn, stateLoc)]
    for bi, block in enumerate(blocks):
        for op, b, p, col, sl, blk, row in block:
            if lo <= p < hi:
                agg[p].append((col, sl))
    print(f"{path}: blocks={len(blocks)} XDIFF={len(xdiff)}  window[{lo},{hi})")
    if not agg:
        print("  (no KSTATELOC in window)")
    # tableColumn separates MISS (large col, start_pos~0) from HIT (small col).
    for p in sorted(agg):
        pairs = sorted(set(agg[p]))
        cols = sorted({c for c, _ in pairs})
        print(f"  pos={p}: " + "  ".join(f"(col={c},slot={s})" for c, s in pairs))
        if verbose:
            print(f"      distinct tableColumn={cols}")
    _show_xdiff(xdiff)


def diff2(l1, l2, lo, hi):
    b1, _, x1 = parse(l1)
    b2, _, x2 = parse(l2)
    print(f"{l1}: blocks={len(b1)}   {l2}: blocks={len(b2)}")
    n = min(len(b1), len(b2))
    if len(b1) != len(b2):
        print(f"WARN block counts differ ({len(b1)} vs {len(b2)}); aligning first {n}")
    diffs = 0
    for i in range(n):
        m1 = _map_in_window(b1[i], lo, hi)
        m2 = _map_in_window(b2[i], lo, hi)
        if not m1 and not m2:
            continue
        for p in sorted(set(m1) | set(m2)):
            s1 = sorted({v[2] for v in m1.get(p, [])})
            s2 = sorted({v[2] for v in m2.get(p, [])})
            if s1 != s2:
                diffs += 1
                print(f"  block#{i} pos={p}: miss={s1} hit={s2}  <== DIFFERS")
    print(f"\n== block-aligned positions with DIFFERENT stateLoc (miss vs hit): {diffs}")
    _show_xdiff(x2 if x2 else x1, tag="log2" if x2 else "log1")


RE_BATCH = re.compile(r"(Prefill|Decode) batch.*?#cached-token:\s*(\d+)")


def phases(path, lo, hi):
    """Segment the log at scheduler 'Prefill/Decode batch ... #cached-token:N'
    lines, so the SAME layer under MISS (cached=0) vs HIT (cached>0) can be
    compared WITHOUT a rebuiid. Prints per-phase window stateLoc + input hash."""
    segs = []            # list of (phase_label, [rw...], [xdiff...])
    label, rws, xds = None, [], []
    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            m = RE_BATCH.search(ln)
            if m:
                if rws or xds:
                    segs.append((label, rws, xds))
                label, rws, xds = f"{m.group(1)} cached={m.group(2)}", [], []
                continue
            rw = _rw_from(ln)
            if rw is not None:
                rws.append(rw)
                continue
            xd = _xdiff_from(ln)
            if xd is not None:
                xds.append(xd)
        if rws or xds:
            segs.append((label, rws, xds))
    print(f"{path}: {len(segs)} phase-segments  window[{lo},{hi})")
    for label, rws, xds in segs:
        per = collections.defaultdict(set)
        for op, b, p, col, sl, blk, row in rws:
            if lo <= p < hi:
                per[p].add(sl)
        if not per and not xds:
            continue
        print(f"\n-- {label}")
        for p in sorted(per):
            print(f"   pos={p} stateLoc={sorted(per[p])}")
        if xds:
            xd = collections.defaultdict(list)
            for layer, idx, start, ntok, shape, s, am, md5 in xds:
                xd[layer].append(md5)
            for layer in sorted(xd):
                uniq = list(collections.OrderedDict.fromkeys(xd[layer]))
                print(f"   [XDIFF] layer={layer}: {'IDENTICAL' if len(uniq) == 1 else 'DIFFERS(' + str(len(uniq)) + ')'}")


def tagged(path, lo, hi):
    """No-rebuild disambiguation: attribute each [KSTATELOC] row to the most
    recent [XDIFF] (layer, xmd5). xmd5 differs by request (MISS vs HIT), so
    grouping by (layer, xmd5) separates the two requests without a layer id in
    the kernel print. Assumes [XDIFF] precedes the op's [KSTATELOC] in the log."""
    cur_layer, cur_md5 = None, None
    buckets = collections.defaultdict(lambda: collections.defaultdict(set))
    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            xd = _xdiff_from(ln)
            if xd is not None:
                cur_layer, cur_md5 = xd[0], xd[7]
                continue
            rw = _rw_from(ln)
            if rw is not None:
                op, b, p, col, sl, blk, row = rw
                if lo <= p < hi:
                    buckets[cur_layer][cur_md5].add(sl)
    print(f"{path}: [XDIFF]-tagged  window[{lo},{hi})")
    for layer in sorted(k for k in buckets if k is not None):
        mds = buckets[layer]
        variants = sorted(mds.items())
        distinct = len({tuple(sorted(v)) for _, v in mds.items()})
        tag = "" if distinct == 1 else "   <== stateLoc differs by request"
        print(f"\n-- layer {layer}: {len(mds)} request-variant(s){tag}")
        for md5, slots in variants:
            print(f"   xmd5={md5[:8]} stateLoc={sorted(slots)}")
    if any(k is None for k in buckets):
        print("\n(note: some KSTATELOC rows preceded any XDIFF -> layer unknown)")


def bycol(path, lo, hi):
    """Group [KSTATELOC] by tableColumn (the state_block_table entry the kernel
    looks up). Reveals each table's (pos -> slot) mapping and whether the
    miss-table vs hit-table differ by a CONSTANT slot offset (benign, e.g. +80)
    or an inconsistent one (table bug)."""
    blocks, _, _ = parse(path)
    per = collections.defaultdict(lambda: collections.defaultdict(set))  # col -> pos -> {slot}
    for block in blocks:
        for op, b, p, col, sl, blk, row in block:
            if lo <= p < hi:
                per[col][p].add(sl)
    print(f"{path}: grouped BY tableColumn  window[{lo},{hi})")
    for col in sorted(per):
        items = per[col]
        multi = {p: s for p, s in items.items() if len(s) > 1}
        print(f"\n-- col={col}  positions={len(items)}" + (f"  multi-slot at {sorted(multi)}" if multi else ""))
        for p in sorted(items):
            print(f"   pos={p} slot={sorted(items[p])}")
    # offset analysis: for each pos, the slot(s) minus the smallest slot
    print("\n== per-pos slot set (offsets vs min) ==")
    bypos = collections.defaultdict(set)
    for col in per:
        for p, s in per[col].items():
            bypos[p] |= s
    for p in sorted(bypos):
        ss = sorted(bypos[p])
        base = ss[0]
        print(f"   pos={p}: slots={ss}  offsets={[s - base for s in ss]}")


def _show_xdiff(xdiff, tag=""):
    if not xdiff:
        return
    by = collections.OrderedDict()
    for layer, idx, start, ntok, shape, s, am, md5 in xdiff:
        by.setdefault(layer, []).append(md5)
    print(f"\n== [XDIFF] {tag}: layer -> x identity ==")
    for layer, md5s in by.items():
        uniq = list(collections.OrderedDict.fromkeys(md5s))
        print(f"   layer={layer}: {'IDENTICAL' if len(uniq) == 1 else 'DIFFERS(' + str(len(uniq)) + ')'} md5={uniq[0]}")


# --- A/B: is the miss!=hit divergence the COMPRESSOR (A) or the c4-INDEXER (B)? ---
# [IDXIN]  layer=<id> ... klog=<md5-of-index-K-op-input> ...   (the c4-indexer
#          index-K op INPUT / "klog", i.e. the COMPRESSOR's output as the
#          indexer consumes it).
# [CMPIDX] layer=<id> ... md5=<top-k md5> tail=[...] blkx=[...]  (the c4-indexer
#          top-k / discrete argmax handed to the sparse attention).
# A single log carries both the MISS and HIT forwards, so per layer:
#   klog>1 distinct  => the compressor output differs (A)
#   klog==1, topk>1  => the compressor is identical, the indexer tie-break differs (B)
TAG_AB = re.compile(r"\[(IDXIN|CMPIDX)\]")
KV = re.compile(r"(\w+)=(\[[^\]]*\]|\([^)]*\)|\S+)")


def parse_ab(path):
    """Return {layer: {'klog': [vals...], 'topk': [md5...]}} from [IDXIN]/[CMPIDX].

    klog = the [IDXIN] ``klog=`` field (index-K op input, the compressor output).
    topk = the [CMPIDX] ``md5=`` field (c4-indexer top-k). Rows without a numeric
    layer (e.g. ``[TAG] skipped: ...``) are ignored."""
    data = collections.defaultdict(lambda: {"klog": [], "topk": []})
    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            m = TAG_AB.search(ln)
            if not m:
                continue
            d = dict(KV.findall(ln))
            layer = d.get("layer")
            if layer is None or not layer.lstrip("-").isdigit():
                continue
            layer = int(layer)
            if m.group(1) == "IDXIN":
                kl = d.get("klog")
                if kl is not None:
                    data[layer]["klog"].append(kl)
            else:  # CMPIDX
                tk = d.get("md5")
                if tk is not None:
                    data[layer]["topk"].append(tk)
    return data


def _uniq(vals):
    """Ordered distinct values (dedup, first-seen order preserved)."""
    return list(collections.OrderedDict.fromkeys(vals))


def _select_layers(data, layers):
    """``all`` => every layer present; else a comma-separated int list."""
    if layers == "all":
        return sorted(data)
    sel = []
    for part in str(layers).split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            sel.append(int(part))
    return sel


def _ab_verdict(nklog, ntopk):
    if nklog == 0 and ntopk == 0:
        return "no probe lines for this layer"
    if nklog > 1 and ntopk > 1:
        return "inconclusive (both differ)"
    if nklog > 1:
        return "(A) compressor"
    if ntopk > 1:
        return "(B) indexer"
    return "same (no divergence captured in this window)"


def ab(path, layers="2"):
    """Judge (A) compressor vs (B) indexer tie-break from a single log."""
    data = parse_ab(path)
    sel = _select_layers(data, layers)
    print(f"{path}: --ab mode  layers={layers}")
    if not sel:
        print("  (no layers selected)")
    for layer in sel:
        d = data.get(layer, {"klog": [], "topk": []})
        kl = _uniq(d["klog"])
        tk = _uniq(d["topk"])
        print(f"\n-- layer {layer}")
        print(
            f"   klog: {len(kl)} distinct variant(s)"
            + (f" md5={kl}" if kl else " (no data)")
        )
        print(
            f"   topk: {len(tk)} distinct variant(s)"
            + (f" md5={tk}" if tk else " (no data)")
        )
        print(f"   VERDICT: {_ab_verdict(len(kl), len(tk))}")


def selftest():
    miss = "\n".join([
        "[KSTATELOC] op=R 0 17534 8 1110 138 6",
        "[KSTATELOC] op=W 0 17534 8 1110 138 6",
        "[SWAW] noise",
        "[KSTATELOC] op=R 0 17534 8 1110 138 6",
    ])
    hit = "\n".join([
        "[KSTATELOC] op=R 0 17534 8 1110 138 6",
        "[KSTATELOC] op=W 0 17534 8 1110 138 6",
        "[SWAW] noise",
        "[KSTATELOC] op=R 0 17534 8 1190 148 6",
    ])
    import tempfile, os
    f1 = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f1.write(miss); f1.close()
    f2 = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f2.write(hit); f2.close()
    # Synthetic A/B log: one line per (layer, miss/hit) so each verdict fires.
    ab_log = "\n".join([
        # layer 2: klog identical, topk differs  -> (B) indexer
        "[IDXIN] layer=2 mode=ForwardMode.PREFILL qblk=[] wblk=[] slq=(0,17523,17523) "
        "slk=(0,17523,17523) klog=aaaaaaaaaaaaaaaa kraw=1 meta=2 bt=3 qshape=(1,1) "
        "wshape=(1,1) kshape=(1,1)",
        "[IDXIN] layer=2 mode=ForwardMode.PREFILL qblk=[] wblk=[] slq=(0,1139,1139) "
        "slk=(0,1139,1139) klog=aaaaaaaaaaaaaaaa kraw=4 meta=5 bt=6 qshape=(1,1) "
        "wshape=(1,1) kshape=(1,1)",
        "[CMPIDX] layer=2 mode=ForwardMode.PREFILL ntok=17523 lastpos=17522 "
        "md5=bbbbbbbbbbbbbbbb tail=[1, 2] blkx=[134:dead]",
        "[CMPIDX] layer=2 mode=ForwardMode.PREFILL ntok=1139 lastpos=1138 "
        "md5=cccccccccccccccc tail=[3, 4] blkx=[8:beef]",
        # layer 3: klog differs, topk identical  -> (A) compressor
        "[IDXIN] layer=3 mode=ForwardMode.PREFILL qblk=[] wblk=[] slq=(0,1,1) "
        "slk=(0,1,1) klog=1111111111111111 kraw=1 meta=2 bt=3 qshape=(1,1) "
        "wshape=(1,1) kshape=(1,1)",
        "[IDXIN] layer=3 mode=ForwardMode.PREFILL qblk=[] wblk=[] slq=(0,1,1) "
        "slk=(0,1,1) klog=2222222222222222 kraw=1 meta=2 bt=3 qshape=(1,1) "
        "wshape=(1,1) kshape=(1,1)",
        "[CMPIDX] layer=3 mode=ForwardMode.PREFILL ntok=1 lastpos=1 "
        "md5=dddddddddddddddd tail=[1] blkx=[0:aa]",
        "[CMPIDX] layer=3 mode=ForwardMode.PREFILL ntok=1 lastpos=1 "
        "md5=dddddddddddddddd tail=[1] blkx=[0:aa]",
        # layer 4: klog and topk both identical  -> same
        "[IDXIN] layer=4 mode=ForwardMode.DECODE qblk=[] wblk=[] slq=(0,1,1) "
        "slk=(0,1,1) klog=3333333333333333 kraw=1 meta=2 bt=3 qshape=(1,1) "
        "wshape=(1,1) kshape=(1,1)",
        "[CMPIDX] layer=4 mode=ForwardMode.DECODE ntok=1 lastpos=5 "
        "md5=eeeeeeeeeeeeeeee tail=[1] blkx=[0:aa]",
        # layer 5: klog and topk both differ  -> inconclusive
        "[IDXIN] layer=5 mode=ForwardMode.PREFILL qblk=[] wblk=[] slq=(0,1,1) "
        "slk=(0,1,1) klog=4444444444444444 kraw=1 meta=2 bt=3 qshape=(1,1) "
        "wshape=(1,1) kshape=(1,1)",
        "[IDXIN] layer=5 mode=ForwardMode.PREFILL qblk=[] wblk=[] slq=(0,1,1) "
        "slk=(0,1,1) klog=5555555555555555 kraw=1 meta=2 bt=3 qshape=(1,1) "
        "wshape=(1,1) kshape=(1,1)",
        "[CMPIDX] layer=5 mode=ForwardMode.PREFILL ntok=1 lastpos=1 "
        "md5=6666666666666666 tail=[1] blkx=[0:aa]",
        "[CMPIDX] layer=5 mode=ForwardMode.PREFILL ntok=1 lastpos=1 "
        "md5=7777777777777777 tail=[1] blkx=[0:bb]",
    ])
    f3 = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f3.write(ab_log); f3.close()
    print("--- single ---")
    single(f1.name, 17520, 17540)
    print("\n--- diff2 ---")
    diff2(f1.name, f2.name, 17520, 17540)
    print("\n--- ab (default layer 2) ---")
    ab(f1.name, "2")
    print("\n--- ab (all layers) ---")
    ab(f3.name, "all")
    data = parse_ab(f3.name)
    expected = {
        2: "(B) indexer",
        3: "(A) compressor",
        4: "same (no divergence captured in this window)",
        5: "inconclusive (both differ)",
    }
    for ly, want in sorted(expected.items()):
        got = _ab_verdict(len(_uniq(data[ly]["klog"])), len(_uniq(data[ly]["topk"])))
        assert got == want, f"ab layer {ly}: got {got!r}, want {want!r}"
    print("ab verdicts: OK")
    os.unlink(f1.name); os.unlink(f2.name); os.unlink(f3.name)
    print("\nSELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--log2")
    ap.add_argument("--lo", type=int, default=17520)
    ap.add_argument("--hi", type=int, default=17540)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--tagged", action="store_true")
    ap.add_argument("--bycol", action="store_true")
    ap.add_argument("--ab", action="store_true",
                    help="judge (A) compressor vs (B) indexer from one log")
    ap.add_argument("--layers", default="2",
                    help="--ab layer(s): N, N,M, or 'all' (default 2)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.log:
        ap.error("log path required (or --selftest)")
    if a.ab:
        ab(a.log, a.layers)
    elif a.bycol:
        bycol(a.log, a.lo, a.hi)
    elif a.tagged:
        tagged(a.log, a.lo, a.hi)
    elif a.phases:
        phases(a.log, a.lo, a.hi)
    elif a.log2:
        diff2(a.log, a.log2, a.lo, a.hi)
    else:
        single(a.log, a.lo, a.hi, verbose=a.verbose)


if __name__ == "__main__":
    main()

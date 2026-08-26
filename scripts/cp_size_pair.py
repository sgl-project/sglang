#!/usr/bin/env python3
"""Offline analyzer for the merged sglang prefill log (row-overwrite case, §24).

Input: ONE log file (all ranks interleaved), lines like::

    [2026-08-24 06:28:55 DP5 ATTN_CP0 TP10 EP10] [cp-size] interleave seq=1403 rank=0/2 max_len=6014 ... ctx=12028
    [2026-08-24 06:28:56 DP5 ATTN_CP1 TP11 EP11] [mf-raw] rid=... slot=16 seg=[122880,134012): n_dirty=5520/11132 ...

CP partners = same DP number, different ATTN_CP number (explicit in the
prefix -- no inference). The analyzer prints:

  1. census        -- line counts per tag / kind, torn [cp-size] lines.
  2. invariants    -- single-rank overflow conditions (no pairing).
  3. pair diverg.  -- per (tag, DP): the CP pair's streams aligned by
     occurrence position; any position where ctx differs (the two ranks
     built different attn_cp_metadata for the SAME forward) or a field
     differs (same batch, differently sized destination) is the overflow
     precondition, reported with both ranks' values and timestamps.
  4. correlation   -- for every [mf-raw] dirty-row hit: whether its DP's CP
     pair had diverged EARLIER in the file, with the nearest divergence
     details. All-hits-have-prior-divergence == conviction; zero
     divergences at all == A family exonerated in this run.
  5. [mf-trans] drop counts per rank (downstream impact of the garbage rows).
  6. retain verdict -- [mf-er] / [mf-raw] / [mf-fp] triage for an
     SGLANG_MF_EAGLE_RETAIN run, mapped onto the §24.10.4 pre-registered
     outcomes (FOUND = source identified; silent = beyond retained family
     or column-stride blind; no reproduction = retention masks -> sham).
  7. cross-hit payload compared on GLOBAL token offsets (the §24.10.3
     position-indexed-stream check; the old window-overlap "distinct"
     verdict was a misalignment artifact and is superseded).
  8. [mf-trap] catches -- the per-layer req_to_token write trap's layer
     windows, correlated with prior pair divergence and later [mf-raw]
     victims (same rank + slot == the caught writer survived to send).
     Labels "gap:post-target"/"gap:post-draft" are the §24.23 gap probes:
     catches there convict the same-request TAIL windows after the last
     layer (target post-loop: CP all-gather/norm/lm_head/logits capture;
     draft prefill forward family), NOT a layer window.
  9. [mf-reg] refused regions -- exact VAs the fabric engine REFUSED to
     register (bisect from transfer_engine.py); their transfers are
     dropped with [mf-trans] src-outside lines.

Exit 1 if any divergence / invariant fired, a [mf-fp] FOUND surfaced, a
[mf-trap] catch or [mf-reg] refusal is present, else 0 (correlation is
reported either way -- it needs [mf-raw] hits to be meaningful).
"""

import argparse
import ast
import collections
import re
import sys

PREFIX_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*?"
    r"\s+DP(?P<dp>\d+) ATTN_CP(?P<cp>\d+) TP(?P<tp>\d+)(?: EP\d+)?\]\s+(?P<rest>.*)$"
)
MFTRAP_RE = re.compile(
    r"\[mf-trap\]\s+caught after=(?P<label>\S+)\s+row=(?P<row>\d+)\s+"
    r"col=(?P<col>\d+)\s+val=(?P<val>-?\d+)\s+seq=(?P<seq>-?\d+)\s+"
    r"window=(?P<w0>\d+)x(?P<w1>\d+)"
)
# layer_trap.py send-path split: pre- vs post-translate dirty counts of the
# SAME r2t segment (emitted inside the [mf-raw] branch, piggyback sync).
MFTRAPSEND_RE = re.compile(
    r"\[mf-trap\] send pre-translate rid=(?P<rid>\S+) slot=(?P<slot>\d+) "
    r"seg=\[(?P<s0>\d+),(?P<s1>\d+)\) n_dirty_pre=(?P<pre>\d+) "
    r"n_dirty_post=(?P<post>\d+) n_tot=(?P<tot>\d+)"
)
# §24.40 Run 24 address-space discriminator: dirty byte address vs
# registered candidate ranges (alias-vs-OOB).
MFTRAPADDR_RE = re.compile(r"\[mf-trap\] addr (?P<body>dirty=.*)$")
# §24.41 Run 25: nearest gather-site buffers around the dirty address
# (rolling registry in layer_trap.py; the lines follow the catch/addr
# lines of the same catch).
MFTRAPGSITES_RE = re.compile(
    r"\[mf-trap\] gsites tag=(?P<tag>\S+) (?P<role>in|out)"
    r"=\[(?P<lo>0x[0-9a-f]+),(?P<hi>0x[0-9a-f]+)\) dist=(?P<dist>[+-]?\d+) "
    r"shape=(?P<shape>\S+) dtype=(?P<dtype>\S+) age=(?P<age>[\d.]+)s"
)
# §24.41 Run 25: one-shot sorted address map of the registered pools
# (layer_trap_register_tensors). Answers "which pool is physically
# adjacent to r2t" even in a catchless run.
MFTRAPLAYOUT_RE = re.compile(
    r"\[mf-trap\] layout name=(?P<name>\S+) lo=(?P<lo>0x[0-9a-f]+) "
    r"hi=(?P<hi>0x[0-9a-f]+)"
)
CPSIZE_RE = re.compile(
    r"\[cp-size\]\s+(?P<tag>\S+)\s+seq=(?P<seq>\d+)\s+rank=(?P<rank>\d+)/(?P<size>\d+)"
    r"\s+(?P<body>.*?)\s+ctx=(?P<ctx>\S+)"
)
MFRAW_RE = re.compile(
    r"\[mf-raw\]\s+rid=(?P<rid>\S+)\s+slot=(?P<slot>\d+)\s+seg=\[(?P<seg0>\d+),(?P<seg1>\d+)\)"
    r":\s+n_dirty=(?P<ndirty>\d+)/(?P<ntot>\d+)\s+n_sentinel=(?P<nsent>\d+)"
    r"\s+first_dirty_off=(?P<fdo>\d+)"
    r"(?:\s+slots\[(?P<s0>\d+):(?P<s1>\d+)\]=\[(?P<slots>[^\]]*)\])?"
    r"(?:\s+floats=\[(?P<floats>[^\]]*)\])?"
    r"(?:\s+pages24=\[(?P<pages>[^\]]*)\])?"
)
MFER_RE = re.compile(
    r"\[mf-er\] eagle retention ON \(budget (?P<budget>\d+) MB\)"
)
# §24.20 audit line: names held by the per-name ring at probe time.
MFERNAMES_RE = re.compile(r"\[mf-fp\] er names=(?P<names>\S+)")
# Draft-side retain sites (dflash.py / dflash_worker_v2.py, §24.16): their
# absence while target-side names are present = those files not deployed.
DRAFT_ER_NAMES = ("draft.embed", "draft.ctx_hidden", "draft.kv0", "draft.kvL")
MFFP_FOUND_RE = re.compile(
    r"\[mf-fp\] payload (?:sequence|stride) FOUND x(?P<count>\d+)"
    r"(?:\s+col=(?P<col>\d+)\s+row=(?P<row>\d+))?\s+in\s+(?P<name>\S+)"
)
MFFP_PROBED_RE = re.compile(r"\[mf-fp\] probed (?P<n>\d+) candidates")
MFFP_RAISED_RE = re.compile(r"\[mf-fp\] probe raised on (?P<name>\S+)")
# transfer_engine.py registration bisect: exact VAs the fabric engine
# REFUSED (per-region re-register after a failed batch). Confront with
# [mf-trans] drop srcs / dirty-row pool VAs when correlating.
MFREG_FAIL_RE = re.compile(
    r"\[mf-reg\] FAIL ptr=(?P<ptr>0x[0-9a-f]+) len=(?P<len>\d+) end=(?P<end>0x[0-9a-f]+)"
)
MFREG_SUM_RE = re.compile(
    r"\[mf-reg\] summary failed=(?P<failed>\d+)/(?P<total>\d+) "
    r"bytes_failed=(?P<bytes>\d+)"
)
# §24.29 discriminator (1): [mf-scatter] fires only when a clamped scatter
# slot saw OOB locs (prefill.py reads get_scatter_oob_counts inside the
# [mf-raw] piggyback D2H). Slots: indexer / swa / compress / compressor-state.
# §24.41 Run 26: layer_trap._drain also emits a catch-time variant (the
# [mf-raw] gate can stay silent while catches fire); same body format, and
# it additionally carries the UNCLAMPED npu-* scatter audit slots
# (npu-fia-kv / npu-kv-k / npu-mla-k / npu-idx-k, each :neg / :hi).
MFSCATTER_RE = re.compile(
    r"\[mf-scatter\] (?:clamped|catch-time) OOB scatters: (?P<body>\{.*\})"
)

CP_TAGS = {
    "zigzag",
    "interleave",
    "overlap",
    "reorg",
    "reorg-kv",
    "rr",
    "dsa-gather",
    # communicator_dsa_cp.py: the DSA-CP pair's own all_gather/reduce_scatter
    # on the persistent local dp buffer (attn_cp group proper).
    "dsa-cp-ag",
    "dsa-cp-rs",
}
# NOT pairable: under DP each rank's a2a row count legitimately differs, so
# the a2a-norm line (deepep _dispatch_core flight recorder) is census /
# flight-recorder only. Cross-rank pairing on it would always false-positive.
# The overlap-launch emitter tags carry an event-key suffix ("overlap:gather",
# "overlap:combine", ...); they are pairable but must not cross-pair between
# different event keys, so membership is tested on the base while the pairing
# stream stays keyed by the FULL tag.
OVERLAP_PREFIX = "overlap:"
SCATTER_TAGS = {"dp-rs", "dp-rsv-async", "dsv4-combine"}
MAX_REPORT = 40


def parse(files):
    """-> (cp, mfraw, mftrans, census, torn, mfer, mffp_found, mfer_names, mftrap)."""
    cp = []
    mfraw = []
    mftrans = collections.Counter()
    census = collections.Counter()
    torn = 0
    mfer = []
    mffp_found = []
    mfer_names = set()
    mftrap = []
    mfreg_fail = []
    mftrap_send = []
    mfscatter = []
    mflayout = []
    order = 0
    for path in files:
        with open(path, errors="replace") as fh:
            for line in fh:
                if "[cp-size]" in line:
                    pm = PREFIX_RE.match(line)
                    m = CPSIZE_RE.search(line)
                    if not m:
                        torn += 1
                        continue
                    fields = {}
                    for tok in m.group("body").split():
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            fields[k] = v
                    cp.append(
                        {
                            "order": order,
                            "ts": pm.group("ts") if pm else "",
                            "dp": int(pm.group("dp")) if pm else -1,
                            "cp": int(pm.group("cp")) if pm else int(m.group("rank")),
                            "tp": int(pm.group("tp")) if pm else -1,
                            "tag": m.group("tag"),
                            "seq": int(m.group("seq")),
                            "ctx": m.group("ctx"),
                            "fields": fields,
                        }
                    )
                    census[f"cp-size:{m.group('tag')}"] += 1
                elif "[mf-raw]" in line:
                    pm = PREFIX_RE.match(line)
                    m = MFRAW_RE.search(line)
                    if m and pm:
                        mfraw.append(
                            {
                                "order": order,
                                "ts": pm.group("ts"),
                                "dp": int(pm.group("dp")),
                                "cp": int(pm.group("cp")),
                                "tp": int(pm.group("tp")),
                                **{k: v for k, v in m.groupdict().items()},
                            }
                        )
                        census["mf-raw"] += 1
                elif "[mf-trans]" in line:
                    pm = PREFIX_RE.match(line)
                    if pm:
                        mftrans[(int(pm.group("dp")), int(pm.group("cp")))] += 1
                        census["mf-trans"] += 1
                elif "[mf-er]" in line:
                    pm = PREFIX_RE.match(line)
                    m = MFER_RE.search(line)
                    if pm and m:
                        mfer.append(
                            {
                                "ts": pm.group("ts"),
                                "dp": int(pm.group("dp")),
                                "cp": int(pm.group("cp")),
                                "tp": int(pm.group("tp")),
                                "budget": int(m.group("budget")),
                            }
                        )
                        census["mf-er"] += 1
                elif "[mf-fp]" in line:
                    pm = PREFIX_RE.match(line)
                    census["mf-fp"] += 1
                    m = MFFP_FOUND_RE.search(line)
                    if m and pm:
                        mffp_found.append(
                            {
                                "ts": pm.group("ts"),
                                "dp": int(pm.group("dp")),
                                "cp": int(pm.group("cp")),
                                "tp": int(pm.group("tp")),
                                "count": int(m.group("count")),
                                "name": m.group("name"),
                            }
                        )
                        census["mf-fp:found"] += 1
                    elif MFFP_PROBED_RE.search(line):
                        census["mf-fp:probed"] += 1
                    elif MFFP_RAISED_RE.search(line):
                        census["mf-fp:raised"] += 1
                    else:
                        nm = MFERNAMES_RE.search(line)
                        if nm and nm.group("names") != "-":
                            mfer_names.update(
                                n for n in nm.group("names").split(",") if n
                            )
                            census["mf-fp:ernames"] += 1
                elif "[mf-reg]" in line:
                    pm = PREFIX_RE.match(line)
                    m = MFREG_FAIL_RE.search(line)
                    if m:
                        mfreg_fail.append(
                            {
                                "ts": pm.group("ts") if pm else "",
                                "dp": int(pm.group("dp")) if pm else -1,
                                "cp": int(pm.group("cp")) if pm else -1,
                                "ptr": m.group("ptr"),
                                "len": int(m.group("len")),
                                "end": m.group("end"),
                            }
                        )
                        census["mf-reg:fail"] += 1
                    elif MFREG_SUM_RE.search(line):
                        census["mf-reg:summary"] += 1
                    else:
                        census["mf-reg:ok"] += 1
                elif "[mf-scatter]" in line:
                    pm = PREFIX_RE.match(line)
                    m = MFSCATTER_RE.search(line)
                    if m:
                        mfscatter.append(
                            {
                                "ts": pm.group("ts") if pm else "",
                                "dp": int(pm.group("dp")) if pm else -1,
                                "cp": int(pm.group("cp")) if pm else -1,
                                "tp": int(pm.group("tp")) if pm else -1,
                                "body": m.group("body"),
                            }
                        )
                        census["mf-scatter"] += 1
                elif "[mf-trap]" in line:
                    pm = PREFIX_RE.match(line)
                    m = MFTRAP_RE.search(line)
                    ms = MFTRAPSEND_RE.search(line) if not m else None
                    if m and pm:
                        mftrap.append(
                            {
                                "order": order,
                                "ts": pm.group("ts"),
                                "dp": int(pm.group("dp")),
                                "cp": int(pm.group("cp")),
                                "tp": int(pm.group("tp")),
                                "label": m.group("label"),
                                "row": int(m.group("row")),
                                "col": int(m.group("col")),
                                "val": int(m.group("val")),
                                "seq": int(m.group("seq")),
                                "w0": m.group("w0"),
                                "w1": m.group("w1"),
                            }
                        )
                        census["mf-trap"] += 1
                        if m.group("label").startswith("gap:"):
                            census["mf-trap:gap"] += 1
                        elif m.group("label").startswith("glue:"):
                            census["mf-trap:glue"] += 1
                        elif m.group("label").startswith("GL:"):
                            census["mf-trap:gl"] += 1
                        elif m.group("label").startswith("PL:"):
                            census["mf-trap:pl"] += 1
                    elif ms:
                        mftrap_send.append(
                            {
                                "order": order,
                                "ts": pm.group("ts") if pm else "",
                                "dp": int(pm.group("dp")) if pm else -1,
                                "cp": int(pm.group("cp")) if pm else -1,
                                "tp": int(pm.group("tp")) if pm else -1,
                                **{k: v for k, v in ms.groupdict().items()},
                            }
                        )
                        census["mf-trap:send"] += 1
                    elif "addr dirty=" in line:
                        # §24.40 Run 24: address-space discriminator line that
                        # FOLLOWS its catch line; attach to the last catch.
                        ma = MFTRAPADDR_RE.search(line)
                        if ma and mftrap:
                            mftrap[-1]["addr"] = ma.group("body")
                            census["mf-trap:addr"] += 1
                    elif "gsites" in line:
                        # §24.41 Run 25: nearest gather-site buffer lines that
                        # follow the catch/addr lines; attach to the last catch.
                        mg = MFTRAPGSITES_RE.search(line)
                        if mg and mftrap:
                            mftrap[-1].setdefault("gsites", []).append(
                                mg.groupdict()
                            )
                            census["mf-trap:gsites"] += 1
                        elif "(registry empty)" in line:
                            census["mf-trap:gsites:empty"] += 1
                    elif "layout name=" in line:
                        ml = MFTRAPLAYOUT_RE.search(line)
                        if ml:
                            mflayout.append(
                                {
                                    "dp": int(pm.group("dp")) if pm else -1,
                                    "cp": int(pm.group("cp")) if pm else -1,
                                    "tp": int(pm.group("tp")) if pm else -1,
                                    "name": ml.group("name"),
                                    "lo": int(ml.group("lo"), 16),
                                    "hi": int(ml.group("hi"), 16),
                                }
                            )
                            census["mf-trap:layout"] += 1
                    elif "fwd-freeze" in line:
                        # §24.35 liveness marker: the pre-cache forward-stream
                        # freeze discriminator is deployed in this run; the
                        # glue catch interpretations below change accordingly.
                        census["mf-trap:freeze"] += 1
                    elif "gather-freeze" in line:
                        # §24.39 Run 23 liveness marker: gather-stream drain
                        # active at every pool snapshot.
                        census["mf-trap:gather-freeze"] += 1
                    elif "armed" in line:
                        # Liveness marker: trap hooks deployed & env reached.
                        # Its absence in a catchless log = the trap NEVER RAN
                        # (stale deploy / env miss), and every "silent" verdict
                        # below is void.
                        census["mf-trap:armed"] += 1
                order += 1
    return cp, mfraw, mftrans, census, torn, mfer, mffp_found, mfer_names, mftrap, mfreg_fail, mftrap_send, mfscatter, mflayout


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def local_invariants(tag, fields):
    hits = []
    if tag == "dp-ar":
        start, num = to_int(fields.get("start")), to_int(fields.get("num"))
        buf, rows = to_int(fields.get("buf_rows")), to_int(fields.get("in_rows"))
        if start is not None and num is not None and buf is not None:
            if start + num > buf:
                hits.append(f"OOB-WRITE start+num={start + num} > buf_rows={buf}")
        if num is not None and rows is not None and num != rows:
            hits.append(f"ROWS!=REPORTED in_rows={rows} num={num}")
    if tag == "dp-agv":
        s, buf = to_int(fields.get("sum")), to_int(fields.get("buf_rows"))
        mine, rows = to_int(fields.get("mine")), to_int(fields.get("in_rows"))
        if s is not None and buf is not None and s != buf:
            hits.append(f"SUM!=BUF sum={s} buf_rows={buf}")
        if mine is not None and rows is not None and mine != rows:
            hits.append(f"MINE!=IN_ROWS mine={mine} in_rows={rows}")
    if tag in SCATTER_TAGS:
        sizes = [to_int(x) for x in fields.get("sizes", "").split(",") if x]
        out_rows = to_int(fields.get("out_rows"))
        if sizes and out_rows is not None and max(sizes) > out_rows:
            hits.append(f"OOB-WRITE max(sizes)={max(sizes)} > out_rows={out_rows}")
        s, in_rows = to_int(fields.get("sum")), to_int(fields.get("in_rows"))
        if s is not None and in_rows is not None and s > in_rows:
            hits.append(f"READ-OVERSUM sum={s} > in_rows={in_rows}")
    if tag == "dsa-cp-rs":
        # Equal-chunk reduce_scatter: tensor_split(cp_size)[cp_rank] must match
        # the flat in_rows // cp_size every rank assumes, else chunks unequal.
        split_rows, out_rows = to_int(fields.get("split_rows")), to_int(
            fields.get("out_rows")
        )
        if split_rows is not None and out_rows is not None and split_rows != out_rows:
            hits.append(
                f"UNEQUAL-CHUNK split_rows={split_rows} != out_rows={out_rows}"
                " (in_rows % cp_size != 0)"
            )
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="merged sglang log file(s)")
    ap.add_argument("--max-report", type=int, default=MAX_REPORT)
    args = ap.parse_args()

    cp, mfraw, mftrans, census, torn, mfer, mffp_found, mfer_names, mftrap, mfreg_fail, mftrap_send, mfscatter, mflayout = parse(args.logs)
    if not cp and not mfraw and not mfreg_fail and not mftrap_send:
        print("nothing recognizable found (no [cp-size]/[mf-raw] lines)")
        return 1

    fired = 0

    print("== census ==")
    for k in sorted(census):
        print(f"  {k:24s} {census[k]}")
    if torn:
        print(f"  torn/unparsable [cp-size] lines: {torn}")

    # ---- §24.41 Run 25: one-shot registered-pool layout map ---------------
    # Sorted by address per rank; prints r2t's physical neighbors so the
    # catch-time "no registered candidate overlaps" verdict can be read
    # against the true adjacency (a catch just past pool X's hi = X's
    # scatter overflow).
    if mflayout:
        print("\n== [mf-trap] layout: registered pool map (per rank, sorted) ==")
        by_rank = collections.defaultdict(list)
        for e in mflayout:
            by_rank[(e["dp"], e["cp"], e["tp"])].append(e)
        for (_ldp, _lcp, _ltp), entries in sorted(by_rank.items()):
            entries.sort(key=lambda e: e["lo"])
            print(f"  DP{_ldp} CP{_lcp} TP{_ltp}:")
            for i, e in enumerate(entries):
                gap_prev = (
                    f" (+{e['lo'] - entries[i-1]['hi']} past {entries[i-1]['name']})"
                    if i > 0
                    else ""
                )
                print(
                    f"    {e['name']:12s} [{e['lo']:#x},{e['hi']:#x}) "
                    f"size={e['hi'] - e['lo']}{gap_prev}"
                )

    # ---- EAGLE_RETAIN run verdict (§24.10.4 pre-registered outcomes) -------
    if census["mf-er"] or census["mf-fp"]:
        print("\n== EAGLE_RETAIN run verdict ==")
        if mfer or mfer_names:
            ranks = sorted({(e["dp"], e["cp"], e["tp"]) for e in mfer})
            budgets = sorted({e["budget"] for e in mfer})
            src = "[mf-er] ON" if mfer else "[mf-fp] er names line"
            print(
                f"  retention: ACTIVE ({src}; {len(ranks)} rank(s), budgets"
                f" {budgets} MB; held names: {sorted(mfer_names) or '?'})"
            )
            missing_draft = [n for n in DRAFT_ER_NAMES if n not in mfer_names]
            if mfer_names and missing_draft:
                fired += 1
                print(
                    "  WARNING: draft-side retain sites MISSING from held"
                    f" names: {missing_draft} -- dflash.py /"
                    " dflash_worker_v2.py retain hooks not deployed (or draft"
                    " path idle); the draft-family exclusion is VOID for"
                    " those names."
                )
            if census["mf-fp:raised"]:
                print(
                    f"  matcher health: {census['mf-fp:raised']} probe-raised"
                    " line(s) -- candidates listed there were NOT probed;"
                    " their exclusion does not hold."
                )
        else:
            print(
                "  retention: NOT activated (no [mf-er] line and no non-empty"
                " er names line) -- env did not reach the scheduler, or a"
                " legacy log; the retain-family verdict below is VOID."
            )
        print(f"  reproduction: {census['mf-raw']} [mf-raw] hit(s)")
        if mffp_found:
            fired += 1
            print(f"  FOUND: {len(mffp_found)} payload match(es):")
            for f in mffp_found:
                print(
                    f"    {f['ts']} DP{f['dp']} CP{f['cp']} TP{f['tp']}: "
                    f"{f['name']} x{f['count']}"
                )
            print(
                "  VERDICT (outcome 1): payload source IDENTIFIED by"
                " retain-then-fingerprint. The named tensor is the clobber"
            )
            print(
                "  payload; next: find the writer that misdirects it into the"
                " req_to_token rows."
            )
        elif mfer and census["mf-raw"] == 0:
            print(
                "  VERDICT (outcome 3): no reproduction with retention ON ->"
                " retention masks the bug (M1 pinning / M2 timing)."
            )
            print(
                f"  workload check: {len(cp)} [cp-size] lines vs ~24.5K in the"
                " reproducing runs -- a much smaller count means the run may"
            )
            print(
                "  never have reached the trigger; confirm before concluding"
                " masking. next: SGLANG_MF_RETAIN_NOP=1 sham run (splits M1/M2)."
            )
        elif mfer or mfer_names:
            print(
                "  VERDICT (outcome 2): reproduced with retention ON but no"
                " FOUND -> source beyond the retained family, OR a"
            )
            print(
                "  column-stride (1 elem/token) read of a [T,H] tensor that"
                " the contiguous matcher cannot see. Valid ONLY with no"
                " draft-MISSING warning and no probe-raised lines above."
            )

    print("\n== local invariants (single rank, no pairing) ==")
    inv = 0
    for r in cp:
        for hit in local_invariants(r["tag"], r["fields"]):
            inv += 1
            fired += 1
            if inv <= args.max_report:
                print(f"  [{r['tag']}] DP{r['dp']} CP{r['cp']} seq={r['seq']} ctx={r['ctx']}: {hit}")
    if inv == 0:
        print("  (none)")
    elif inv > args.max_report:
        print(f"  ... {inv - args.max_report} more")

    # ---- cross-tag, single rank: dsa-cp-rs consumes dsa-cp-ag's stream ----
    # The gather fills the persistent full-stream buffer; the reduce_scatter is
    # its symmetric inverse on the same buffer within one forward. A rank whose
    # rs in_rows disagrees with its own latest ag out_rows mixed buffers from
    # two different batches -- itself a finding, no pairing needed.
    ag_by_rank = {}
    xtag = 0
    for r in cp:
        key = (r["dp"], r["cp"])
        if r["tag"] == "dsa-cp-ag":
            ag_by_rank[key] = r
        elif r["tag"] == "dsa-cp-rs" and key in ag_by_rank:
            ag = ag_by_rank[key]
            ag_out = to_int(ag["fields"].get("out_rows"))
            rs_in = to_int(r["fields"].get("in_rows"))
            if ag_out is not None and rs_in is not None and ag_out != rs_in:
                xtag += 1
                fired += 1
                if xtag <= args.max_report:
                    print(
                        f"  DP{r['dp']} CP{r['cp']}: rs in_rows={rs_in} (seq={r['seq']})"
                        f" != latest ag out_rows={ag_out} (seq={ag['seq']})"
                    )
    if xtag > args.max_report:
        print(f"  ... {xtag - args.max_report} more")

    # ---- per-(tag, DP) CP-pair positional alignment ----------------------
    print("\n== CP pair divergence (same DP, ATTN_CP0 vs ATTN_CP1) ==")
    streams = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in cp:
        if r["dp"] >= 0 and (
            r["tag"] in CP_TAGS or r["tag"].startswith(OVERLAP_PREFIX)
        ):
            streams[(r["tag"], r["dp"])][r["cp"]].append(r)

    divergences = []  # {dp, order, kind, detail}
    pairs_seen = 0
    for (tag, dp), bycp in sorted(streams.items()):
        if len(bycp) < 2:
            continue
        pairs_seen += 1
        a_list, b_list = bycp[0], bycp[1]
        if len(a_list) != len(b_list):
            print(
                f"  [{tag}] DP{dp}: COUNT MISMATCH CP0={len(a_list)} vs "
                f"CP1={len(b_list)} lines (one rank ran different/extra collectives)"
            )
            divergences.append(
                {"dp": dp, "order": min(a_list[0]["order"], b_list[0]["order"]),
                 "kind": "count", "detail": f"[{tag}] DP{dp}: {len(a_list)} vs {len(b_list)}"}
            )
            fired += 1
        for i in range(min(len(a_list), len(b_list))):
            a, b = a_list[i], b_list[i]
            if a["ctx"] != b["ctx"]:
                detail = (
                    f"[{tag}] DP{dp} @pos {i}: CTX DIVERGED "
                    f"CP0 seq={a['seq']} ctx={a['ctx']} ({a['ts']}) vs "
                    f"CP1 seq={b['seq']} ctx={b['ctx']} ({b['ts']})"
                )
                kind = "ctx"
            else:
                keys = sorted(set(a["fields"]) | set(b["fields"]))
                dd = [
                    f"{k}: {a['fields'].get(k)} vs {b['fields'].get(k)}"
                    for k in keys
                    if a["fields"].get(k) != b["fields"].get(k)
                ]
                if not dd:
                    continue
                detail = (
                    f"[{tag}] DP{dp} @pos {i}: ctx={a['ctx']} FIELDS DIVERGED "
                    + "; ".join(dd)
                    + f" (CP0 seq={a['seq']} / CP1 seq={b['seq']}, {a['ts']})"
                )
                kind = "fields"
            divergences.append({"dp": dp, "order": a["order"], "kind": kind, "detail": detail})
            fired += 1
    shown = 0
    for d in divergences:
        if shown >= args.max_report:
            break
        if d["kind"] != "count":
            print(f"  {d['detail']}")
            shown += 1
    extra = sum(1 for d in divergences if d["kind"] != "count") - shown
    if not divergences:
        print(f"  (none -- {pairs_seen} pair-streams compared, all positions agree)")
    elif extra > 0:
        print(f"  ... {extra} more")

    # ---- [mf-trap]: split layer-window catches vs gap-probe catches -------
    # A trap hit identifies the window ((prev, after] checkpoints) the
    # writer acted in and carries the rank's [cp-size] seq at catch time:
    # same-rank lines with seq <= catch are the collectives that ran before
    # the catch -- their tail is the suspect set. The divergence correlation
    # upgrades the catch into a family verdict (prior pair divergence on this
    # DP == the locally-sized-collective family caught in the act; a clean
    # pair exonerates it), and the [mf-raw] cross-check ties the caught
    # writer to the send-path victim (same rank; trap row == mf-raw slot).
    # gap:* labels name the §24.23 tail windows (post-target: CP all-gather/
    # norm/lm_head/logits; post-draft: draft prefill forward) and get their
    # own verdicts below -- they are NOT layer-window catches. PL:* labels
    # are the fine-grained post-loop bisect marks (§24.24): pre-ag/A-ag/
    # B-hcnorm/C-logits bracket the all-gather, hc_head+norm, and
    # logits_processor sub-windows inside the convicted post-target window.
    # GL:* labels (§24.26) bisect the residual glue window
    # (PL:C-logits, gap:post-target]: A0/A1 bracket the model dispatch tail
    # and the DP MLP sync; C0/C1 bracket sample/logprob and the early-return
    # delayed-sample path.
    def _kind(lbl):
        s = str(lbl)
        if s.startswith("gap:"):
            return "gap"
        if s.startswith("glue:"):
            return "glue"
        if s.startswith("PL:"):
            return "pl"
        if s.startswith("GL:"):
            return "gl"
        return "layer"

    # ---- re-observation artifact detector (Run 19, 2026-08-25) --------------
    # A trap catch only TIMES the write if the dirty row was not already seen
    # dirty earlier. When the victim's own [mf-raw] (send) fired BEFORE the
    # catch AND the catch's seq exceeds the victim rank's last collective
    # before that send, the catching snapshot belongs to a LATER forward: the
    # dirt pre-existed, the catching iteration's earlier marks were
    # window-blind (their batch max(ext+prefix) < col; only the full-width
    # D0/gap pool snapshots see it), and any "writer convicted in <window>"
    # reading is VOID.
    def _reobs(h):
        for m in mfraw:
            if (
                m["dp"] == h["dp"]
                and m["cp"] == h["cp"]
                and int(m.get("slot", -1)) == h["row"]
                and m["order"] < h["order"]
            ):
                seqs = [
                    r["seq"]
                    for r in cp
                    if r["dp"] == h["dp"]
                    and r["cp"] == h["cp"]
                    and r["order"] < m["order"]
                ]
                vs = max(seqs) if seqs else None
                if vs is not None and int(h["seq"]) > vs + 1:
                    return m, int(h["seq"]) - vs, vs
                return m, 0, vs
        return None, 0, None

    def _reobs_lines(h):
        m, delta, vs = _reobs(h)
        if not delta:
            return []
        return [
            "      -> RE-OBSERVATION ARTIFACT: the dirt was already observed "
            f"at the victim's send ([mf-raw] @ {m['ts']}); this catch's "
            f"snapshot belongs to a forward {delta} collectives LATER (the "
            f"victim's own forward ended ~seq {vs}) -- the catch TIMES "
            "NOTHING and the window reading is VOID. The catching "
            "iteration's earlier marks were window-blind (batch "
            f"max(ext+prefix) < col {h['col']}; only full-width D0/gap "
            "snapshots see it).",
            "      true write window = (the victim's own final-chunk "
            "iteration's post-draft snapshot [clean], the victim's send-loop "
            "read [dirty]) = the scheduler glue: eagle-worker return -> "
            "process_batch_result (maybe_cache_unfinished_req: radix insert "
            "+ FreeDeviceKV free device ops) -> send entry.",
        ]

    gap_hits = [h for h in mftrap if _kind(h["label"]) == "gap"]
    glue_hits = [h for h in mftrap if _kind(h["label"]) == "glue"]
    pl_hits = [h for h in mftrap if _kind(h["label"]) == "pl"]
    gl_hits = [h for h in mftrap if _kind(h["label"]) == "gl"]
    layer_hits = [h for h in mftrap if _kind(h["label"]) == "layer"]

    def _addr_lines(h):
        """§24.40 Run 24: print the addr line + alias-vs-OOB verdict.

        NOTE: the dirty address is computed from r2t's own layout, so it is
        INSIDE r2t by construction -- an in-range check is meaningless. The
        discriminators are: (a) any REGISTERED CANDIDATE whose range contains
        the dirty address (= the candidate shares storage with r2t ->
        aliasing); (b) the r2t-relative byte offset (fixed offsets across
        catches = a fixed layout relationship); (c) the fp32
        reinterpretation (payload family)."""
        body = h.get("addr")
        if not body:
            return []
        lines = [f"      addr: {body}"]
        hit_names = []
        # §24.41 Run 25: extended candidate set -- DSV4 NPU sub-pools
        # (swa/c4/c128 KV, indexer index_k/index_scale, c4 attention/indexer
        # compress-state rings) joined the static map alongside the Run-24
        # names. Any HIT convicts that pool's scatter/OOB family.
        for name in (
            "topk_p",
            "hidden",
            "kv_k0",
            "kv_v0",
            "swa_kv_pool",
            "c4_kv_pool",
            "c128_kv_pool",
            "idx_k0",
            "idx_s0",
            "c4st0",
            "c4sti0",
            "c4stsc0",
            "c4stisc0",
            "c128st0",
            "c128sti0",
            "c128stsc0",
            "c128stisc0",
        ):
            if re.search(name + r"=\[0x[0-9a-f]+,0x[0-9a-f]+\)HIT", body):
                hit_names.append(name)
        if hit_names:
            for name in hit_names:
                lines.append(
                    f"      -> candidate '{name}' range CONTAINS the dirty "
                    "address -> r2t and this tensor share storage (ALIASING "
                    "confirmed); audit the tensor-view/pointer path "
                    f"producing '{name}'."
                )
        else:
            lines.append(
                "      -> no registered candidate overlaps the dirty "
                "address: the writer is either a kernel that got r2t's "
                "pointer DIRECTLY as its dst (parameter/aliasing bug in an "
                "unregistered call site) or an OOB scatter from an "
                "UNREGISTERED neighbor tensor. Compare off_in_r2t across "
                "catches (constant offset = fixed layout relationship) and "
                "the fp32 value (payload family)."
            )
        return lines

    def _gsites_lines(h):
        """§24.41 Run 25: nearest gather-site buffers around the dirty addr.

        dist==0 -> dirty INSIDE the buffer (shared storage / aliasing);
        small |dist| -> dirty just past the end (or before the start) of a
        locally-sized gather output -> the sized-from-local OOB family; the
        tag names the exact call site. age tells whether the block was still
        live at catch time (allocator-recycled stale entries have large age).
        """
        gs = h.get("gsites")
        if gs is None:
            return []
        if not gs:
            return [
                "      gsites: registry EMPTY at catch time -- no gather site "
                "was recorded this process (sites not reached / env off); "
                "proximity verdict VOID for this catch."
            ]
        lines = []
        for g in gs:
            lines.append(
                f"      gsites: {g['tag']} {g['role']}=[{g['lo']},{g['hi']}) "
                f"dist={g['dist']} shape={g['shape']} dtype={g['dtype']} "
                f"age={g['age']}s"
            )
        inside = [g for g in gs if int(g["dist"]) == 0]
        near = [g for g in gs if 0 < abs(int(g["dist"])) <= (8 << 20)]
        if inside:
            g0 = inside[0]
            lines.append(
                f"      -> dist=0: dirty address INSIDE gather site "
                f"'{g0['tag']}' {g0['role']} buffer -> r2t and the gather "
                "buffer SHARE STORAGE (allocator / symmetric-memory aliasing); "
                "audit that site's allocation path."
            )
        elif near:
            g0 = near[0]
            d0 = int(g0["dist"])
            side = "past the END" if d0 > 0 else "BEFORE the start"
            lines.append(
                f"      -> dist={d0:+d} ({side} of '{g0['tag']}' "
                f"{g0['role']}, age={g0['age']}s): SIZED-FROM-LOCAL OVERFLOW "
                "-- the collective wrote more rows than the local-sized "
                "output holds (cp_utils.py / interleave.py discriminators); "
                "this site is the writer. Constant dist across catches = "
                "fixed layout relationship."
            )
        return lines

    if layer_hits:
        print("\n== [mf-trap] writer catches (layer windows) ==")
        fired += len(layer_hits)
        div_by_dp_t = collections.defaultdict(list)
        for d in divergences:
            div_by_dp_t[d["dp"]].append(d)
        trap_convicted = trap_clean = 0
        for h in layer_hits[: args.max_report]:
            print(
                f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: "
                f"after={h['label']} row={h['row']} col={h['col']} "
                f"val={h['val']} seq={h['seq']}"
            )
            for _ln in _reobs_lines(h):
                print(_ln)
            for _ln in _addr_lines(h):
                print(_ln)
            for _ln in _gsites_lines(h):
                print(_ln)
            mine = [
                r
                for r in cp
                if r["dp"] == h["dp"]
                and r["cp"] == h["cp"]
                and 0 <= r["seq"] <= h["seq"]
            ]
            for r in mine[-3:]:
                print(
                    f"      suspect seq={r['seq']} [{r['tag']}] {r['fields']} ctx={r['ctx']}"
                )
            prior = [
                d
                for d in div_by_dp_t.get(h["dp"], [])
                if d["order"] < h["order"]
            ]
            if prior:
                trap_convicted += 1
                print(
                    f"      family: DIVERGED-BEFORE ({len(prior)} prior) -> "
                    "locally-sized-collective family caught in the act"
                )
                for d in prior[-2:]:
                    print(f"          nearest: {d['detail']}")
            else:
                trap_clean += 1
                print(
                    "      family: pair-clean -> A family exonerated at this "
                    "catch; the writer is inside the window's kernels, not a "
                    "cross-rank size mismatch (go §24.7 B/C)"
                )
            # Victim cross-check: a later mf-raw hit on this rank for the
            # same req slot is the same clobber surviving to send time.
            same = [
                m
                for m in mfraw
                if m["dp"] == h["dp"]
                and m["cp"] == h["cp"]
                and m["order"] > h["order"]
                and int(m.get("slot", -1)) == h["row"]
            ]
            if same:
                print(
                    f"      victim match: mf-raw slot={h['row']} hit later "
                    f"({same[0]['ts']}) -- caught writer == send-path victim"
                )
        if len(layer_hits) > args.max_report:
            print("  (more suppressed; first-hit discipline keeps this at ~1/rank/forward)")
        if trap_convicted and not trap_clean:
            print(
                "  VERDICT: every catch preceded by pair divergence -> fix "
                "the diverging collective per the suspect list"
            )
        elif trap_clean and not trap_convicted:
            print(
                "  VERDICT: catches on pair-clean DPs -> A family exonerated; "
                "bisect INSIDE the named layer windows (finer trap marks)"
            )
        if not mfraw:
            print(
                "  note: no [mf-raw] lines -- trap evidence stands alone "
                "(send-path probe absent or silent this run)"
            )
    elif (
        census.get("mf-trap:armed")
        and not layer_hits
        and not gap_hits
        and not pl_hits
        and not gl_hits
        and mfraw
    ):
        print("\n== [mf-trap] writer catches (layer windows) ==")
        print(
            "  armed but ZERO catches (layer, gap, PL, GL probes) while "
            "[mf-raw] hit -- the writer acted outside every trapped window. "
            "With the send split above showing pre>0, the remaining windows "
            "are: decode/mixed forwards + the between-forward gap (overlap "
            "loop / scheduler / other batches' sends). Next: extend the "
            "trap gate beyond extend-only."
        )
        # §24.33 discriminator 6 verdict (b2b marks deployed): silence of
        # D0b/D0c is itself discriminating -- grep the deployment first.
        print(
            "  NOTE (§24.33 discriminator 6): if the D0 back-to-back marks "
            "(GL:D0b-b2b/GL:D0c-b2b, eagle_worker_v2.py) are deployed, this "
            "total silence while a row still turns dirty means the write "
            "landed AFTER the last snapshot kernel => SAME-STREAM post-"
            "snapshot task family (draft prefill forward kernels / later "
            "overlap-loop windows), device-time-axis interleaving model. "
            "Go trap the draft prefill forward + decode/mixed gates."
        )

    # ---- gap-probe catches: the §24.23 same-request tail windows -----------
    if gap_hits:
        print("\n== [mf-trap] gap-probe catches (post-layer tail windows) ==")
        fired += len(gap_hits)
        div_by_dp_g = collections.defaultdict(list)
        for d in divergences:
            div_by_dp_g[d["dp"]].append(d)
        for h in gap_hits[: args.max_report]:
            print(
                f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: "
                f"after={h['label']} row={h['row']} col={h['col']} "
                f"val={h['val']} seq={h['seq']} window={h['w0']}x{h['w1']}"
            )
            for _ln in _addr_lines(h):
                print(_ln)
            for _ln in _gsites_lines(h):
                print(_ln)
            re_lines = _reobs_lines(h)
            if re_lines:
                for _ln in re_lines:
                    print(_ln)
            elif h["label"] == "gap:pre-target":
                print(
                    "      -> WRITER CONVICTED: scheduler launch glue between "
                    "the previous iteration's batch-result marks and this "
                    "forward's start (host + B-stream prologue; C-stream had "
                    "not launched the target yet) -- NOT a forward kernel."
                )
            elif h["label"] == "gap:pre-draft":
                print(
                    "      -> WRITER CONVICTED: TARGET side of the next batch "
                    "(target forward layers + post-loop: CP all-gather / "
                    "norm / lm_head / logits capture). Draft is excluded."
                )
            elif h["label"] == "gap:post-draft-fwd":
                print(
                    "      -> WRITER CONVICTED: DRAFT prefill forward itself "
                    "(draft_runner.forward: embed + draft layers + draft "
                    "head/logits) -- bracketed by pre-draft (clean) and this "
                    "mark."
                )
            elif h["label"] == "gap:post-topk":
                print(
                    "      -> WRITER CONVICTED: draft post-forward glue "
                    "(nan/inf detect + renorm_draft_probs + fast_topk/"
                    "fast_sample kernels) -- between post-draft-fwd and this "
                    "mark."
                )
            elif h["label"] == "gap:post-target":
                gl_earlier = [
                    g for g in gl_hits if g["order"] < h["order"]
                ]
                if gl_earlier:
                    print(
                        "      -> superseded: a GL:* mark (above) already "
                        "convicted a narrower sub-window inside this one."
                    )
                else:
                    print(
                        "      -> WRITER CONVICTED: target post-loop path "
                        "(after the last layer: CP all-gather / final norm / "
                        "lm_head / logits capture). NOT a layer window, NOT "
                        "translate. With PL and GL(A/C) marks silent, the "
                        "write landed in (PL:C-logits, post-target] with "
                        "every forward-thread checkpoint covered => OFF-"
                        "STREAM WRITER: it slides through the mark gaps on "
                        "the DEVICE timeline (other batches' sends / "
                        "scheduler copy stream). Go stream-level."
                    )
            elif h["label"] == "gap:post-draft":
                print(
                    "      -> WRITER CONVICTED: draft prefill forward family "
                    "(_draft_extend_for_prefill) -- separate ForwardMode, "
                    "invisible to the extend-only layer trap. NOTE (§24.38): "
                    "the catch after=glue:entry drains THIS snapshot one "
                    "iteration later; with the new pre-draft probe deployed, "
                    "an after=glue:entry catch now means the dirt was NOT in "
                    "the draft window (pre-draft copy saw it clean only if no "
                    "gap:pre-draft catch fired) -- re-derive from the full "
                    "chain."
                )
            else:
                print(
                    f"      -> unnamed gap label '{h['label']}': check "
                    "layer_trap.py probe sites"
                )
            # Same correlation machinery as layer catches: prior pair
            # divergence (family) + mf-raw victim (survival to send).
            # Direction note: gap/PL/GL catches are DRAINED one iteration
            # later (next layer_trap_start / post-draft probe), i.e. AFTER
            # the victim's own send loop already fired [mf-raw] -- so the
            # victim mf-raw line typically PRECEDES the catch line. Match
            # both directions, not just "later".
            prior = [
                d
                for d in div_by_dp_g.get(h["dp"], [])
                if d["order"] < h["order"]
            ]
            if prior:
                print(
                    f"      family: DIVERGED-BEFORE ({len(prior)} prior) -> "
                    "see nearest divergence for the collective family"
                )
                for d in prior[-2:]:
                    print(f"          nearest: {d['detail']}")
            same = [
                m
                for m in mfraw
                if m["dp"] == h["dp"]
                and m["cp"] == h["cp"]
                and int(m.get("slot", -1)) == h["row"]
            ]
            if same:
                rel = "later" if same[0]["order"] > h["order"] else "earlier"
                print(
                    f"      victim match: mf-raw slot={h['row']} hit {rel} "
                    f"({same[0]['ts']}) -- caught writer == send-path victim"
                )
        if len(gap_hits) > args.max_report:
            print("  (more suppressed)")

    # ---- PL:* post-loop bisect catches: sub-window inside post-target ------
    if pl_hits:
        print("\n== [mf-trap] post-loop bisect catches (PL:* sub-windows) ==")
        fired += len(pl_hits)
        PL_WINDOWS = {
            "PL:pre-ag": (
                "CP all-gather rerange family (cp_all_gather_rerange_output, "
                "incl. DSpark aux gathers) -- the sub-window between the last "
                "layer and mark A"
            ),
            "PL:A-ag": (
                "hc_head + final norm sub-window (between marks A and B)"
            ),
            "PL:B-hcnorm": (
                "lm_head + logits_processor sub-window (between marks B and C) "
                "-- hidden capture / logits256 live here (retain-ring names "
                "cap.full_hidden / cap.last_hidden / target.logits256)"
            ),
            "PL:C-logits": (
                "target-return tail after logits_processor (between mark C and "
                "the eagle-worker post-target snapshot) -- capture_hidden "
                "output path"
            ),
        }
        div_by_dp_p = collections.defaultdict(list)
        for d in divergences:
            div_by_dp_p[d["dp"]].append(d)
        for h in pl_hits[: args.max_report]:
            print(
                f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: "
                f"after={h['label']} row={h['row']} col={h['col']} "
                f"val={h['val']} seq={h['seq']} window={h['w0']}x{h['w1']}"
            )
            re_lines = _reobs_lines(h)
            if re_lines:
                for _ln in re_lines:
                    print(_ln)
            else:
                desc = PL_WINDOWS.get(h["label"])
                if desc:
                    print(f"      -> WRITER CONVICTED in: {desc}")
                else:
                    print(
                        f"      -> unknown PL label '{h['label']}': check "
                        "deepseek_v4.py PL mark sites"
                    )
            prior = [
                d
                for d in div_by_dp_p.get(h["dp"], [])
                if d["order"] < h["order"]
            ]
            if prior:
                print(
                    f"      family: DIVERGED-BEFORE ({len(prior)} prior) -> "
                    "see nearest divergence for the collective family"
                )
                for d in prior[-2:]:
                    print(f"          nearest: {d['detail']}")
            same = [
                m
                for m in mfraw
                if m["dp"] == h["dp"]
                and m["cp"] == h["cp"]
                and int(m.get("slot", -1)) == h["row"]
            ]
            if same:
                rel = "later" if same[0]["order"] > h["order"] else "earlier"
                print(
                    f"      victim match: mf-raw slot={h['row']} hit {rel} "
                    f"({same[0]['ts']}) -- caught writer == send-path victim"
                )
        if len(pl_hits) > args.max_report:
            print("  (more suppressed)")

    # ---- GL:* glue bisect catches: sub-window inside (C-logits, post-tgt] --
    if gl_hits:
        print(
            "\n== [mf-trap] glue bisect catches (GL:* sub-windows of "
            "(PL:C-logits, gap:post-target]) =="
        )
        fired += len(gl_hits)
        GL_WINDOWS = {
            "GL:A0-fwd": (
                "model dispatch tail (eager/graph/split runner return -> "
                "mark A0): logits_processor return through the runner/worker "
                "result plumbing before the DP MLP sync"
            ),
            "GL:A1-mlpsync": (
                "DP MLP sync collective (post_forward_mlp_sync_batch, "
                "between marks A0 and A1) -- the forward stream stalls here; "
                "A0 clean + A1 dirty means the write landed during the sync"
            ),
            "GL:C0-sample": (
                "sample / logprob post-processing on the forward thread "
                "(between mark A1/C1 and C0)"
            ),
            "GL:C1-delayed": (
                "delayed-sample early-return path (tp_worker overlap branch); "
                "last on-stream point before the eagle_worker post-target "
                "snapshot"
            ),
            "GL:D0-eagle": (
                "eagle_worker return path (§24.27 plan A dense drain): "
                "target worker returned, plain Python between GL:C0/C1 and "
                "the gap snapshot. A hit here with GL silent convicts the "
                "return path or an off-stream write landing in this interval; "
                "silent + gap hit => OFF-STREAM WRITER CONVICTED (the write "
                "slides through every forward-thread checkpoint gap -- go "
                "stream-level: other batches' sends / scheduler copy stream)"
            ),
            "GL:D0b-b2b": (
                "back-to-back snapshot 1 (§24.33 discriminator 6): the write "
                "is visible between two CONSECUTIVELY enqueued snapshot "
                "kernels (copy_D0 then gather_D0b, ~zero host gap). A "
                "same-stream task cannot interleave there by stream FIFO => "
                "OFF-STREAM WRITER CONVICTED with a device-time anchor; "
                "correlate ts with the victim's mf-raw send moment"
            ),
            "GL:D0c-b2b": (
                "back-to-back snapshot 2 (§24.33 discriminator 6): same "
                "stream-FIFO argument as D0b -- the write landed in "
                "(copy_D0b, gather_D0c] => OFF-STREAM WRITER CONVICTED; "
                "the anchor brackets the write to a us-scale device window"
            ),
        }
        div_by_dp_g = collections.defaultdict(list)
        for d in divergences:
            div_by_dp_g[d["dp"]].append(d)
        for h in gl_hits[: args.max_report]:
            print(
                f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: "
                f"after={h['label']} row={h['row']} col={h['col']} "
                f"val={h['val']} seq={h['seq']} window={h['w0']}x{h['w1']}"
            )
            re_lines = _reobs_lines(h)
            if re_lines:
                for _ln in re_lines:
                    print(_ln)
            else:
                desc = GL_WINDOWS.get(h["label"])
                if desc:
                    print(f"      -> WRITER CONVICTED in: {desc}")
                else:
                    print(
                        f"      -> unknown GL label '{h['label']}': check "
                        "model_runner.py / tp_worker.py GL mark sites"
                    )
            prior = [
                d
                for d in div_by_dp_g.get(h["dp"], [])
                if d["order"] < h["order"]
            ]
            if prior:
                print(
                    f"      family: DIVERGED-BEFORE ({len(prior)} prior) -> "
                    "see nearest divergence for the collective family"
                )
                for d in prior[-2:]:
                    print(f"          nearest: {d['detail']}")
            same = [
                m
                for m in mfraw
                if m["dp"] == h["dp"]
                and m["cp"] == h["cp"]
                and int(m.get("slot", -1)) == h["row"]
            ]
            if same:
                rel = "later" if same[0]["order"] > h["order"] else "earlier"
                print(
                    f"      victim match: mf-raw slot={h['row']} hit {rel} "
                    f"({same[0]['ts']}) -- caught writer == send-path victim"
                )
        if len(gl_hits) > args.max_report:
            print("  (more suppressed)")

    # ---- glue:* catches: §24.34 scheduler-glue window bisect ----------------
    # Row-scoped marks in process_batch_result_disagg_prefill's final-chunk
    # branch (layer_trap.py glue_probe). A catch on label X means the write
    # landed in (previous snapshot, X's snapshot] on the device timeline:
    if glue_hits:
        print(
            "\n== [mf-trap] glue bisect catches (scheduler glue window) =="
        )
        fired += len(glue_hits)
        freeze_on = census.get("mf-trap:freeze", 0) > 0
        if freeze_on:
            print(
                "  (§24.35 fwd-freeze discriminator ACTIVE: the forward stream "
                "is synchronized before every glue:pre-cache snapshot. A "
                "catch after=glue:post-cache now convicts the DEFAULT-STREAM "
                "window (maybe_cache_unfinished_req: radix insert + write-back "
                "+ free ops) with stream C EXCLUDED -- clean conviction. A "
                "catch after=glue:pre-cache means the dirt landed in (glue:"
                "entry snapshot, post-freeze pre-cache snapshot]: either the "
                "scheduler-stream prologue (finalize scatters / "
                "move_logprobs D2H) OR a stream-C forward kernel that wrote "
                "after the entry copy -- the freeze guarantees C completion, "
                "so C-family dirt IS captured; cross-check the col=0 "
                "glue:entry signature to weigh the C reading.)"
            )
        GLUE_WINDOWS = {
            "glue:entry": (
                "eagle-worker return tail / event-loop glue between the "
                "forward return and this batch-result entry (pure host; the "
                "result-copy stream D2H may still be in flight) -- off-stream "
                "suspects live here"
            ),
            "glue:pre-cache": (
                "batch-result PROLOGUE: copy_done.synchronize + "
                "TopkCaptureOutput.finalize() host scatters "
                "(routed_experts/indexer) + move_logprobs_to_cpu D2H"
            ),
            "glue:post-cache": (
                "RADIX CACHE/FREE PATH CONVICTED: maybe_cache_unfinished_req "
                "(radix insert + FreeDeviceKV free device ops -- unique/cat "
                "kernels; the same path as the §22 swa.py:330 crash family). "
                "Next: confront the free/insert device ops and the tree-held "
                "index values they consume"
            ),
            "glue:post-extract": (
                "eagle output extraction: draft_input.topk_p[i]/topk_index[i] "
                "views + hidden_states[i].cpu().clone() D2H (+ "
                "dsa_topk_indices clone) -- the D2H clone family"
            ),
            "glue:pre-send": (
                "logprob / sampling-mask processing glue (host + result-copy "
                "stream reads)"
            ),
            "glue:mid-loop": (
                "per-req loop tail: this req's branch just completed (send "
                "launched / logprob glue / grammar accept) -- window from "
                "the previous mark inside this iteration"
            ),
            "glue:post-loop": (
                "batch-result tail after the req loop (metrics report / "
                "remaining host glue before returning to the event loop)"
            ),
            "glue:pre-inflight": (
                "event-loop window: process_disagg_prefill_inflight_queue "
                "(send polling / transfer engine) + "
                "launch_batch_sample_if_needed -- between iterations, "
                "stream C may be running the next batch"
            ),
            "glue:chunk-pre-cache": (
                "process_prefill_chunk batch-prep tail (previous pending "
                "snapshot, this mark] -- scheduler glue before the chunked "
                "radix cache"
            ),
            "glue:chunk-post-cache": (
                "RADIX CACHE/FREE PATH (chunked variant): "
                "maybe_cache_unfinished_req(chunked=True) in "
                "process_prefill_chunk -- same insert + FreeDeviceKV family "
                "as the §22 crash; middle-chunk victims land here"
            ),
            "glue:mid-pre-send": (
                "middle-chunk tail: input-logprob processing + loop glue "
                "before the chunk send"
            ),
        }
        # Cross-rank signature (§24.36): identical (ts,row,col,val) across
        # DPs = a DETERMINISTIC kernel writing the same computed value on
        # every rank (same input -> same output) through a corrupted index
        # tensor -- NOT recycled-memory garbage (that would be random).
        sig = collections.defaultdict(list)
        for h in glue_hits:
            sig[(h["ts"], h["row"], h["col"], h["val"])].append(h)
        for h in glue_hits[: args.max_report]:
            print(
                f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: "
                f"after={h['label']} row={h['row']} col={h['col']} "
                f"val={h['val']} seq={h['seq']} window={h['w0']}x{h['w1']}"
            )
            for _ln in _addr_lines(h):
                print(_ln)
            for _ln in _gsites_lines(h):
                print(_ln)
            peers = [
                p for p in sig[(h["ts"], h["row"], h["col"], h["val"])]
                if p is not h
            ]
            if peers:
                peer_list = ", ".join(
                    "DP%d CP%d" % (p["dp"], p["cp"]) for p in peers
                )
                print(
                    f"      cross-rank: {len(peers)} other rank(s) "
                    f"({peer_list}) "
                    "share the EXACT (row,col,val) this second -> "
                    "DETERMINISTIC kernel + corrupted-index scatter, not "
                    "freed-memory garbage"
                )
            if abs(h["val"]) < 1 << 20:
                print(
                    "      small-int payload: index-shaped corruption "
                    "(not fp32-bit-cast) -- a broken index value itself, "
                    "distinct from the fp32-garbage signature"
                )
            re_lines = _reobs_lines(h)
            if re_lines:
                for _ln in re_lines:
                    print(_ln)
            else:
                desc = GLUE_WINDOWS.get(h["label"])
                if freeze_on and h["label"] == "glue:pre-cache":
                    desc = (
                        "PRE-CACHE CATCH WITH FREEZE (§24.35): dirt landed in "
                        "(glue:entry snapshot, post-freeze pre-cache "
                        "snapshot]. Stream C was fully drained, so the writer "
                        "is EITHER the scheduler-stream prologue (finalize "
                        "scatters / move_logprobs D2H) OR a stream-C forward "
                        "kernel that wrote after the entry copy (the col=0 "
                        "glue:entry family). col==0 across many ranks favors "
                        "the C-forward reading; next discriminator: freeze "
                        "before glue:entry."
                    )
                elif freeze_on and h["label"] == "glue:post-cache":
                    desc = (
                        "DEFAULT-STREAM WRITE-BACK CONVICTED (§24.35): stream "
                        "C was frozen idle across the whole (pre-cache, "
                        "post-cache] window, so the write came from the "
                        "scheduler stream -- the radix write-back "
                        "req_to_token_pool.write planting tree-held "
                        "device_indices (unified_radix_cache.py "
                        "cache_unfinished_req) or the insert/free device ops. "
                        "Confront the tree-held value tensors' lifetime "
                        "(evict vs. cat views)."
                    )
                if desc:
                    print(f"      -> WRITER CONVICTED in: {desc}")
                else:
                    print(
                        f"      -> unknown glue label '{h['label']}': check "
                        "layer_trap.py glue_probe call sites"
                    )
            same = [
                m
                for m in mfraw
                if m["dp"] == h["dp"]
                and m["cp"] == h["cp"]
                and int(m.get("slot", -1)) == h["row"]
            ]
            if same:
                rel = "later" if same[0]["order"] > h["order"] else "earlier"
                print(
                    f"      victim match: mf-raw slot={h['row']} hit {rel} "
                    f"({same[0]['ts']}) -- caught writer == send-path victim"
                )
        if len(glue_hits) > args.max_report:
            print("  (more suppressed)")
        # §24.41 Run 25 cross-catch aggregation: (tag, role, dist) repeated
        # across catches/ranks = a FIXED layout relationship between r2t and
        # that site's buffer -- the single-writer signature. A spread of
        # dists/tags = allocator-relative noise (weaker).
        gs_all = [g for h in glue_hits for g in h.get("gsites") or []]
        if gs_all:
            dist_key = collections.Counter(
                (g["tag"], g["role"], g["dist"]) for g in gs_all
            )
            tag_key = collections.Counter(g["tag"] for g in gs_all)
            top_dist = dist_key.most_common(3)
            top_tag = tag_key.most_common(3)
            print(
                "  (§24.41 GSITES AGGREGATE: "
                + "; ".join(
                    f"{t}[{r}]dist={d} x{n}"
                    for (t, r, d), n in top_dist
                )
                + f" -- top tags: "
                + ", ".join(f"{t} x{n}" for t, n in top_tag)
                + ". Repeated (tag,dist) across catches/ranks = FIXED "
                "LAYOUT relationship -> that site's overflow is the writer; "
                "all dists large/varied -> no live neighbor, weigh the "
                "C-stream kernel-OOB reading.)"
            )
        elif any("gsites" in h for h in glue_hits):
            print(
                "  (§24.41: gsites registry EMPTY at every catch -- gather "
                "sites were not recorded; check SGLANG_MF_LAYER_TRAP and the "
                "new note_gather call sites are deployed)"
            )
        print(
            "  (all three marks silent + send pre>0 => the write landed in "
            "(glue:pre-send snapshot, send pre-reduction] -- send-entry "
            "internals or a us-scale off-stream slide; correlate with the "
            "send-split lines)"
        )
        if freeze_on:
            has_post = any(h["label"] == "glue:post-cache" for h in glue_hits)
            has_pre = any(h["label"] == "glue:pre-cache" for h in glue_hits)
            gather_freeze_on = census.get("mf-trap:gather-freeze", 0) > 0
            if gather_freeze_on:
                print(
                    "  (§24.39 GATHER-FREEZE VERDICT: catches PERSIST with "
                    "gather streams drained -> transfer family excluded; "
                    "sole remaining suspect = stream-C CP collective "
                    "kernels (NPU-side OOB write, invisible to python "
                    "audits). Go kernel-level: CANN op dump / tiling audit "
                    "of the interleave all-gather family.)"
                )
            if not has_post and not has_pre and glue_hits:
                fp32_shaped = [
                    h for h in glue_hits if abs(h["val"]) >= 1 << 30
                ]
                print(
                    "  (§24.36 FREEZE VERDICT: only glue:entry catches while "
                    "frozen, and no [mf-raw]. Run 20 caught the same writer "
                    "after glue:post-cache WITH stream-C concurrency; "
                    "serializing C removed the post-cache window write AND "
                    "send-time survival -> the write REQUIRES stream-C "
                    "concurrency (cross-stream race or C-kernel off-stream "
                    "write). Payload audit: the radix write-back chain "
                    "(values copy -> node.value clone -> torch.cat -> "
                    "r2t write) carries INT-INDEX tensors only -- it can "
                    f"NEVER produce fp32-bitcast values. "
                    f"{len(fp32_shaped)} catch(es) here are fp32-bitcast -> "
                    "FOREIGN KERNEL physically writing r2t memory (OOB "
                    "scatter from an adjacent pool on stream C; "
                    "cross-rank same-val = deterministic kernel + corrupted "
                    "index). Only small-int catches (|val| < 2^20) can be "
                    "the write-back family. Next: dump the memory layout "
                    "(r2t vs kv-pool vs hidden-state tensor addresses) on "
                    "a victim rank and audit the C-stream scatter kernels' "
                    "index tensors."
                )
            elif has_post:
                print(
                    "  (§24.35: post-cache catch survived the freeze -> "
                    "default-stream writer confirmed; see branch text above)"
                )

    # ---- §24.39 Run 23: gather-freeze zero-catch verdict -------------------
    # Must live OUTSIDE the glue_hits block: the conviction case is exactly
    # glue_hits == [] (catches vanished with gather streams drained), which
    # would never print from inside.
    gather_freeze_on = census.get("mf-trap:gather-freeze", 0) > 0
    if (
        gather_freeze_on
        and census.get("mf-trap:armed", 0) > 0
        and not glue_hits
    ):
        print(
            "\n== [mf-trap] §24.39 gather-freeze verdict =="
        )
        print(
            "  ZERO glue catches with gather streams drained at every pool "
            "snapshot (same workload without SGLANG_MF_FREEZE_GATHER, Run "
            "22: 13 catches) -> KV-TRANSFER/GATHER-STREAM FAMILY CONVICTED: "
            "the r2t writer rides the transfer worker threads' staging-"
            "gather activity. Audit the transfer path: page_idx H2D on the "
            "default stream issued from worker threads, staging ring "
            "allocator reuse, NIXL/mooncake registered-memory interactions. "
            "CAVEAT: also confirm the workload reproduced (compare run "
            "duration / cp-size counts with Run 22; a too-short run also "
            "yields zero catches)."
        )

    # ---- send-path split: pre- vs post-translate dirty of the same segment --
    if mftrap_send:
        print("\n== [mf-trap] send-path split (pre- vs post-translate) ==")
        for h in mftrap_send[: args.max_report]:
            pre, post, tot = int(h["pre"]), int(h["post"]), int(h["tot"])
            print(
                f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']} "
                f"rid={h['rid'][:8]} slot={h['slot']} seg=[{h['s0']},{h['s1']}) "
                f"pre={pre} post={post} tot={tot}"
            )
            if pre > 0:
                print(
                    "      -> PRE-TRANSLATE dirty: row was already garbage at "
                    "send-loop entry. Writer acts in decode/mixed forwards or "
                    "the between-forward gap (overlap loop / scheduler / other "
                    "batches' sends) -- NOT translate. The gap probes "
                    "(post-target/post-draft) bracket the same-request tail; "
                    "if they stayed silent, extend the trap gate to "
                    "decode/mixed/draft forwards."
                )
            elif post > 0:
                print(
                    "      -> entered AT TRANSLATE: pre-clean, post-dirty. "
                    "Confront translate_kv_indices_for_transfer + pool-side "
                    "mapping/scatter kernels (new kernel-side family)."
                )
        if len(mftrap_send) > args.max_report:
            print("  (more suppressed)")
    elif mfraw and census.get("mf-trap:armed") and not mftrap:
        print(
            "\n== [mf-trap] send-path split: NO send-pre lines while [mf-raw] "
            "hit -- the send probe never fired (old deploy / env miss); the "
            "pre/post split is VOID this run."
        )

    # ---- §24.29 S1/S2 scatter-loc OOB discriminator ------------------------
    # [mf-scatter] lines only appear when a clamped scatter slot saw OOB locs.
    # mf-raw hit + zero [mf-scatter] lines = discriminator (1) result: the
    # AUDITED scatter entries (indexer/swa/compress, and compressor-state
    # once §24.29 discriminator (2) is deployed) were clean at that D2H.
    # §24.43: the body now also carries block-table audit slots from the
    # npu_sparse_attn_sharedkv call sites -- swapt*/cmp4pt/cmp128pt/c4topk
    # (each :neg/:hi/:min/:max/:negrow/:negcol) plus the swapt:fulltable-
    # fallback path flag. Page-table verdicts print separately below.
    if mfscatter:
        print("\n== [mf-scatter] clamped OOB scatter locs (S1/S2 loc entries) ==")
        fired += len(mfscatter)

        def _split_slots(body):
            """Split a {k: v, ...} body into scatter-loc vs page-table slots."""
            sc, pt, fl = {}, {}, {}
            try:
                d = ast.literal_eval(body)
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k.startswith("swapt") or k.startswith("cmp") or k.startswith("c4topk"):
                            pt[k] = v
                        elif "fallback" in k:
                            fl[k] = v
                        else:
                            sc[k] = v
            except (ValueError, SyntaxError):
                sc["<unparsed>"] = body
            return sc, pt, fl

        any_pt = False
        for h in mfscatter[: args.max_report]:
            sc, pt, fl = _split_slots(h["body"])
            if sc:
                print(
                    f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: {sc}"
                )
            if fl:
                print(
                    f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: paths {fl}"
                )
            if pt:
                any_pt = True
                print(
                    f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']}: pagetables {pt}"
                )
        print(
            "      -> VERDICT (scatter slots): OOB locs reached a clamped "
            "scatter entry; 'compressor-state' present => S1 VALUE-entry "
            "convicted (garbage positions/req_pool_indices -> ring state_loc "
            "OOB -> fused compressor fp32 ring write lands outside the ring); "
            "'compressor-index:hi' => S1 KERNEL-INDEX convicted (seqused + "
            "coff*cmpRatio exceeds the table width -- the kernel's column "
            "read runs past the table, pulling off-table memory in as "
            "stateLoc; verify-path draft-augmented seq_lens is the prime "
            "suspect); 'compressor-index:bs' => table batch-dim != live "
            "batch size (kernel batchIdx reads past the table's batch "
            "axis); 'swa'/'compress' => S2/S3 family "
            "(npu_scatter_nd_update_/index_put OOB)."
        )
        print(
            "      -> VERDICT (§24.47 b2b + produce-time): 'swa-b2b' > 0 "
            "=> the S2 set_swa_buffer index_put (or the ops queued "
            "between the two same-stream checks) dirtied ACTIVE r2t "
            "rows -- S2 convicted (contradicts Run 30's fp32 payload "
            "family; re-examine). 'swa-b2b' == 0 across a run WITH "
            "catches => S2 terminally excluded. "
            "'c4topk:proc-junk' > 0 => the lightning-indexer kernel "
            "EMITTED garbage indices (source) -> CANN op dump / tiling "
            "audit of npu_quant_lightning_indexer. 'c4topk:proc-junk' "
            "== 0 while attention-time 'c4topk:junk' > 0 (§24.46) => "
            "the buffer was CLEAN at production and dirtied in "
            "between (victim) -> reopen the writer model, weigh the "
            "logprob/move_logprobs family (fp32 payload, "
            "glue:entry catch)."
        )
        if any_pt:
            print(
                "      -> VERDICT (page tables, §24.43): 'swapt*:neg' > 0 => "
                "the CANN attention kernel received NEGATIVE swa page ids "
                "(full_to_swa -1 sentinel / graph tail fill) and computes "
                "kv_base + page*stride BELOW the swa pool base -> the r2t "
                "last-rows corruption geometry; check swapt:min (== -1 "
                "sentinel vs other garbage) and swapt:negrow/negcol for the "
                "first offending request/page slot. 'swapt*:hi' > 0 with "
                "'swapt:fulltable-fallback' > 0 => the eager path fed "
                "FULL-pool page ids into the swa kernel (index-space "
                "mismatch, positive OOB). 'cmp4pt/cmp128pt' same reading "
                "for the compressed tables; 'c4topk:min < 0' => topk sparse "
                "indices negative. §24.45: '*:tailneg' - '*:neg' = -1 mass "
                "OUTSIDE the valid region (the graph tables' -1 tail past "
                "the seqused-derived width); a catch with large tailneg - "
                "neg on a rank whose eager tables carry no -1 fill => the "
                "kernel's read width EXCEEDS the valid width and it walked "
                "into the -1 tail -> kv_base - stride below-pool writes. "
                "All-zero page-table slots => the "
                "kernel's INPUT tables were clean: remaining suspect is a "
                "kernel-INTERNAL addressing bug (go CANN op dump / tiling "
                "audit)."
            )
        if len(mfscatter) > args.max_report:
            print("  (more suppressed)")
    elif mfraw:
        print(
            "\n== [mf-scatter] S1/S2 discriminator: ZERO [mf-scatter] lines "
            "while [mf-raw] hit -- all AUDITED scatter entries stayed clean "
            "at the piggyback D2H. With §24.30 discriminators (2)+(3) "
            "deployed this covers BOTH the compressor table VALUES "
            "(compressor-state) AND the kernel-side read indices "
            "(compressor-index:hi/:bs): S1's Python-visible surfaces are "
            "exonerated -- remaining S1 face is a kernel-INTERNAL addressing "
            "bug (wrong stateLoc arithmetic inside WriteToCacheState); go "
            "kernel-level (CANN op dump / tiling audit) or re-rank S4/S5."
        )

    # ---- [mf-raw] correlation --------------------------------------------
    print("\n== [mf-raw] dirty-row hits vs earlier CP-pair divergence (same DP) ==")
    if not mfraw:
        print("  (no [mf-raw] hits in this log -- run must reproduce for a verdict)")
    else:
        div_by_dp = collections.defaultdict(list)
        for d in divergences:
            div_by_dp[d["dp"]].append(d)
        with_div = 0
        for h in mfraw:
            prior = [d for d in div_by_dp.get(h["dp"], []) if d["order"] < h["order"]]
            tag = "DIVERGED-BEFORE" if prior else "pair-clean-so-far"
            if prior:
                with_div += 1
            if len(mfraw) <= args.max_report:
                print(
                    f"  {h['ts']} DP{h['dp']} CP{h['cp']} TP{h['tp']} rid={h['rid'][:8]} "
                    f"slot={h['slot']} n_dirty={h['ndirty']}/{h['ntot']} "
                    f"n_sentinel={h['nsent']} -> {tag} ({len(prior)} prior)"
                )
                for d in prior[-2:]:
                    print(f"      nearest: {d['detail']}")
        print(
            f"\n  summary: {len(mfraw)} dirty-row hits, {with_div} on DPs whose CP "
            f"pair had diverged earlier; divergent DPs: "
            f"{sorted(div_by_dp) if divergences else 'none'}"
        )
        if not divergences:
            print("  VERDICT: zero pair divergence + dirty rows present -> A family "
                  "exonerated in this run, go §24.7 B/C.")
        elif with_div == len(mfraw) and mfraw:
            print("  VERDICT: every hit preceded by its DP's pair divergence -> "
                  "conviction; fix per §24.5.")

    # ---- cross-hit payload on GLOBAL token offsets -------------------------
    # slots[k] sits at global row position seg0 + s0 + k; pages24[j] at
    # seg0 + j*128. Two hits carved from one position-indexed stream agree
    # at the SAME global offsets (the §24.10.3 "896 alignment"); comparing
    # raw windows is meaningless when the windows differ (the retired
    # "distinct" verdict was exactly that artifact).
    def _gpos_maps(hit):
        """(slots_map, pages_map): global token position -> value, per kind.

        slots and pages sample the same rows at different granularity
        (slot id vs slot//128), so they must be compared within their own
        kind -- mixing them at one position would compare a raw slot with
        a page id and always mismatch.
        """
        gs, gp = {}, {}
        try:
            seg0 = int(hit["seg0"])
        except (KeyError, TypeError, ValueError):
            return gs, gp
        if hit.get("s0") is not None and hit.get("slots"):
            try:
                for k, v in enumerate(
                    int(x) for x in hit["slots"].replace(",", " ").split()
                ):
                    gs[seg0 + int(hit["s0"]) + k] = v
            except ValueError:
                pass
        if hit.get("pages"):
            try:
                for j, v in enumerate(
                    int(x) for x in hit["pages"].replace(",", " ").split()
                ):
                    gp[seg0 + j * 128] = v
            except ValueError:
                pass
        return gs, gp

    gmaps = [(h, *_gpos_maps(h)) for h in mfraw]
    if sum(1 for _h, gs, gp in gmaps if gs or gp) >= 2:
        print("\n== cross-hit payload on GLOBAL token offsets ==")
        for i in range(len(gmaps)):
            for j in range(i + 1, len(gmaps)):
                ha, ga_s, ga_p = gmaps[i]
                hb, gb_s, gb_p = gmaps[j]
                if not (ga_s or ga_p) or not (gb_s or gb_p):
                    continue
                common = match = run = best = 0
                for ga, gb in ((ga_s, gb_s), (ga_p, gb_p)):
                    shared = sorted(set(ga) & set(gb))
                    common += len(shared)
                    streak = 0
                    for p in shared:
                        if ga[p] == gb[p]:
                            match += 1
                            streak += 1
                            best = max(best, streak)
                        else:
                            streak = 0
                if common and match == common and common >= 4:
                    verdict = "SAME positional stream"
                elif match:
                    verdict = "partial"
                else:
                    verdict = "no aligned overlap"
                print(
                    f"  hit{i}(DP{ha['dp']}CP{ha['cp']} slot={ha['slot']}) vs "
                    f"hit{j}(DP{hb['dp']}CP{hb['cp']} slot={hb['slot']}): "
                    f"common={common} matched={match} best_run={best}"
                    f" -> {verdict}"
                )
        print("  (SAME positional stream == one position-indexed payload")
        print("   stream sampled twice, §24.10.3; supersedes the old")
        print("   window-overlap 'distinct' verdict)")

    # ---- flight recorder: the victim rank's own forwards before each hit ---
    if mfraw:
        print("\n== flight recorder: victim rank's own [cp-size] lines before each hit ==")
        for idx, h in enumerate(mfraw[: args.max_report]):
            mine = [
                r for r in cp
                if r["dp"] == h["dp"] and r["cp"] == h["cp"] and r["order"] < h["order"]
            ]
            print(
                f"  hit{idx} {h['ts']} DP{h['dp']} CP{h['cp']} rid={h['rid'][:8]} "
                f"slot={h['slot']} seg=[{h['seg0']},{h['seg1']}) n_dirty={h['ndirty']}/{h['ntot']}"
            )
            if h.get("slots"):
                print(f"      fdo={h['fdo']} slots={h['slots']}")
            if h.get("floats"):
                print(f"      floats={h['floats']}")
            if h.get("pages"):
                print(f"      pages24={h['pages']}")
            for r in mine[-5:]:
                shape = " ".join(
                    f"{k}={r['fields'][k]}"
                    for k in ("max_len", "local_len", "x_rows", "out_rows")
                    if k in r["fields"]
                )
                print(f"      {r['ts']} seq={r['seq']} [{r['tag']}] {shape} ctx={r['ctx']}")
            if not mine:
                print("      (no prior [cp-size] lines from this rank)")

    if mftrans:
        print("\n== [mf-trans] drops per rank (downstream impact) ==")
        for (dp, cp_), n in sorted(mftrans.items()):
            print(f"  DP{dp} CP{cp_}: {n}")

    if mfreg_fail:
        print("\n== [mf-reg] engine-refused regions (fabric registration bisect) ==")
        by_rank = collections.defaultdict(list)
        for f in mfreg_fail:
            by_rank[(f["dp"], f["cp"])].append(f)
        fired += len(mfreg_fail)
        for (dp, cp_), fs in sorted(by_rank.items()):
            print(f"  DP{dp} CP{cp_}: {len(fs)} refused region(s)")
            for f in fs[: args.max_report]:
                print(
                    f"    {f['ts']} ptr={f['ptr']} len={f['len']} end={f['end']}"
                )
        print(
            "  (refused => excluded from the known set; transfers touching "
            "them are dropped with [mf-trans] src-outside lines. Confront "
            "these VAs with dirty-row pool addresses when correlating.)"
        )

    if not census.get("mf-trap:armed") and not mftrap:
        print(
            "\nNOTE: no [mf-trap] armed marker and no catches -- the layer "
            "trap never ran in this log (stale deploy or env not set); any "
            "'forward was clean' reading from this file is VOID."
        )
    print(
        f"\n{sum(census.values())} relevant lines: "
        + ("FINDINGS PRESENT" if fired else "no divergence/invariant findings")
    )
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())

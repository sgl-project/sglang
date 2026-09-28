"""Parse an SGLANG_DEBUG_MTE_TRACE-instrumented prefill log after a crash.

Usage (one command, everything derived automatically):
  python3 test/manual/parse_mte_trace.py <prefill_log> \
      --plog-dir ../plog_pd_prefill_host_rdma/debug/plog

  # or point at the faulting plog file directly
  python3 test/manual/parse_mte_trace.py <prefill_log> --plog <plog_file>

What it does:
  1. Scans the plog for the faulting kernel name and the
     "args(0 to 19) after execute:..." line; the first 10 hex tokens are the
     kernel's tensor pointers (mixes, rsqrt, hc_scale, hc_base, x, y, post,
     comb, workspace, tiling).
  2. Cross-references every tensor address against the [mte.hcpre] lines:
     which logged field it is and how many calls already used it.
  3. Prints the actor timeline (alloc/hook/radix/send) before the final
     hc_pre call, plus the steady-state per-field addresses over the last
     50 calls and the final 25 tagged events.

Verdict guide for the faulting tensor's usage count:
  1        -> VA bad from allocation (boundary / out-of-bounds family)
  N >> 1   -> VA invalidated mid-run by an actor outside the allocator
              (driver-level; allocator releases would show as [mte.alloc]
              negative deltas or [mte.hook] empty_cache)
"""

import argparse
import glob
import os
import re
from collections import Counter

TAG_RE = re.compile(r"\[mte\.(\w+)\]")
FIELD_RE = re.compile(r"(x|y|post|comb)=0x([0-9a-f]+)")
TENSOR_LABELS = [
    "mixes(in0)", "rsqrt(in1)", "hc_scale(in2)", "hc_base(in3)",
    "x(in4)", "y(out5)", "post(out6)", "comb(out7)", "workspace(8)", "tiling(9)",
]


def find_fault_plog(plog_dir: str):
    """Newest plog in the dir that names the faulting kernel."""
    best = None
    for path in sorted(glob.glob(os.path.join(plog_dir, "*.log"))):
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if "fault kernel_name=HcPreSinkhorn" in text:
            mtime = os.path.getmtime(path)
            if best is None or mtime > best[0]:
                best = (mtime, path, text)
    return (best[1], best[2]) if best else (None, None)


def parse_plog_args(text: str):
    kernel = None
    m = re.findall(r"fault kernel_name=(\S+?)[, ]", text)
    if m:
        kernel = m[-1]
    addrs = []
    am = re.search(r"args\(0 to \d+\) after execute:([^\n]*)", text)
    if am:
        for tok in am.group(1).split(","):
            tok = tok.strip()
            # Real pointers only: 0x + 10+ hex digits (scalars like 0x800
            # are shorter).
            if tok.startswith("0x") and len(tok) >= 12:
                addrs.append(tok)
    return kernel, addrs[: len(TENSOR_LABELS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--plog", default=None, help="faulting plog file")
    parser.add_argument("--plog-dir", default=None,
                        help="directory of plog-*.log; newest one naming "
                        "HcPreSinkhorn is used automatically")
    args = parser.parse_args()

    with open(args.log, errors="replace") as f:
        lines = f.read().splitlines()

    events = []
    for i, line in enumerate(lines):
        m = TAG_RE.search(line)
        if m:
            events.append((i, m.group(1), line))

    counts = Counter(tag for _, tag, _ in events)
    print(f"total lines {len(lines)}, mte events {sum(counts.values())}")
    print("event counts:", dict(counts))

    hcpre_lines = [line for _, tag, line in events if tag in ("hcpre", "launch")]
    hcpre_idx = [i for i, tag, _ in events if tag in ("hcpre", "launch")]
    last_hcpre = hcpre_idx[-1] if hcpre_idx else None
    # A [mte.launch] line with no [mte.hcpre] line before the next launch is
    # the faulting call: the wrapper raised inside the op before the
    # post-call trace could print.
    launches = [i for i, tag, _ in events if tag == "launch"]
    completed = [i for i, tag, _ in events if tag == "hcpre"]
    faulting = [
        i
        for k, i in enumerate(launches)
        if not any(
            i < j < (launches[k + 1] if k + 1 < len(launches) else len(lines))
            for j in completed
        )
    ]
    if faulting:
        print("FAULTING launch lines (launched, never completed):")
        for i in faulting:
            print(f"  line {i}: {lines[i].strip()[:240]}")
    print(f"hcpre calls: {len(hcpre_idx)}, last at line {last_hcpre}")
    if hcpre_lines:
        print(f"final hcpre line: {hcpre_lines[-1].strip()[:240]}")

    tail = hcpre_lines[-50:]
    for field in ("x", "y", "post", "comb"):
        per_field = Counter(
            m.group(0) for line in tail for m in FIELD_RE.finditer(line)
            if m.group(1) == field
        )
        print(f"last-50 {field}: {dict(per_field)}")

    if args.plog_dir and not args.plog:
        args.plog, text = find_fault_plog(args.plog_dir)
    else:
        text = None
    if args.plog:
        if text is None:
            with open(args.plog, errors="replace") as f:
                text = f.read()
        kernel, addrs = parse_plog_args(text)
        print(f"\nplog: {args.plog}")
        print(f"fault kernel: {kernel}")
        print(f"tensor addrs ({len(addrs)}):")
        matched_any = False
        for label, addr in zip(TENSOR_LABELS, addrs):
            needle = f"=0x{addr[2:].lower()}"
            hits = [line for line in hcpre_lines if needle in line]
            fields = sorted(
                {
                    m.group(1)
                    for line in hits
                    for m in FIELD_RE.finditer(line)
                    if m.group(0).lower().endswith(needle[1:])
                }
            )
            note = f" -> {len(hits)} hcpre uses (fields: {fields})" if hits else ""
            if hits:
                matched_any = True
                first = hits[0].strip()[:200]
                last = hits[-1].strip()[:200]
                verdict = (
                    "FIRST-USE FAULT (bad from allocation)"
                    if len(hits) <= 1
                    else f"faulted after {len(hits) - 1} successful uses"
                )
                print(f"  {label:14s} {addr}{note}")
                print(f"      verdict: {verdict}")
                print(f"      first  : {first}")
                print(f"      last   : {last}")
            else:
                print(f"  {label:14s} {addr} (not seen in hcpre log)")
        if not matched_any:
            print("none of the plog tensor addresses appear in the hcpre log"
                  " (different run or layout?)")

    for tag in ("alloc", "hook", "radix", "send"):
        prior = [
            (i, line)
            for i, t, line in events
            if t == tag and (last_hcpre is None or i <= last_hcpre)
        ]
        print(f"\nlast 5 [{tag}] before the final hcpre (total {len(prior)}):")
        for i, line in prior[-5:]:
            print(f"  line {i}: {line.strip()[:240]}")

    print("\nfinal 25 tagged events:")
    for i, tag, line in events[-25:]:
        print(f"  line {i}: {line.strip()[:240]}")


if __name__ == "__main__":
    main()

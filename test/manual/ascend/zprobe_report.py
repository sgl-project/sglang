"""Summarize ZPROBE logs from zero_probe.py into a paste-back report.

Reads a raw server log (or pre-grepped ZPROBE lines, or stdin), dedupes the
per-rank fingerprint blocks, compresses layer/sampler timelines, applies the
incident decision table, and prints a compact report ending in VERDICT lines.

Usage:
  python3 zprobe_report.py <server.log | zprobe.log>
  grep ZPROBE server.log | python3 zprobe_report.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

PREFIX = re.compile(r"\[ZPROBE r(-?\d+)\] (.*)")
STATS = re.compile(
    r"shape=\(([^)]*)\).*?min=(\S+) max=(\S+) mean=(\S+) nan=(\d+) inf=(\d+)"
)
LAYER = re.compile(r"extend (\d+) (.+?): in\[(.*?)\] out\[(.*?)\]")
EXTEND = re.compile(
    r"extend (\d+): mode=(\S+) bs=(\S+) input_ids\[:\d+\]=\[([^\]]*)\]"
    r"(?: positions\[:\d+\]=\[([^\]]*)\])?(.*)$"
)
S_LOGITS = re.compile(r"sampler (\d+) logits: (.*)")
S_ARGMAX = re.compile(
    r"sampler (\d+) argmax: npu=\[([^\]]*)\] cpu=\[([^\]]*)\] match=(True|False)"
)


def parse_stats(s: str):
    m = STATS.search(s)
    if not m:
        return None

    def _f(v: str) -> float:
        try:
            return float(v)
        except ValueError:
            return float("nan")

    return {
        "shape": m.group(1),
        "min": _f(m.group(2)),
        "max": _f(m.group(3)),
        "mean": _f(m.group(4)),
        "nan": int(m.group(5)),
        "inf": int(m.group(6)),
    }


def classify(st):
    """OK / bad reason for one stats dict."""
    if st is None:
        return "?"
    if st["nan"] or st["inf"]:
        return "nan/inf"
    if st["max"] == 0 and st["min"] == 0:
        return "all-zero"
    if st["max"] == st["min"]:
        return "constant"
    if abs(st["max"]) > 1e4 or abs(st["min"]) > 1e4:
        return "explode"
    return "OK"


def short(st):
    if st is None:
        return "?"
    return (
        f"min={st['min']:.4g} max={st['max']:.4g} mean={st['mean']:.4g} "
        f"nan={st['nan']} inf={st['inf']}"
    )


def main() -> None:
    lines = []
    if len(sys.argv) > 1:
        with open(sys.argv[1], errors="replace") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()
    lines = [l.rstrip("\n") for l in lines if "ZPROBE" in l]

    if not lines:
        print("NO ZPROBE LINES FOUND: was SGLANG_ZERO_PROBE=1 set at launch?")
        return

    fp = defaultdict(list)  # fingerprint lines, per rank
    probes = defaultdict(dict)  # (rank, extend) -> {tag: (in_st, out_st)}
    extends = {}  # (rank, extend) -> summary line
    s_logits = defaultdict(dict)  # rank -> {step: st}
    s_argmax = defaultdict(dict)  # rank -> {step: (npu, cpu, match)}
    errs = []

    for l in lines:
        m = PREFIX.search(l)
        if not m:
            continue
        rank, msg = int(m.group(1)), m.group(2)
        if any(
            k in msg
            for k in (
                "fingerprint ts",
                "sglang file",
                "torch=",
                "CANN version",
                "is_npu_arch35",
                "pkg ",
                "extra-ops so",
                "env:",
                "ops:",
                "selfcheck",
                "wrapped",
                "kernel selfcheck",
                "version.cfg",
            )
        ):
            fp[msg].append(rank)
            continue
        if "error" in msg:
            errs.append(f"r{rank}: {msg}")
            continue
        m = LAYER.search(msg)
        if m:
            probes[(rank, int(m.group(1)))][m.group(2)] = (
                parse_stats(m.group(3)),
                parse_stats(m.group(4)),
            )
            continue
        m = EXTEND.search(msg)
        if m:
            extends[(rank, int(m.group(1)))] = msg
            continue
        m = S_LOGITS.search(msg)
        if m:
            s_logits[rank][int(m.group(1))] = parse_stats(m.group(2))
            continue
        m = S_ARGMAX.search(msg)
        if m:
            s_argmax[rank][int(m.group(1))] = (
                m.group(2),
                m.group(3),
                m.group(4) == "True",
            )
            continue
        fp[msg].append(rank)

    print("=" * 72)
    print("ZPROBE REPORT")
    print("=" * 72)

    print("\n-- fingerprint (unique lines; [ranks]) --")
    for msg, ranks in fp.items():
        rs = sorted(set(ranks))
        tag = f"[r{rs[0]}]" if len(rs) == 1 else f"[ranks {','.join(map(str, rs))}]"
        print(f"{tag} {msg}")

    print("\n-- probe errors --")
    print("\n".join(errs) if errs else "none")

    print("\n-- extend batches --")
    for key in sorted(extends):
        print(f"[r{key[0]}] {extends[key]}")

    print("\n-- module scan (first bad module tag per rank/extend) --")

    def tag_order(tag):
        m = re.match(r"layer (\d+)(?: (\w+))?", tag)
        if m:
            part = m.group(2) or ""
            return (0, int(m.group(1)), {"": 0, "attn": 1, "moe": 2}.get(part, 3))
        return (1, 0, {"final_norm": 0, "lm_head": 1}.get(tag, 9))

    first_bad = {}
    for key in sorted(probes):
        tags = sorted(probes[key], key=tag_order)
        bad_tag = None
        for tag in tags:
            st_in, st_out = probes[key][tag]
            if classify(st_in) != "OK" or classify(st_out) != "OK":
                bad_tag = tag
                break
        first_bad[key] = bad_tag
        if bad_tag:
            lm = re.match(r"layer (\d+)", bad_tag)
            group_tags = (
                [t for t in tags if t.startswith(f"layer {lm.group(1)}")]
                if lm
                else [bad_tag]
            )
            print(f"[r{key[0]} extend {key[1]}] FIRST BAD at '{bad_tag}':")
            for t in group_tags:
                st_in, st_out = probes[key][t]
                print(
                    f"    {t}: in[{classify(st_in)} {short(st_in)}] "
                    f"out[{classify(st_out)} {short(st_out)}]"
                )
        else:
            outs = [probes[key][t][1] for t in tags]
            ok = [o for o in outs if o]
            mx = max((o["max"] for o in ok), default=0)
            mn = min((o["min"] for o in ok), default=0)
            print(
                f"[r{key[0]} extend {key[1]}] all {len(tags)} module tags OK "
                f"(out value range {mn:.4g}..{mx:.4g})"
            )

    print("\n-- sampler timeline --")
    all_steps_healthy = True
    for rank in sorted(s_logits):
        steps = sorted(s_logits[rank])
        bads = [
            (s, classify(s_logits[rank][s]))
            for s in steps
            if classify(s_logits[rank][s]) != "OK"
        ]
        mism = [s for s in steps if s in s_argmax[rank] and not s_argmax[rank][s][2]]
        print(
            f"[r{rank}] steps={len(steps)} bad_logits={len(bads)} "
            f"argmax_mismatch={len(mism)}"
        )
        for s, why in bads[:5]:
            print(f"    step {s}: {why} {short(s_logits[rank][s])}")
        if bads:
            all_steps_healthy = False
            print(f"    first bad step: {bads[0][0]} ({bads[0][1]})")
        if 1 in s_argmax[rank]:
            npu, cpu, match = s_argmax[rank][1]
            print(f"    step1 argmax npu[{npu}] cpu[{cpu}] match={match}")
        cpu_zeros = sum(
            1
            for s in steps
            if s in s_argmax[rank]
            and s_argmax[rank][s][1]
            and all(t == "0" for t in s_argmax[rank][s][1].split(","))
        )
        if cpu_zeros:
            print(
                f"    NOTE: {cpu_zeros} steps where CPU argmax is all-0 "
                "(finite logits genuinely point at token 0)"
            )

    print("\n-- VERDICT --")
    verdicts = []
    arch = [m for m in fp if m.startswith("is_npu_arch35=False")]
    if arch:
        verdicts.append("arch35 misdetected -> wrong kernel branch selected")
    if any("selfcheck argmax" in m and "match=False" in m for m in fp):
        verdicts.append("NPU argmax kernel broken (selfcheck)")
    for m in fp:
        mm = re.search(r"rel_err=([\d.eE+-]+)", m)
        if mm and float(mm.group(1)) > 0.05:
            verdicts.append(
                f"NPU matmul selfcheck rel_err={mm.group(1)} -> GEMM broken"
            )
    zero_ids = [
        e for e in extends.values() if re.search(r"input_ids\[:\d+\]=\[0(?:, 0)*\]", e)
    ]
    if zero_ids:
        verdicts.append("prefill input_ids all zero -> tokenizer/IPC input side")
    dead_first = sorted(
        r
        for r, steps in s_logits.items()
        if 1 in steps and classify(steps[1]) != "OK"
    )
    if dead_first:
        verdicts.append(
            f"logits dead at sampler entry from step 1 on ranks {dead_first} "
            f"({classify(next(iter(s_logits.values()))[1])}) -> forward/lm_head "
            "side; sampler and argmax exonerated by selfchecks"
        )
    for key, bad_tag in sorted(first_bad.items()):
        if not bad_tag:
            continue
        lm = re.match(r"layer (\d+)(?: (\w+))?", bad_tag)
        if not lm:
            family = {"final_norm": "final norm op", "lm_head": "lm_head matmul / "
                       "dp-lm-head gather"}.get(bad_tag, bad_tag)
        elif lm.group(1) == "0" and not lm.group(2):
            family = "embedding/input already bad (layer0 in)"
        elif lm.group(2) == "attn":
            family = f"attention op at layer {lm.group(1)}"
        elif lm.group(2) == "moe":
            family = f"MoE op at layer {lm.group(1)} (FP4 experts / deepep)"
        else:
            family = f"layer {lm.group(1)} output"
        verdicts.append(f"forward broken at r{key[0]} extend {key[1]}: {family}")
    if any("ops:" in m and "NO" in m for m in fp):
        verdicts.append("some expected custom ops unregistered (see ops: line)")
    if not verdicts and s_logits:
        ranks_bad_later = []
        for rank, steps in s_logits.items():
            ok = [classify(v) == "OK" for v in steps.values()]
            if all(ok):
                continue
            first_bad_step = min(s for s, o in zip(sorted(steps), ok) if not o)
            if first_bad_step > 1:
                ranks_bad_later.append((rank, first_bad_step))
        if ranks_bad_later:
            for rank, s in ranks_bad_later:
                verdicts.append(
                    f"r{rank}: prefill logits healthy, first bad decode-ish "
                    f"step {s} -> decode KV read / graph replay side"
                )
        else:
            verdicts.append(
                "probes all healthy -> corruption is post-sampling "
                "(id copy/overlap output path) or lm_head if step1 argmax sane"
            )
    if not verdicts:
        verdicts.append("no failure signature captured; check probe limits")
    for v in verdicts:
        print(f"* {v}")
    print("\nPaste this whole report back.")


if __name__ == "__main__":
    main()

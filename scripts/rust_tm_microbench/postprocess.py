#!/usr/bin/env python3
"""Join rust TTFT_STAMP + scheduler TTFT_STAMP_PY server-log lines with the
driver's client stamps into a per-request TTFT waterfall, then aggregate.

All stamps share CLOCK_MONOTONIC, so segments are plain differences. The wire
rid is the client rid plus a '#<16 hex>' uniquifier; it is stripped to join.

Usage:
  postprocess.py --server-log server.log [more.log ...] \
                 --driver in1024.jsonl [...] --out breakdown.json
"""

import argparse
import json
import re
import statistics
from collections import defaultdict

RUST_RE = re.compile(r"TTFT_STAMP rid=(\S+) st=(\S+) t=(\d+)")
PY_RE = re.compile(
    r"TTFT_STAMP_PY rid=(\S+) recv=([\d.]+) waitq=([\d.]+) fwd_entry=([\d.]+) "
    r"prefill_fin=([\d.]+) push_ns=(\d+)"
)
UNIQ_RE = re.compile(r"#[0-9a-f]{16}$")

# input_ids requests skip the tokenizer pool: one segment covers the whole
# ingress FSM instead of the four tokenize-path segments.
IDS_COLLAPSE = {"ingress_fsm", "tokenize", "tok_return_wait", "presend_validate"}
SEG_IDS = ("ingress_no_tokenize", "ing_pickup", "encode_start")

# (segment, from-stamp, to-stamp): telescopes client_send -> client_first.
SEGMENTS = [
    ("net_send", "client_send", "http_recv"),
    ("body_read_parse", "http_recv", "handler_entry"),
    ("fanout_validate", "handler_entry", "submit_ready"),
    ("tm_inbox_wait", "submit_ready", "ing_pickup"),
    ("ingress_fsm", "ing_pickup", "tok_start"),
    ("tokenize", "tok_start", "tok_done"),
    ("tok_return_wait", "tok_done", "ing_pickup2"),
    ("presend_validate", "ing_pickup2", "encode_start"),
    ("msgpack_encode_push", "encode_start", "ring_push"),
    ("ring_wait_drain", "ring_push", "sched_recv"),
    ("req_init_queue_add", "sched_recv", "waitq"),
    ("sched_queue_wait", "waitq", "fwd_entry"),
    ("prefill_forward", "fwd_entry", "prefill_fin"),
    ("output_batching", "prefill_fin", "push"),
    ("egress_dispatch", "push", "detok_first_recv"),
    ("detokenize_first", "detok_first_recv", "detok_first_emit"),
    ("sse_framing", "detok_first_emit", "sse_first_yield"),
    ("net_first_token", "sse_first_yield", "client_first"),
]


def parse_server_log(path):
    stamps = defaultdict(dict)  # client rid -> {stamp: ns}, first occurrence wins
    with open(path, errors="replace") as f:
        for line in f:
            m = RUST_RE.search(line)
            if m:
                rid = UNIQ_RE.sub("", m.group(1))
                stamps[rid].setdefault(m.group(2), int(m.group(3)))
                continue
            m = PY_RE.search(line)
            if m:
                d = stamps[UNIQ_RE.sub("", m.group(1))]
                for key, val in zip(
                    ("sched_recv", "waitq", "fwd_entry", "prefill_fin"), m.groups()[1:5]
                ):
                    d.setdefault(key, int(float(val) * 1e9))
                d.setdefault("push", int(m.group(6)))
    return stamps


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", nargs="+", required=True)
    ap.add_argument("--driver", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stamps = defaultdict(dict)
    for log in args.server_log:
        for rid, d in parse_server_log(log).items():
            stamps[rid].update(d)

    rows, missing = [], defaultdict(int)
    for path in args.driver:
        for line in open(path):
            r = json.loads(line)
            s = dict(stamps.get(r["rid"], {}))
            s["client_send"] = r["t_send_ns"]
            s["client_first"] = r["t_first_ns"]
            row = {
                "rid": r["rid"],
                "input_len": r["input_len_target"],
                "prompt_tokens": r.get("prompt_tokens"),
                "ttft_ms": r["ttft_ms"],
                "segments_ms": {},
                "incomplete": False,
            }
            chain = SEGMENTS if "tok_start" in s else [
                SEG_IDS if n[0] == "ingress_fsm" else n
                for n in SEGMENTS if n[0] not in IDS_COLLAPSE or n[0] == "ingress_fsm"
            ]
            for name, a, b in chain:
                if a in s and b in s:
                    row["segments_ms"][name] = (s[b] - s[a]) / 1e6
                else:
                    row["incomplete"] = True
                    missing[f"{name} ({a if a not in s else b})"] += 1
            if not row["incomplete"]:
                row["residual_ms"] = r["ttft_ms"] - sum(row["segments_ms"].values())
            rows.append(row)

    complete = [r for r in rows if not r["incomplete"]]
    print(f"{len(complete)}/{len(rows)} requests with complete stamp chains")
    for k, v in sorted(missing.items()):
        print(f"  missing: {k} x{v}")

    by_len = defaultdict(list)
    for r in complete:
        by_len[r["input_len"]].append(r)
    agg = {}
    for length, rs in sorted(by_len.items()):
        seg_stats = {}
        for name in rs[0]["segments_ms"]:
            vals = [r["segments_ms"][name] for r in rs]
            seg_stats[name] = {
                "mean": statistics.mean(vals),
                "p50": statistics.median(vals),
                "p99": pctl(vals, 0.99),
                "min": min(vals),
                "max": max(vals),
            }
        ttfts = [r["ttft_ms"] for r in rs]
        agg[str(length)] = {
            "n": len(rs),
            "prompt_tokens_mean": statistics.mean(
                r["prompt_tokens"] for r in rs if r["prompt_tokens"]
            ),
            "ttft_ms": {
                "mean": statistics.mean(ttfts),
                "p50": statistics.median(ttfts),
                "p99": pctl(ttfts, 0.99),
            },
            "residual_ms_mean": statistics.mean(r["residual_ms"] for r in rs),
            "segments": seg_stats,
        }

    with open(args.out, "w") as f:
        json.dump({"aggregate": agg, "requests": rows}, f, indent=1)

    lens = sorted(by_len)
    print(f"\n{'segment':<22}" + "".join(f"{length:>12}" for length in lens))
    for name in agg[str(lens[0])]["segments"]:
        row = "".join(f"{agg[str(le)]['segments'][name]['mean']:>12.3f}" for le in lens)
        print(f"{name:<22}{row}")
    print(
        f"{'TOTAL (client TTFT)':<22}"
        + "".join(f"{agg[str(le)]['ttft_ms']['mean']:>12.3f}" for le in lens)
    )
    print(
        f"{'residual':<22}"
        + "".join(f"{agg[str(le)]['residual_ms_mean']:>12.3f}" for le in lens)
    )


if __name__ == "__main__":
    main()

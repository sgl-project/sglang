#!/usr/bin/env python3
"""
Poll sglang's Prometheus /metrics endpoint at high frequency.

Subcommands:
  peek       one-shot list of available metrics + label sets
  poll       continuously sample /metrics, write JSONL
  summarize  read JSONL, print histogram averages and gauge stats over the run

Workflow:
  1) python3 sgl_metrics.py peek --url http://localhost:8002/metrics
     (see what's exposed; identify func_latency_seconds entries that
      correspond to tokenizer manager / pre-scheduler stages)

  2) python3 sgl_metrics.py poll --output run.jsonl --print &
     POLL_PID=$!
     python3 -m sglang.bench_serving ... your bench command ...
     kill $POLL_PID

  3) python3 sgl_metrics.py summarize run.jsonl
"""

import argparse
import json
import re
import signal
import sys
import time
import urllib.request
from collections import defaultdict

PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM_LINE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        labels = {}
        if m.group(2):
            for k, lv in LABEL_RE.findall(m.group(2)[1:-1]):
                labels[k] = lv.encode().decode("unicode_escape")
        out.append((m.group(1), labels, v))
    return out


def fetch(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def cmd_peek(args):
    text = fetch(args.url)
    helps, types = {}, {}
    for line in text.splitlines():
        if line.startswith("# HELP "):
            parts = line.split(" ", 3)
            if len(parts) == 4:
                helps[parts[2]] = parts[3]
        elif line.startswith("# TYPE "):
            parts = line.split(" ", 3)
            if len(parts) == 4:
                types[parts[2]] = parts[3]

    samples = parse_prom(text)
    label_sets = defaultdict(set)
    for name, labels, _ in samples:
        if labels:
            label_sets[name].add(tuple(sorted(labels.items())))

    names = sorted(set(list(helps) + list(types) + [s[0] for s in samples]))
    shown = 0
    for name in names:
        if args.filter and args.filter not in name:
            continue
        shown += 1
        kind = types.get(name, "?")
        print(f"{name}  [{kind}]")
        if name in helps:
            print(f"  {helps[name]}")
        if name in label_sets:
            for ls in sorted(label_sets[name])[:6]:
                print(f"    {dict(ls)}")
    print(f"\n# {shown} metrics matching filter={args.filter!r}", file=sys.stderr)


def cmd_poll(args):
    out = open(args.output, "w", buffering=1)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    print(f"poll {args.url} every {args.interval}s -> {args.output}", file=sys.stderr)
    n = 0
    last_print = 0.0
    while not stop[0]:
        t0 = time.time()
        try:
            text = fetch(args.url, timeout=max(args.interval * 5, 2.0))
            samples = parse_prom(text)
            if args.filter:
                samples = [s for s in samples if args.filter in s[0]]
            rec = {"t": t0, "m": [[nm, lb, v] for nm, lb, v in samples]}
            out.write(json.dumps(rec) + "\n")
            n += 1
            if args.print and t0 - last_print >= 1.0:
                last_print = t0
                quick = {}
                for nm, _, v in samples:
                    short = nm.rsplit(":", 1)[-1]
                    if short in (
                        "num_running_reqs",
                        "num_queue_reqs",
                        "cache_hit_rate",
                        "gen_throughput",
                        "num_used_tokens",
                    ):
                        quick[short] = v
                print(
                    f"[{n:>5} {time.strftime('%H:%M:%S')}] "
                    + " ".join(f"{k}={v:.4g}" for k, v in quick.items()),
                    file=sys.stderr,
                )
        except Exception as e:
            out.write(json.dumps({"t": t0, "err": str(e)}) + "\n")
        dt = time.time() - t0
        if dt < args.interval:
            time.sleep(args.interval - dt)
    out.close()
    print(f"wrote {n} samples to {args.output}", file=sys.stderr)


def cmd_summarize(args):
    series = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "m" not in rec:
                continue
            t = rec["t"]
            for nm, lb, v in rec["m"]:
                series[(nm, tuple(sorted(lb.items())))].append((t, v))

    counts, sums, gauges = {}, {}, {}
    for (name, labels), pts in series.items():
        if name.endswith("_count"):
            counts[(name[:-6], labels)] = pts
        elif name.endswith("_sum"):
            sums[(name[:-4], labels)] = pts
        elif not name.endswith(("_bucket", "_created")):
            gauges[(name, labels)] = pts

    print("# histograms (delta over run)")
    print(f"{'metric':<55}{'labels':<45}{'n':>7}{'sum_s':>10}{'avg_ms':>10}")
    rows = []
    for key in counts:
        if key not in sums:
            continue
        c = counts[key]
        s = sums[key]
        dn = c[-1][1] - c[0][1]
        ds = s[-1][1] - s[0][1]
        if dn <= 0:
            continue
        rows.append((key, dn, ds, ds / dn * 1000))
    rows.sort(key=lambda r: -r[3])
    for (name, labels), dn, ds, avg_ms in rows:
        lstr = ",".join(f"{k}={v}" for k, v in labels)[:43]
        print(f"{name[:53]:<55}{lstr:<45}{int(dn):>7}{ds:>10.2f}{avg_ms:>10.2f}")

    print("\n# gauges (min / mean / max)")
    print(f"{'metric':<55}{'labels':<45}{'min':>10}{'mean':>10}{'max':>10}")
    for (name, labels), pts in sorted(gauges.items()):
        vals = [v for _, v in pts]
        if not vals:
            continue
        lstr = ",".join(f"{k}={v}" for k, v in labels)[:43]
        print(
            f"{name[:53]:<55}{lstr:<45}"
            f"{min(vals):>10.4g}{sum(vals)/len(vals):>10.4g}{max(vals):>10.4g}"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    pk = sp.add_parser("peek")
    pk.add_argument("--url", default="http://localhost:8002/metrics")
    pk.add_argument("--filter", default="sglang")
    pk.set_defaults(func=cmd_peek)

    po = sp.add_parser("poll")
    po.add_argument("--url", default="http://localhost:8002/metrics")
    po.add_argument("--interval", type=float, default=0.2)
    po.add_argument("--output", default="metrics.jsonl")
    po.add_argument("--filter", default="sglang")
    po.add_argument("--print", action="store_true")
    po.set_defaults(func=cmd_poll)

    su = sp.add_parser("summarize")
    su.add_argument("input")
    su.set_defaults(func=cmd_summarize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

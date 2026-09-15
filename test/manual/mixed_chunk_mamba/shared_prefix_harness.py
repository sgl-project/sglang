#!/usr/bin/env python3
"""Shared-prefix mismatch / latency harness for sglang #39342 (chat completions,
thinking disabled, explicit rids).

Protocol per run (identical prompts, seed, sampling and reset protocol for
every build/arm):
  ref_c1_cold        flush, concurrency 1 (no co-batching possible)
  ref_c1_warm        concurrency 1 again on the warm cache (c1 noise floor)
  cN_cold_i / warm_i flush, concurrency N cold, then warm, repeated R times
Reports per pass: mismatches vs ref_c1_cold, wrong vs the known facts, latency
percentiles; plus pairwise mismatches between the cN passes (concurrent
run-to-run variability) and the scheduler-log counts of prefill batches that
co-batched running requests and the cached tokens they hit.
"""

import argparse
import concurrent.futures as cf
import itertools
import json
import random
import re
import statistics
import time

import requests

PREFILL_LINE = re.compile(
    r"Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+)"
)


def build(seed, n_facts, n_prompts):
    rng = random.Random(seed)
    facts = [rng.randint(0, 999) for _ in range(n_facts)]
    prefix = " ".join(f"fact{i}: the value is {facts[i]}." for i in range(n_facts))
    idx = [rng.randint(0, n_facts - 1) for _ in range(n_prompts)]
    questions = [f"What is fact{i}? Answer with the number only." for i in idx]
    expected = [str(facts[i]) for i in idx]
    return prefix, questions, expected


def chat(url, model, rid, prefix, user, max_tokens):
    t0 = time.perf_counter()
    r = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "rid": rid,
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=1200,
    )
    r.raise_for_status()
    b = r.json()
    usage = b.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return (
        b["choices"][0]["message"]["content"].strip(),
        time.perf_counter() - t0,
        cached,
    )


def flush(url):
    requests.post(f"{url}/flush_cache", timeout=300).raise_for_status()
    time.sleep(1.0)


def run_pass(url, model, tag, prefix, questions, concurrency, max_tokens):
    with cf.ThreadPoolExecutor(concurrency) as ex:
        res = list(
            ex.map(
                lambda iq: chat(
                    url, model, f"{tag}-{iq[0]}", prefix, iq[1], max_tokens
                ),
                enumerate(questions),
            )
        )
    return [o for o, _, _ in res], [l for _, l, _ in res], [c for _, _, c in res]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def log_counts(path, offset):
    with open(path, "r", errors="replace") as f:
        f.seek(offset)
        text = f.read()
        end = f.tell()
    total = mixed = cached = 0
    for m in PREFILL_LINE.finditer(text):
        total += 1
        cached += int(m.group(3))
        if int(m.group(4)) > 0:
            mixed += 1
    tracked_forwards = sum(
        1
        for line in text.splitlines()
        if "[39342] forward" in line
        and "tracked=[(" in line
        and "one_token_rows=0" not in line
    )
    return {
        "prefill_batches": total,
        "prefill_batches_with_running_reqs": mixed,
        "cached_tokens": cached,
        "mixed_forwards_with_tracked_rows": tracked_forwards,
    }, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--prefix-facts", type=int, default=380)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prefix, questions, expected = build(args.seed, args.prefix_facts, args.prompts)
    with open(args.log, "rb") as f:
        f.seek(0, 2)
        offset = f.tell()

    report = {"tag": args.tag, "n": len(questions), "passes": []}
    outputs = {}

    def record(name, outs, lats, cached, ref):
        nonlocal offset
        counts, offset = log_counts(args.log, offset)
        entry = {
            "pass": name,
            "mismatch_vs_ref": None
            if ref is None
            else sum(a != b for a, b in zip(outs, ref)),
            "wrong_vs_expected": sum(
                not o.startswith(e) for o, e in zip(outs, expected)
            ),
            "p50_s": pct(lats, 50),
            "p90_s": pct(lats, 90),
            "p99_s": pct(lats, 99),
            "mean_s": statistics.fmean(lats),
            "mean_cached_tokens": (
                statistics.fmean(c for c in cached if c is not None)
                if any(c is not None for c in cached)
                else None
            ),
            "log": counts,
        }
        outputs[name] = outs
        report["passes"].append(entry)
        print(json.dumps(entry), flush=True)

    flush(args.url)
    ref, lats, cached = run_pass(
        args.url, args.model, f"{args.tag}-ref", prefix, questions, 1, args.max_tokens
    )
    record("ref_c1_cold", ref, lats, cached, None)
    o, lats, cached = run_pass(
        args.url, args.model, f"{args.tag}-refw", prefix, questions, 1, args.max_tokens
    )
    record("ref_c1_warm", o, lats, cached, ref)
    for i in range(args.repeats):
        flush(args.url)
        o, lats, cached = run_pass(
            args.url,
            args.model,
            f"{args.tag}-c{i}",
            prefix,
            questions,
            args.concurrency,
            args.max_tokens,
        )
        record(f"c{args.concurrency}_cold_{i}", o, lats, cached, ref)
        o, lats, cached = run_pass(
            args.url,
            args.model,
            f"{args.tag}-w{i}",
            prefix,
            questions,
            args.concurrency,
            args.max_tokens,
        )
        record(f"c{args.concurrency}_warm_{i}", o, lats, cached, ref)

    conc = [k for k in outputs if k.startswith(f"c{args.concurrency}_")]
    report["pairwise_concurrent_mismatch"] = {
        f"{a}|{b}": sum(x != y for x, y in zip(outputs[a], outputs[b]))
        for a, b in itertools.combinations(conc, 2)
    }
    report["outputs"] = outputs
    report["expected"] = expected
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("pairwise:", json.dumps(report["pairwise_concurrent_mismatch"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

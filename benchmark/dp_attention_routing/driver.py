"""Open-loop mixed long/short trace for comparing DP-attention routing.

Sends the same synthetic trace to a running SGLang server and records, for every
request, the DP rank it was routed to and the wall-clock time of every streamed
token. Results go to OUT/{warmup,burst-N,staggered-N}.json; summarize.py turns
them into the tables in the PR.

Trace (per trial):
  * 30 short requests: 1024 random prompt tokens, 96 output tokens, one every 0.4 s.
  * One 32K and one 64K prompt, 1 output token each. Their order alternates
    between trials. The first arrives at 1.05 s; the second arrives 0.2 s later
    ("burst") or 4 s later ("staggered").
Prompts are random token ids, so the tokenizer and prefix cache play no role.
"""

import argparse
import concurrent.futures
import json
import math
import random
import time
import urllib.request
from pathlib import Path

PROFILES = ("burst", "staggered")
SHORT_PROMPT = 1024
SHORT_OUTPUT = 96
NUM_SHORT = 30
TRIALS = 3


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def pack_requests(profile, rep):
    short = [(i * 0.4, SHORT_PROMPT, SHORT_OUTPUT) for i in range(NUM_SHORT)]
    lengths = (32768, 65536) if rep % 2 == 0 else (65536, 32768)
    second_long = 1.25 if profile == "burst" else 5.05
    long = [(when, n, 1) for when, n in zip((1.05, second_long), lengths)]
    requests = []
    for i, (when, n, gen) in enumerate(sorted(short + long)):
        rng = random.Random(10_000 + 100 * rep + i)
        body = json.dumps(
            {
                "input_ids": [rng.randrange(1000, 6000) for _ in range(n)],
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen,
                    "ignore_eos": True,
                },
            }
        ).encode()
        requests.append((when, n, gen, body))
    return requests


def send(url, origin, spec):
    offset, n, gen, body = spec
    req = urllib.request.Request(
        url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    time.sleep(max(0, origin + offset - time.monotonic()))
    sent = time.monotonic()
    events = []
    meta = {}
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            now = time.monotonic()
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            if "error" in item:
                raise RuntimeError(item["error"])
            meta = item.get("meta_info", {})
            count = meta.get("completion_tokens", 0)
            if count > (events[-1][1] if events else 0):
                events.append((now - origin, count))

    if not events or events[-1][1] != gen or meta.get("prompt_tokens") != n:
        raise RuntimeError(f"Expected prompt={n} output={gen}, got {meta}")
    rank = meta.get("dp_rank")
    if rank is None:
        raise RuntimeError(f"No DP rank in the response: {meta}")
    return {
        "scheduled_s": offset,
        "sent_s": sent - origin,
        "prompt_tokens": n,
        "output_tokens": gen,
        "dp_rank": rank,
        "ttft_ms": 1000 * (origin + events[0][0] - sent),
        # (seconds since trial start, completion tokens so far) per streamed chunk
        "events": events,
    }


def summarize(requests):
    short = [r for r in requests if r["prompt_tokens"] == SHORT_PROMPT]
    long = [r for r in requests if r["prompt_tokens"] != SHORT_PROMPT]
    gaps = [
        1000 * (b[0] - a[0])
        for req in short
        for a, b in zip(req["events"], req["events"][1:])
    ]
    ranks = sorted({r["dp_rank"] for r in requests})
    rank_prompt = [
        sum(r["prompt_tokens"] for r in requests if r["dp_rank"] == rank)
        for rank in ranks
    ]
    duration = max(r["events"][-1][0] for r in requests) - min(
        r["sent_s"] for r in requests
    )
    return {
        "rank_prompt_tokens": rank_prompt,
        "hot_rank_prompt_share_pct": round(
            100 * max(rank_prompt) / sum(rank_prompt), 3
        ),
        "long_ttft_ms": {str(r["prompt_tokens"]): round(r["ttft_ms"], 3) for r in long},
        "long_ranks": {str(r["prompt_tokens"]): r["dp_rank"] for r in long},
        "short_gap_p99_ms": round(percentile(gaps, 0.99), 3),
        "short_gaps_over_500ms_per_request": round(
            sum(g > 500 for g in gaps) / len(short), 3
        ),
        "output_tok_s": round(sum(r["output_tokens"] for r in requests) / duration, 3),
        "duration_s": round(duration, 3),
        "max_submit_lag_ms": round(
            1000 * max(r["sent_s"] - r["scheduled_s"] for r in requests), 3
        ),
    }


def trial(pool, url, profile, rep):
    specs = pack_requests(profile, rep)
    origin = time.monotonic() + 0.7
    futures = [pool.submit(send, url, origin, spec) for spec in specs]
    requests = [future.result() for future in futures]
    return {
        "profile": profile,
        "rep": rep,
        "summary": summarize(requests),
        "requests": requests,
    }


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:31023")
    parser.add_argument(
        "--out", required=True, help="new directory for this run's results"
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for endpoint in ("/model_info", "/server_info"):
        with urllib.request.urlopen(args.url + endpoint, timeout=30) as response:
            save(out / (endpoint[1:] + ".json"), json.load(response))

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        warmup = trial(pool, args.url, "burst", 3)
        save(out / "warmup.json", warmup)
        print("warmup", json.dumps(warmup["summary"], sort_keys=True), flush=True)
        for rep in range(TRIALS):
            # Rotate the profile order to limit carry-over from the previous trial.
            for profile in PROFILES[rep % 2 :] + PROFILES[: rep % 2]:
                result = trial(pool, args.url, profile, rep)
                save(out / f"{profile}-{rep}.json", result)
                print(
                    profile,
                    rep,
                    json.dumps(result["summary"], sort_keys=True),
                    flush=True,
                )
                time.sleep(1)


if __name__ == "__main__":
    main()

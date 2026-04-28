#!/usr/bin/env python3
"""
LPLB vs Dynamic dispatch benchmark.

Usage:
    python3 benchmark_lplb.py <dispatch> <dataset> <start_idx> <output_dir>

Example:
    python3 benchmark_lplb.py lp mmlu 1000 /raid/fei/lplb/results
"""
import asyncio, aiohttp, json, os, sys, time
import numpy as np
from datasets import load_dataset

DISPATCH = sys.argv[1]       # "dynamic" or "lp"
DATASET = sys.argv[2]        # "mmlu"
START = int(sys.argv[3])     # start index (skip stat-collection portion)
OUTPUT_DIR = sys.argv[4]
CONC = int(sys.argv[5]) if len(sys.argv) > 5 else 128
MAX_TOKENS = int(sys.argv[6]) if len(sys.argv) > 6 else 1
SERVER = "http://localhost:30000"
WARMUP_N = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def verify_backend(server: str, expected: str) -> None:
    """Hit /server_info and abort if `ep_dispatch_algorithm` doesn't match.

    Guards against the case where a previous server (with a different config)
    is still answering /health while the current one boots — otherwise we'd
    silently benchmark the wrong dispatcher.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{server}/server_info", timeout=10) as r:
            info = json.load(r)
    except Exception as e:
        raise SystemExit(f"[{DISPATCH}] Failed to fetch /server_info: {e}")
    actual = info.get("ep_dispatch_algorithm")
    if actual != expected:
        raise SystemExit(
            f"[{DISPATCH}] Server reports ep_dispatch_algorithm={actual!r} but "
            f"benchmark expects {expected!r}. Aborting to avoid silently "
            "measuring the wrong dispatcher."
        )
    print(f"[{DISPATCH}] Verified server is running ep_dispatch_algorithm={actual!r}", flush=True)


verify_backend(SERVER, DISPATCH)

# Load dataset
prompts = []
if DATASET == "mmlu":
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    for i, item in enumerate(ds):
        if i >= 2000: break
        ch = item["choices"]
        prompts.append(f"Q: {item['question']} A) {ch[0]} B) {ch[1]} C) {ch[2]} D) {ch[3]}")

prompts = prompts[START:]
print(f"[{DISPATCH}/{DATASET}] Loaded {len(prompts)} prompts (start={START})", flush=True)


async def send_one(s, sem, prompt, out):
    async with sem:
        t0 = time.perf_counter()
        try:
            async with s.post(
                SERVER + "/v1/completions",
                json={"model": "default", "prompt": prompt, "max_tokens": MAX_TOKENS},
                timeout=aiohttp.ClientTimeout(total=3600),
            ) as r:
                d = await r.json()
                dt = time.perf_counter() - t0
                if r.status == 200:
                    usage = d.get("usage", {}) or {}
                    out.append({
                        "ttft": dt,
                        "ptok": usage.get("prompt_tokens", 0),
                        "ctok": usage.get("completion_tokens", 0),
                        "status": 200,
                    })
                else:
                    out.append({"ttft": dt, "err": True, "status": r.status})
        except Exception as e:
            out.append({"ttft": time.perf_counter() - t0, "err": True, "exc": str(e)})


async def send_batch(prompts_batch):
    sem = asyncio.Semaphore(CONC)
    out = []
    async with aiohttp.ClientSession() as s:
        await asyncio.gather(*[send_one(s, sem, p, out) for p in prompts_batch])
    return out


# Warmup (excluded from metrics)
warmup_prompts = prompts[:WARMUP_N]
bench_prompts = prompts[WARMUP_N:]
print(f"[{DISPATCH}/{DATASET}] Warmup: {len(warmup_prompts)} requests...", flush=True)
asyncio.run(send_batch(warmup_prompts))
print(f"[{DISPATCH}/{DATASET}] Warmup done", flush=True)

# Benchmark
print(f"[{DISPATCH}/{DATASET}] Benchmarking {len(bench_prompts)} prompts, concurrency={CONC}...", flush=True)
t0 = time.time()
res = asyncio.run(send_batch(bench_prompts))
wall = time.time() - t0

ok = [r for r in res if "err" not in r]
ttfts = [r["ttft"] for r in ok]
ptok = sum(r["ptok"] for r in ok)
ctok = sum(r.get("ctok", 0) for r in ok)
errs = len(res) - len(ok)

result = {
    "dispatch": DISPATCH,
    "dataset": DATASET,
    "start_idx": START,
    "n_prompts": len(bench_prompts),
    "n_warmup": len(warmup_prompts),
    "n_ok": len(ok),
    "n_errors": errs,
    "wall_s": round(wall, 3),
    "concurrency": CONC,
    "max_tokens": MAX_TOKENS,
    "tput_input_tok_s": round(ptok / wall, 1) if wall else 0,
    "tput_output_tok_s": round(ctok / wall, 1) if wall else 0,
    "tput_total_tok_s": round((ptok + ctok) / wall, 1) if wall else 0,
}
if ttfts:
    result.update({
        "lat_p50": round(float(np.percentile(ttfts, 50)), 4),
        "lat_p90": round(float(np.percentile(ttfts, 90)), 4),
        "lat_p99": round(float(np.percentile(ttfts, 99)), 4),
        "lat_mean": round(float(np.mean(ttfts)), 4),
        "lat_min": round(float(np.min(ttfts)), 4),
        "lat_max": round(float(np.max(ttfts)), 4),
    })

rpath = f"{OUTPUT_DIR}/{DISPATCH}_{DATASET}_result.json"
with open(rpath, "w") as f:
    json.dump(result, f, indent=2)

print(f"[{DISPATCH}/{DATASET}] p50={result.get('ttft_p50','?')}s "
      f"p90={result.get('ttft_p90','?')}s "
      f"tput={result.get('tput_input_tok_s','?')} tok/s "
      f"err={errs} "
      f"saved={rpath}", flush=True)

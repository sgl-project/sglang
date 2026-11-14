"""Finite cold-cache decode benchmark using synthetic token IDs.

Run against an otherwise idle SGLang server. This clears its prefix cache.
Example:
    python benchmark/serving/benchmark_h20_decode.py \
        --model-path /path/to/DeepSeek-V4.1-Flash --output before.jsonl

Repeat after installing the H20-3e configurations with identical server flags.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import statistics
import time
import urllib.request

from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--label", default="measurement")
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--model-path", type=pathlib.Path, required=True)
p.add_argument("--output", type=pathlib.Path, required=True)
args = p.parse_args()
out = args.output
assert not out.exists()
tokenizer = Tokenizer.from_file(str(args.model_path / "tokenizer.json"))
filler = tokenizer.encode(
    (
        "The research server processes independent records. Each record contains an identifier, a timestamp, and a measured value. We compare latency using the same data and the same output budget.\n"
    )
    * 16000,
    add_special_tokens=False,
).ids


def emit(row):
    row = {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        **row,
    }
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def request(path, body=None):
    return urllib.request.urlopen(
        urllib.request.Request(
            args.url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=None,
    )


def flush():
    with request("/flush_cache?timeout=30", {}) as r:
        r.read()


def native(case, isl, osl, index):
    ids = (
        tokenizer.encode(
            f"Independent record {index}: {case}.\n", add_special_tokens=False
        ).ids
        + filler
    )[:isl]
    start = time.perf_counter()
    first = last = None
    final = None
    events = 0
    body = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": osl,
            "ignore_eos": True,
        },
        "stream": True,
    }
    with request("/generate", body) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            if value == b"[DONE]":
                break
            obj = json.loads(value)
            now = time.perf_counter()
            if obj.get("error"):
                raise RuntimeError(str(obj["error"]))
            if obj.get("meta_info", {}).get("completion_tokens", 0) > 0:
                first = first or now
                last = now
                final = obj
                events += 1
    elapsed = time.perf_counter() - start
    assert final is not None, (case, index)
    meta = final["meta_info"]
    assert meta["completion_tokens"] == osl and meta["prompt_tokens"] == isl, meta
    assert meta.get("cached_tokens") == 0, meta
    return {
        "kind": "request",
        "case": case,
        "index": index,
        "input_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "prompt_tokens": isl,
        "completion_tokens": osl,
        "cached_tokens": meta["cached_tokens"],
        "ttft_s": first - start,
        "elapsed_s": elapsed,
        "decode_tokens_per_s": (osl - 1) / (last - first) if osl > 1 else None,
        "tpot_ms": 1000 * (last - first) / (osl - 1) if osl > 1 else None,
        "events": events,
    }


cases = [
    ("8k_1000_c1", 8192, 1000, 1),
    ("1k_128_c1", 1024, 128, 1),
    ("1k_128_c8", 1024, 128, 8),
    ("1k_128_c32", 1024, 128, 32),
    ("8k_128_c8", 8192, 128, 8),
    ("32k_128_c1", 32768, 128, 1),
]
emit(
    {
        "kind": "start",
        "cases": cases,
        "repetitions": 3,
    }
)
try:
    for case, isl, osl, concurrency in cases:
        flush()
        emit({"kind": "warmup_start", "case": case})
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(
                pool.map(
                    lambda i: native("warm_" + case, isl, 16, i), range(concurrency)
                )
            )
        for repeat in range(3):
            flush()
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows = list(
                    pool.map(
                        lambda i: native(case, isl, osl, i + repeat * 100),
                        range(concurrency),
                    )
                )
            wall = time.perf_counter() - start
            for row in rows:
                emit({**row, "repeat": repeat})
            emit(
                {
                    "kind": "batch",
                    "case": case,
                    "repeat": repeat,
                    "concurrency": concurrency,
                    "wall_s": wall,
                    "output_tokens_per_s": concurrency * osl / wall,
                    "median_ttft_s": statistics.median(x["ttft_s"] for x in rows),
                    "median_decode_tokens_per_s": statistics.median(
                        x["decode_tokens_per_s"] for x in rows
                    ),
                    "median_tpot_ms": statistics.median(x["tpot_ms"] for x in rows),
                }
            )
    emit({"kind": "complete"})
except Exception as exc:
    emit({"kind": "failed", "error": repr(exc)})
    raise

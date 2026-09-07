"""HiCache L2 load-back mixed-batch E2E benchmark.

This benchmark creates a deliberately favorable case for batch-level splitting:
some requests hit a long prefix in host HiCache and need H2D load-back, while
short cache-miss requests are submitted at the same time. It compares mixed
submission against fast-only and split fast-then-slow controls.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import aiohttp
import requests

from sglang.benchmark.utils import get_tokenizer, remove_prefix

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)
AIOHTTP_READ_BUFSIZE = 10 * 1024**2


@dataclass
class RequestResult:
    label: str
    kind: str
    success: bool
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cached_tokens_details: Optional[Dict[str, Any]] = None
    error: str = ""
    start_offset_ms: float = 0.0


@dataclass
class ScenarioResult:
    scenario: str
    prompt_len: int
    fast_count: int
    slow_count: int
    results: List[RequestResult] = field(default_factory=list)
    elapsed_ms: float = 0.0


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = index - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
        "max": max(vals),
    }


def get_vocab_ids(tokenizer: Any, seed: int, limit: int) -> List[int]:
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids = [
        int(v)
        for v in vocab.values()
        if isinstance(v, int) and v >= 0 and int(v) not in special
    ]
    if not ids:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        ids = [i for i in range(vocab_size) if i not in special]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:limit]


def make_token_ids(
    vocab_ids: List[int], token_num: int, rng: random.Random
) -> List[int]:
    return rng.choices(vocab_ids, k=token_num)


def build_prompts(
    vocab_ids: List[int],
    prompt_len: int,
    slow_count: int,
    fast_count: int,
    fast_len: int,
    suffix_len: int,
    evict_count: int,
    evict_len: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    slow_prefixes = [
        make_token_ids(vocab_ids, prompt_len, rng) for _ in range(slow_count)
    ]
    slow_suffixes = [
        make_token_ids(vocab_ids, suffix_len, rng) for _ in range(slow_count)
    ]
    fast = [make_token_ids(vocab_ids, fast_len, rng) for _ in range(fast_count)]
    evict = [make_token_ids(vocab_ids, evict_len, rng) for _ in range(evict_count)]

    return {
        "warm": [
            {"label": f"warm-{i}", "kind": "warm", "input_ids": ids}
            for i, ids in enumerate(slow_prefixes)
        ],
        "slow": [
            {
                "label": f"slow-{i}",
                "kind": "slow",
                "input_ids": prefix + suffix,
            }
            for i, (prefix, suffix) in enumerate(zip(slow_prefixes, slow_suffixes))
        ],
        "fast": [
            {"label": f"fast-{i}", "kind": "fast", "input_ids": ids}
            for i, ids in enumerate(fast)
        ],
        "evict": [
            {"label": f"evict-{i}", "kind": "evict", "input_ids": ids}
            for i, ids in enumerate(evict)
        ],
    }


def wait_server(base_url: str, timeout_s: float) -> Dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            resp = requests.get(f"{base_url}/server_info", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server not ready after {timeout_s}s: {last_error}")


def flush_cache(
    base_url: str,
    retries: int = 120,
    retry_interval_s: float = 0.5,
) -> None:
    last_error = ""
    for _ in range(retries):
        resp = requests.post(f"{base_url}/flush_cache", timeout=60)
        if resp.status_code == 200:
            return
        last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        time.sleep(retry_interval_s)
    raise RuntimeError(f"flush_cache failed after {retries} retries: {last_error}")


async def request_one(
    session: aiohttp.ClientSession,
    url: str,
    item: Dict[str, Any],
    output_len: int,
    start_time: float,
    start_delay_ms: float = 0.0,
) -> RequestResult:
    if start_delay_ms > 0:
        await asyncio.sleep(start_delay_ms / 1000.0)

    payload = {
        "input_ids": item["input_ids"],
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    req_start = time.perf_counter()
    result = RequestResult(
        label=item["label"],
        kind=item["kind"],
        success=False,
        start_offset_ms=(req_start - start_time) * 1000.0,
    )
    last_meta: Dict[str, Any] = {}
    latency = 0.0

    try:
        async with session.post(url=url, json=payload) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result

            async for chunk_bytes in resp.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                latency = time.perf_counter() - req_start
                if chunk == "[DONE]":
                    continue
                data = json.loads(chunk)
                meta = data.get("meta_info") or {}
                if meta:
                    last_meta = meta
                if result.ttft_ms == 0.0 and (
                    data.get("output_ids") or meta.get("completion_tokens", 0) > 0
                ):
                    result.ttft_ms = latency * 1000.0

        result.success = True
        result.latency_ms = latency * 1000.0
        result.prompt_tokens = int(
            last_meta.get("prompt_tokens", len(item["input_ids"]))
        )
        result.completion_tokens = int(last_meta.get("completion_tokens", 0))
        result.cached_tokens = int(last_meta.get("cached_tokens", 0))
        result.cached_tokens_details = last_meta.get("cached_tokens_details")
        return result
    except Exception as exc:
        result.error = repr(exc)
        return result


async def run_items(
    base_url: str,
    items: List[Dict[str, Any]],
    output_len: int,
    max_concurrency: int,
    delay_by_kind_ms: Optional[Dict[str, float]] = None,
) -> List[RequestResult]:
    url = f"{base_url}/generate"
    connector = aiohttp.TCPConnector(limit=max(max_concurrency, len(items)) + 4)
    async with aiohttp.ClientSession(
        timeout=AIOHTTP_TIMEOUT,
        read_bufsize=AIOHTTP_READ_BUFSIZE,
        connector=connector,
    ) as session:
        semaphore = asyncio.Semaphore(max_concurrency)
        start_time = time.perf_counter()

        async def limited(item: Dict[str, Any]) -> RequestResult:
            async with semaphore:
                delay = (delay_by_kind_ms or {}).get(item["kind"], 0.0)
                return await request_one(
                    session, url, item, output_len, start_time, delay
                )

        tasks = [asyncio.create_task(limited(item)) for item in items]
        return await asyncio.gather(*tasks)


def print_result_table(scenario: ScenarioResult) -> None:
    fast_ttft = [r.ttft_ms for r in scenario.results if r.success and r.kind == "fast"]
    slow_ttft = [r.ttft_ms for r in scenario.results if r.success and r.kind == "slow"]
    slow_host = [
        float((r.cached_tokens_details or {}).get("host", 0))
        for r in scenario.results
        if r.success and r.kind == "slow"
    ]
    failures = [r for r in scenario.results if not r.success]
    row = {
        "scenario": scenario.scenario,
        "prompt_len": scenario.prompt_len,
        "fast_count": scenario.fast_count,
        "slow_count": scenario.slow_count,
        "elapsed_ms": round(scenario.elapsed_ms, 2),
        "fast_ttft_ms": {k: round(v, 2) for k, v in summarize(fast_ttft).items()},
        "slow_ttft_ms": {k: round(v, 2) for k, v in summarize(slow_ttft).items()},
        "slow_host_tokens": {k: round(v, 2) for k, v in summarize(slow_host).items()},
        "failures": len(failures),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)


def write_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def scenario_record(scenario: ScenarioResult) -> Dict[str, Any]:
    return {
        "scenario": scenario.scenario,
        "prompt_len": scenario.prompt_len,
        "fast_count": scenario.fast_count,
        "slow_count": scenario.slow_count,
        "elapsed_ms": scenario.elapsed_ms,
        "results": [r.__dict__ for r in scenario.results],
    }


async def prepare_host_hit(
    args: argparse.Namespace,
    prompt_sets: Dict[str, List[Dict[str, Any]]],
) -> None:
    flush_cache(args.base_url)
    await run_items(
        args.base_url,
        prompt_sets["warm"],
        args.output_len,
        max(1, min(args.slow_count, args.max_concurrency)),
    )
    if args.post_warm_sleep > 0:
        await asyncio.sleep(args.post_warm_sleep)

    for begin in range(0, len(prompt_sets["evict"]), args.evict_parallel):
        batch = prompt_sets["evict"][begin : begin + args.evict_parallel]
        await run_items(
            args.base_url,
            batch,
            args.output_len,
            max(1, min(args.evict_parallel, args.max_concurrency)),
        )
    if args.post_evict_sleep > 0:
        await asyncio.sleep(args.post_evict_sleep)


async def run_scenario(
    args: argparse.Namespace,
    name: str,
    prompt_len: int,
    fast_count: int,
    prompt_sets: Dict[str, List[Dict[str, Any]]],
    needs_prepare: bool,
    delay_by_kind_ms: Optional[Dict[str, float]] = None,
) -> ScenarioResult:
    if needs_prepare:
        await prepare_host_hit(args, prompt_sets)
    else:
        flush_cache(args.base_url)

    if name == "fast_only":
        items = prompt_sets["fast"][:fast_count]
    elif name == "slow_only":
        items = prompt_sets["slow"]
    elif name == "split_fast_then_slow":
        start = time.perf_counter()
        fast_results = await run_items(
            args.base_url,
            prompt_sets["fast"][:fast_count],
            args.output_len,
            args.max_concurrency,
        )
        slow_results = await run_items(
            args.base_url,
            prompt_sets["slow"],
            args.output_len,
            args.max_concurrency,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        return ScenarioResult(
            scenario=name,
            prompt_len=prompt_len,
            fast_count=fast_count,
            slow_count=args.slow_count,
            results=fast_results + slow_results,
            elapsed_ms=elapsed,
        )
    else:
        items = prompt_sets["slow"] + prompt_sets["fast"][:fast_count]

    start = time.perf_counter()
    results = await run_items(
        args.base_url,
        items,
        args.output_len,
        args.max_concurrency,
        delay_by_kind_ms=delay_by_kind_ms,
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    return ScenarioResult(
        scenario=name,
        prompt_len=prompt_len,
        fast_count=fast_count,
        slow_count=args.slow_count,
        results=results,
        elapsed_ms=elapsed,
    )


def server_summary(info: Dict[str, Any]) -> Dict[str, Any]:
    memory_usage = info.get("memory_usage")
    internal_states = info.get("internal_states") or []
    if memory_usage is None and internal_states:
        memory_usage = internal_states[0].get("memory_usage")
    return {
        "status": info.get("status"),
        "version": info.get("version"),
        "model_path": info.get("model_path"),
        "context_length": info.get("context_length"),
        "max_total_num_tokens": info.get("max_total_num_tokens"),
        "schedule_policy": info.get("schedule_policy"),
        "enable_hierarchical_cache": info.get("enable_hierarchical_cache"),
        "hicache_size": info.get("hicache_size"),
        "hicache_write_policy": info.get("hicache_write_policy"),
        "hicache_io_backend": info.get("hicache_io_backend"),
        "hicache_mem_layout": info.get("hicache_mem_layout"),
        "memory_usage": memory_usage,
    }


async def main_async(args: argparse.Namespace) -> None:
    info = wait_server(args.base_url, args.wait_timeout)
    print(
        json.dumps({"server_info": server_summary(info)}, ensure_ascii=False),
        flush=True,
    )

    tokenizer = get_tokenizer(args.model_path)
    vocab_ids = get_vocab_ids(tokenizer, args.seed, args.vocab_limit)
    if not vocab_ids:
        raise RuntimeError("failed to construct usable vocab ids")

    scenarios = [
        ("fast_only", False, None),
        ("slow_only", True, None),
        ("mixed_slow_first", True, None),
        ("mixed_fast_delayed", True, {"fast": args.fast_delay_ms}),
        ("split_fast_then_slow", True, None),
    ]

    for prompt_len in args.prompt_lens:
        evict_len = args.evict_len or prompt_len
        for fast_count in args.fast_counts:
            prompt_sets = build_prompts(
                vocab_ids=vocab_ids,
                prompt_len=prompt_len,
                slow_count=args.slow_count,
                fast_count=fast_count,
                fast_len=args.fast_len,
                suffix_len=args.suffix_len,
                evict_count=args.evict_count,
                evict_len=evict_len,
                seed=args.seed + prompt_len * 1000 + fast_count,
            )
            for scenario_name, needs_prepare, delay_by_kind_ms in scenarios:
                scenario = await run_scenario(
                    args,
                    scenario_name,
                    prompt_len,
                    fast_count,
                    prompt_sets,
                    needs_prepare,
                    delay_by_kind_ms=delay_by_kind_ms,
                )
                print_result_table(scenario)
                write_jsonl(args.output_jsonl, scenario_record(scenario))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-lens", type=parse_csv_ints, default=[4096, 8192])
    parser.add_argument("--fast-counts", type=parse_csv_ints, default=[1, 8, 32])
    parser.add_argument("--slow-count", type=int, default=1)
    parser.add_argument("--fast-len", type=int, default=256)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--evict-count", type=int, default=6)
    parser.add_argument("--evict-len", type=int, default=0)
    parser.add_argument("--evict-parallel", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--fast-delay-ms", type=float, default=20.0)
    parser.add_argument("--post-warm-sleep", type=float, default=1.0)
    parser.add_argument("--post-evict-sleep", type=float, default=2.0)
    parser.add_argument("--wait-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vocab-limit", type=int, default=50000)
    parser.add_argument("--output-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()

# Adapted from benchmark/hicache/bench_serving.py and python/sglang/bench_serving.py

"""
Benchmark warm-cache serving with exact shared-prefix control.

This benchmark is designed for cache-focused studies where each request has a
fixed total input length and an exactly controlled shared-prefix ratio. For each
shared-prefix percentage, the benchmark:

1. Flushes the server KV cache via /flush_cache.
2. Builds prompts with an identical shared prefix and random unique suffixes.
3. Warms only the shared prefix once.
4. Benchmarks the full prompts via the selected backend endpoint.

Supported backends:
  sglang            - native /generate endpoint (token-id level)
  sglang-oai        - /v1/completions (OpenAI-compatible)
  sglang-oai-chat   - /v1/chat/completions (OpenAI-compatible)
  vllm / vllm-chat  - /v1/completions or /v1/chat/completions
  lmdeploy / lmdeploy-chat - same

Compared with the existing hicache shared-prefix benchmarks, this benchmark
provides direct control over total length, shared-prefix length, and suffix
length at the token-id level.
"""

import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.utils import get_tokenizer, remove_prefix, set_ulimit

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)
AIOHTTP_READ_BUFSIZE = 10 * 1024**2


global args


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    start_time: float = 0.0


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


def _create_bench_client_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=AIOHTTP_TIMEOUT,
        read_bufsize=AIOHTTP_READ_BUFSIZE,
    )


async def async_request_sglang_generate(
    api_url: str,
    input_ids: List[int],
    prompt_len: int,
    output_len: int,
    pbar: Optional[Any] = None,
) -> RequestFuncOutput:
    async with _create_bench_client_session() as session:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": output_len,
                "ignore_eos": not args.disable_ignore_eos,
            },
            "stream": True,
            **args.extra_request_body,
        }

        output = RequestFuncOutput(prompt_len=prompt_len)

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        last_output_len = 0
        latency = 0.0

        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)

                        if "text" in data and data["text"]:
                            timestamp = time.perf_counter()
                            generated_text = data["text"]
                            current_output_len = data["meta_info"]["completion_tokens"]

                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft
                            else:
                                num_new_tokens = current_output_len - last_output_len
                                if num_new_tokens == 0:
                                    continue
                                chunk_gap = timestamp - most_recent_timestamp
                                adjust_itl = chunk_gap / num_new_tokens
                                output.itl.extend([adjust_itl] * num_new_tokens)

                            most_recent_timestamp = timestamp
                            last_output_len = current_output_len
                            output.output_len = current_output_len

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception as exc:
            output.success = False
            output.error = str(exc)

    if pbar:
        pbar.update(1)
    return output


def _get_auth_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }


async def async_request_openai_completions(
    api_url: str,
    input_ids: List[int],
    prompt_len: int,
    output_len: int,
    pbar: Optional[Any] = None,
) -> RequestFuncOutput:
    async with _create_bench_client_session() as session:
        payload = {
            "model": args.model,
            "prompt": input_ids,
            "temperature": 0.0,
            "max_tokens": output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": not args.disable_ignore_eos,
            **args.extra_request_body,
        }
        headers = _get_auth_headers()
        output = RequestFuncOutput(prompt_len=prompt_len)

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        latency = 0.0

        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()

                        usage = data.get("usage") or {}
                        if usage:
                            output.output_len = usage.get(
                                "completion_tokens", output.output_len
                            )
                            continue

                        text = data["choices"][0].get("text", "")
                        if text:
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            generated_text += text
                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception as exc:
            output.success = False
            output.error = str(exc)

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    api_url: str,
    prompt_text: str,
    prompt_len: int,
    output_len: int,
    pbar: Optional[Any] = None,
) -> RequestFuncOutput:
    async with _create_bench_client_session() as session:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.0,
            "max_tokens": output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": not args.disable_ignore_eos,
            **args.extra_request_body,
        }
        headers = _get_auth_headers()
        output = RequestFuncOutput(prompt_len=prompt_len)

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        latency = 0.0

        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()

                        usage = data.get("usage") or {}
                        if usage:
                            output.output_len = usage.get(
                                "completion_tokens", output.output_len
                            )
                            continue

                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            generated_text += content
                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception as exc:
            output.success = False
            output.error = str(exc)

    if pbar:
        pbar.update(1)
    return output


BACKEND_TO_PORTS = {
    "sglang": 30000,
    "sglang-oai": 30000,
    "sglang-oai-chat": 30000,
    "vllm": 8000,
    "vllm-chat": 8000,
    "lmdeploy": 23333,
    "lmdeploy-chat": 23333,
}

COMPLETIONS_BACKENDS = {"sglang-oai", "vllm", "lmdeploy"}
CHAT_BACKENDS = {"sglang-oai-chat", "vllm-chat", "lmdeploy-chat"}


async def run_batch(
    api_url: str,
    prompts: List[Dict[str, Any]],
    output_len: int,
    max_concurrency: Optional[int],
    backend: str,
    pbar: Optional[Any] = None,
) -> List[RequestFuncOutput]:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    if backend in COMPLETIONS_BACKENDS:
        request_fn = async_request_openai_completions
    elif backend in CHAT_BACKENDS:
        request_fn = async_request_openai_chat_completions
    else:
        request_fn = async_request_sglang_generate

    async def limited_request(prompt: Dict[str, Any]) -> RequestFuncOutput:
        if backend in CHAT_BACKENDS:
            kw = {
                "api_url": api_url,
                "prompt_text": prompt["prompt_text"],
                "prompt_len": prompt["prompt_len"],
                "output_len": output_len,
                "pbar": pbar,
            }
        else:
            kw = {
                "api_url": api_url,
                "input_ids": prompt["input_ids"],
                "prompt_len": prompt["prompt_len"],
                "output_len": output_len,
                "pbar": pbar,
            }
        if semaphore is None:
            return await request_fn(**kw)
        async with semaphore:
            return await request_fn(**kw)

    tasks = [asyncio.create_task(limited_request(prompt)) for prompt in prompts]
    return await asyncio.gather(*tasks)


def flush_cache(base_url: str) -> None:
    response = requests.post(f"{base_url}/flush_cache", timeout=30)
    response.raise_for_status()


def gen_token_ids(
    vocab_ids: List[int],
    token_num: int,
    rng: random.Random,
) -> List[int]:
    if token_num <= 0:
        return []
    return rng.choices(vocab_ids, k=token_num)


def _gen_token_pool(
    token_count: int,
    tokenizer: PreTrainedTokenizerBase,
    rng: random.Random,
) -> List[int]:
    """Encode random ASCII into a pool of *token_count* stable token IDs.

    One ``tokenizer.encode`` call for the entire pool; callers slice from
    the result and ``tokenizer.decode`` each slice (which is cheap).
    """
    if token_count <= 0:
        return []
    chars = "abcdefghijklmnopqrstuvwxyz "
    raw = "".join(rng.choices(chars, k=token_count * 6))
    return tokenizer.encode(raw, add_special_tokens=False)[:token_count]


def build_prompts(
    vocab_ids: List[int],
    total_tokens: int,
    shared_pct: int,
    num_prompts: int,
    rng: random.Random,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    need_chat_text: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (prompts, shared_prefix_text).

    When *need_chat_text* is False (sglang / completions backends), prompts
    contain raw ``input_ids`` built from random token IDs.

    When *need_chat_text* is True (chat backends), prompts contain
    ``prompt_text`` built from random ASCII that tokenizes to the target
    length.  A single ``tokenizer.encode`` call builds the entire token
    pool; per-prompt work is decode-only.
    """
    prefix_len = total_tokens * shared_pct // 100
    suffix_len = total_tokens - prefix_len

    shared_prefix_text: Optional[str] = None

    if need_chat_text and tokenizer is not None:
        pool = _gen_token_pool(prefix_len + num_prompts * suffix_len, tokenizer, rng)
        prefix_ids = pool[:prefix_len]
        shared_prefix_text = tokenizer.decode(prefix_ids) if prefix_len > 0 else ""

        prompts: List[Dict[str, Any]] = []
        for i in range(num_prompts):
            offset = prefix_len + i * suffix_len
            suffix_ids = pool[offset : offset + suffix_len]
            input_ids = prefix_ids + suffix_ids
            suffix_text = tokenizer.decode(suffix_ids) if suffix_len > 0 else ""
            prompts.append(
                {
                    "input_ids": input_ids,
                    "prompt_text": shared_prefix_text + suffix_text,
                    "prompt_len": len(input_ids),
                }
            )
        return prompts, shared_prefix_text

    shared_prefix = gen_token_ids(vocab_ids, prefix_len, rng)
    prompts = []
    for _ in range(num_prompts):
        suffix = gen_token_ids(vocab_ids, suffix_len, rng)
        input_ids = shared_prefix + suffix
        prompts.append({"input_ids": input_ids, "prompt_len": len(input_ids)})

    return prompts, None


async def warm_shared_prefix(
    api_url: str,
    shared_prefix_ids: List[int],
    backend: str,
    shared_prefix_text: Optional[str] = None,
) -> None:
    if not shared_prefix_ids:
        return

    prompt_len = len(shared_prefix_ids)
    if backend in CHAT_BACKENDS:
        warmup = await async_request_openai_chat_completions(
            api_url=api_url,
            prompt_text=shared_prefix_text,
            prompt_len=prompt_len,
            output_len=1,
        )
    elif backend in COMPLETIONS_BACKENDS:
        warmup = await async_request_openai_completions(
            api_url=api_url,
            input_ids=shared_prefix_ids,
            prompt_len=prompt_len,
            output_len=1,
        )
    else:
        warmup = await async_request_sglang_generate(
            api_url=api_url,
            input_ids=shared_prefix_ids,
            prompt_len=prompt_len,
            output_len=1,
        )
    if not warmup.success:
        raise RuntimeError(
            "Warmup failed - Please make sure benchmark arguments are correctly "
            f"specified. Error: {warmup.error}"
        )


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []

    for output in outputs:
        if output.success:
            output_len = output.output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(output.generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += output.prompt_len
            if output_len > 1:
                tpots.append((output.latency - output.ttft) / (output_len - 1))
            itls += output.itl
            ttfts.append(output.ttft)
            e2e_latencies.append(output.latency)
            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p90_itl_ms=np.percentile(itls or 0, 90) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
    )
    return metrics, output_lens


def print_benchmark_result(
    metrics: BenchmarkMetrics,
    benchmark_duration: float,
    backend: str,
    request_rate: float,
    max_concurrency: Optional[int],
) -> None:
    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print(
        "{:<40} {:<10}".format(
            "Max request concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_throughput
        )
    )
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", metrics.p90_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", metrics.p90_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P90 ITL (ms):", metrics.p90_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)


def maybe_write_summary_jsonl(
    pct: int,
    prefix_len: int,
    suffix_len: int,
    metrics: BenchmarkMetrics,
    output_file: Optional[str],
    benchmark_duration: float,
) -> None:
    if not output_file:
        return

    result = {
        "backend": args.backend,
        "dataset_name": "warm-cache",
        "request_rate": float("inf"),
        "max_concurrency": args.max_concurrency,
        "shared_prefix_pct": pct,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "total_tokens": args.total_tokens,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "completed": metrics.completed,
        "benchmark_duration": benchmark_duration,
        "total_input": metrics.total_input,
        "total_output": metrics.total_output,
        "total_output_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "output_throughput_retokenized": metrics.output_throughput_retokenized,
        "total_throughput": metrics.total_throughput,
        "total_throughput_retokenized": metrics.total_throughput_retokenized,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p90_itl_ms": metrics.p90_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
        "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
        "concurrency": metrics.concurrency,
    }

    with open(output_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(result) + "\n")


async def benchmark_shared_prefix_pct(
    api_url: str,
    base_url: str,
    tokenizer: PreTrainedTokenizerBase,
    vocab_ids: List[int],
    rng: random.Random,
    pct: int,
) -> Tuple[BenchmarkMetrics, float, int, int, int]:
    prefix_len = args.total_tokens * pct // 100
    suffix_len = args.total_tokens - prefix_len

    print(f"\n{'=' * 70}")
    print(
        f"shared_prefix={pct}%  prefix_len={prefix_len}  "
        f"suffix_len={suffix_len}  total={prefix_len + suffix_len}"
    )
    print(f"{'=' * 70}")

    print("Flushing KV cache ...")
    flush_cache(base_url)
    time.sleep(1)

    need_chat_text = args.backend in CHAT_BACKENDS
    print(f"Building {args.num_prompts} prompts ...")
    prompts, shared_prefix_text = build_prompts(
        vocab_ids=vocab_ids,
        total_tokens=args.total_tokens,
        shared_pct=pct,
        num_prompts=args.num_prompts,
        rng=rng,
        tokenizer=tokenizer if need_chat_text else None,
        need_chat_text=need_chat_text,
    )

    if prefix_len > 0:
        print(f"Warming shared prefix only ({prefix_len} tokens) ...")
        await warm_shared_prefix(
            api_url=api_url,
            shared_prefix_ids=prompts[0]["input_ids"][:prefix_len],
            backend=args.backend,
            shared_prefix_text=shared_prefix_text,
        )

    print(f"Sending requests (max_concurrency={args.max_concurrency}) ...")
    benchmark_start_time = time.perf_counter()
    outputs = await run_batch(
        api_url=api_url,
        prompts=prompts,
        output_len=args.output_len,
        max_concurrency=args.max_concurrency,
        backend=args.backend,
        pbar=None,
    )
    benchmark_duration = time.perf_counter() - benchmark_start_time

    failed_outputs = [output for output in outputs if not output.success]
    if failed_outputs:
        print(f"WARNING: {len(failed_outputs)}/{len(outputs)} requests failed")
        for output in failed_outputs[:5]:
            print(f"  {output.error[:160]}")

    metrics, _ = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    if metrics.completed == 0:
        raise RuntimeError("All requests failed for this shared-prefix percentage.")

    print_benchmark_result(
        metrics=metrics,
        benchmark_duration=benchmark_duration,
        backend=args.backend,
        request_rate=float("inf"),
        max_concurrency=args.max_concurrency,
    )

    maybe_write_summary_jsonl(
        pct=pct,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        metrics=metrics,
        output_file=args.output_file,
        benchmark_duration=benchmark_duration,
    )

    return metrics, benchmark_duration, prefix_len, suffix_len, len(outputs)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        choices=list(BACKEND_TO_PORTS.keys()),
        help="Backend to benchmark. 'sglang' uses the native /generate endpoint; "
        "others use OpenAI-compatible /v1/completions or /v1/chat/completions.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base url if not using host and port.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Defaults to backend-specific port if not set.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the model. Used to load the tokenizer and vocab ids.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name or path of the tokenizer. Defaults to --model.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=64,
        help="Number of prompts to process per shared-prefix percentage.",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=70000,
        help="Total input tokens per request (shared prefix + unique suffix).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=200,
        help="Output length for each request.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--pcts",
        type=str,
        default="0,10,20,30,40,50,60,70,80,90,92,95,97,99",
        help="Comma-separated shared-prefix percentages to sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic prompt generation.",
    )
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional JSONL file to append one result object per shared-prefix percentage.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify additional generate params.",
    )
    global args
    args = parser.parse_args()

    args.extra_request_body = (
        json.loads(args.extra_request_body) if args.extra_request_body else {}
    )

    if args.port is None:
        args.port = BACKEND_TO_PORTS.get(args.backend, 30000)

    base_url = args.base_url or f"http://{args.host}:{args.port}"

    if args.backend in COMPLETIONS_BACKENDS:
        api_url = f"{base_url}/v1/completions"
    elif args.backend in CHAT_BACKENDS:
        api_url = f"{base_url}/v1/chat/completions"
    else:
        api_url = f"{base_url}/generate"
    pcts = [int(p.strip()) for p in args.pcts.split(",") if p.strip()]
    rng = random.Random(args.seed)

    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id)
    vocab_ids = list(tokenizer.get_vocab().values())

    print(f"{args}\n")
    print(f"Loading tokenizer from {tokenizer_id} ...")
    print(f"Tokenizer loaded (vocab_size={len(vocab_ids)})")

    for pct in pcts:
        await benchmark_shared_prefix_pct(
            api_url=api_url,
            base_url=base_url,
            tokenizer=tokenizer,
            vocab_ids=vocab_ids,
            rng=rng,
            pct=pct,
        )

    if args.output_file:
        print(f"JSONL results saved to {args.output_file}")


if __name__ == "__main__":
    set_ulimit()
    asyncio.run(main())

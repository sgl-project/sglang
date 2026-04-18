import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Tuple

import aiohttp
import numpy as np
from data_gen import gen_arguments
from transformers import AutoTokenizer

from sglang.bench_serving import (
    DatasetRow,
    RequestFuncOutput,
    calculate_metrics,
    remove_prefix,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang multi-turn chat with serving-style metrics."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--min-len-q", type=int, default=256)
    parser.add_argument("--max-len-q", type=int, default=512)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--result-file", type=str, default="result_serving.jsonl")
    parser.add_argument("--raw-result-file", type=str, default=None)
    return parser.parse_args()


async def async_request_sglang_generate(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    disable_ignore_eos: bool,
) -> Tuple[RequestFuncOutput, int]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": not disable_ignore_eos,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_logprob": False,
        "logprob_start_len": -1,
    }

    output = RequestFuncOutput()
    output.prompt_len = prompt_len
    generated_text = ""
    ttft = 0.0
    cached_tokens = 0
    prompt_tokens = prompt_len
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    last_output_len = 0

    try:
        async with session.post(url=url, json=payload) as response:
            if response.status != 200:
                output.success = False
                output.error = response.reason or ""
                return output, cached_tokens

            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                latency = time.perf_counter() - st
                if chunk == "[DONE]":
                    continue

                data = json.loads(chunk)
                text = data.get("text", "")
                if not text:
                    continue

                timestamp = time.perf_counter()
                generated_text = text
                output.output_len = (data.get("meta_info") or {}).get(
                    "completion_tokens", output_len
                )

                if ttft == 0.0:
                    ttft = timestamp - st
                    output.ttft = ttft
                    prompt_tokens = (data.get("meta_info") or {}).get(
                        "prompt_tokens", prompt_len
                    )
                    cached_tokens = (data.get("meta_info") or {}).get(
                        "cached_tokens", 0
                    )
                else:
                    num_new_tokens = output.output_len - last_output_len
                    if num_new_tokens > 0:
                        chunk_gap = timestamp - most_recent_timestamp
                        output.itl.extend([chunk_gap / num_new_tokens] * num_new_tokens)

                most_recent_timestamp = timestamp
                last_output_len = output.output_len

            output.generated_text = generated_text
            output.success = True
            output.latency = latency
            output.prompt_len = prompt_tokens
            return output, cached_tokens
    except Exception as e:
        output.success = False
        output.error = str(e)
        return output, cached_tokens


async def run_one_conversation(
    session: aiohttp.ClientSession,
    url: str,
    conv_id: int,
    qas: List[Dict[str, int]],
    tokenizer,
    disable_ignore_eos: bool,
):
    history = ""
    outputs: List[RequestFuncOutput] = []
    input_requests: List[DatasetRow] = []
    cached_tokens_per_turn: List[int] = []
    round_metrics: List[Dict[str, float]] = []

    for turn_idx, qa in enumerate(qas):
        history += qa["prompt"]
        prompt_len = len(tokenizer(history).input_ids)
        input_requests.append(
            DatasetRow(
                prompt=history,
                prompt_len=prompt_len,
                output_len=qa["new_tokens"],
            )
        )
        output, cached_tokens = await async_request_sglang_generate(
            session=session,
            url=url,
            prompt=history,
            prompt_len=prompt_len,
            output_len=qa["new_tokens"],
            disable_ignore_eos=disable_ignore_eos,
        )
        outputs.append(output)
        cached_tokens_per_turn.append(cached_tokens)
        round_metrics.append(
            {
                "conversation_id": conv_id,
                "turn": turn_idx,
                "ttft": output.ttft,
                "latency": output.latency,
                "output_len": output.output_len,
                "prompt_len": output.prompt_len,
                "cached_tokens": cached_tokens,
                "success": output.success,
            }
        )
        if not output.success:
            break
        history += output.generated_text

    return input_requests, outputs, cached_tokens_per_turn, round_metrics


async def run_all(args, multi_qas, tokenizer):
    url = f"http://{args.host}:{args.port}/generate"
    semaphore = asyncio.Semaphore(args.parallel)

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async def run_with_limit(conv_id, qas):
            async with semaphore:
                return await run_one_conversation(
                    session=session,
                    url=url,
                    conv_id=conv_id,
                    qas=qas,
                    tokenizer=tokenizer,
                    disable_ignore_eos=args.disable_ignore_eos,
                )

        tasks = [
            asyncio.create_task(run_with_limit(i, item["qas"]))
            for i, item in enumerate(multi_qas)
        ]
        benchmark_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - benchmark_start

    input_requests: List[DatasetRow] = []
    outputs: List[RequestFuncOutput] = []
    cached_tokens_per_turn: List[int] = []
    raw_round_metrics: List[Dict[str, float]] = []
    for reqs, conv_outputs, conv_cached_tokens, conv_round_metrics in results:
        input_requests.extend(reqs)
        outputs.extend(conv_outputs)
        cached_tokens_per_turn.extend(conv_cached_tokens)
        raw_round_metrics.extend(conv_round_metrics)

    return input_requests, outputs, cached_tokens_per_turn, raw_round_metrics, duration


def summarize_rounds(raw_round_metrics: List[Dict[str, float]]):
    grouped = defaultdict(list)
    for item in raw_round_metrics:
        if item["success"]:
            grouped[item["turn"]].append(item)

    round_summary = {}
    for turn, items in sorted(grouped.items()):
        prompt_sum = sum(item["prompt_len"] for item in items)
        cached_sum = sum(item["cached_tokens"] for item in items)
        round_summary[f"turn_{turn}"] = {
            "requests": len(items),
            "mean_ttft_ms": float(np.mean([item["ttft"] for item in items]) * 1000),
            "mean_e2e_latency_ms": float(
                np.mean([item["latency"] for item in items]) * 1000
            ),
            "cache_hit_rate": 0.0 if prompt_sum == 0 else cached_sum / prompt_sum,
        }
    return round_summary


def print_metrics(metrics, cache_hit_rate, round_summary):
    print("\n{:=^50}".format(" Multi-turn Serving Benchmark Result "))
    print("{:<40} {:<10.2f}".format("Request Throughput (req/s)", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input Token Throughput (tok/s)", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output Token Throughput (tok/s)", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token Throughput (tok/s)", metrics.total_throughput))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms)", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms)", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms)", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Concurrency", metrics.concurrency))
    print("{:<40} {:<10.6f}".format("Cache Hit Rate", cache_hit_rate))
    print("=" * 50)

    if round_summary:
        print("Per-turn summary:")
        for turn_key, item in round_summary.items():
            print(
                f"  {turn_key}: requests={item['requests']}, "
                f"mean_ttft_ms={item['mean_ttft_ms']:.2f}, "
                f"mean_e2e_latency_ms={item['mean_e2e_latency_ms']:.2f}, "
                f"cache_hit_rate={item['cache_hit_rate']:.6f}"
            )


def main():
    args = parse_args()
    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    multi_qas = gen_arguments(args, tokenizer)

    (
        input_requests,
        outputs,
        cached_tokens_per_turn,
        raw_round_metrics,
        duration,
    ) = asyncio.run(run_all(args, multi_qas, tokenizer))

    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=duration,
        tokenizer=tokenizer,
        backend="sglang",
    )

    total_prompt_tokens = sum(
        output.prompt_len for output in outputs if output.success and output.prompt_len > 0
    )
    cache_hit_rate = (
        0.0 if total_prompt_tokens == 0 else sum(cached_tokens_per_turn) / total_prompt_tokens
    )
    round_summary = summarize_rounds(raw_round_metrics)

    print_metrics(metrics, cache_hit_rate, round_summary)

    result = {
        "task": "multi_turn_chat_serving",
        "host": args.host,
        "port": args.port,
        "num_requests": args.num_qa,
        "num_turns": args.turns,
        "parallel": args.parallel,
        "duration": duration,
        "cache_hit_rate": cache_hit_rate,
        "metrics": asdict(metrics),
        "round_summary": round_summary,
        "details": {
            "input_lens": [request.prompt_len for request in input_requests],
            "output_lens": output_lens,
            "ttfts": [output.ttft for output in outputs],
            "latencies": [output.latency for output in outputs],
            "itls": [output.itl for output in outputs],
            "cached_tokens": cached_tokens_per_turn,
            "errors": [output.error for output in outputs],
        },
    }

    with open(args.result_file, "a") as fout:
        fout.write(json.dumps(result) + "\n")

    if args.raw_result_file:
        with open(args.raw_result_file, "w") as fout:
            json.dump(result, fout, indent=2)


if __name__ == "__main__":
    main()

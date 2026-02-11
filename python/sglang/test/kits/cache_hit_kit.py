import asyncio
import json
import time

import aiohttp
import requests

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


async def async_request_sglang_generate(
    payload,
    url,
    pbar=None,
):
    """Send a streaming request to the server and collect cache metrics.

    Returns a RequestFuncOutput with additional cached_tokens and output_ids attributes.
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        generated_text = ""
        all_output_ids = []
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0

                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st

                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # output_ids and text are always returned together
                            if data.get("output_ids"):
                                all_output_ids = data["output_ids"]
                                generated_text = data.get("text", "")
                                timestamp = time.perf_counter()

                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    prompt_tokens = (data.get("meta_info") or {}).get(
                                        "prompt_tokens", 0
                                    )
                                    cached_tokens = (data.get("meta_info") or {}).get(
                                        "cached_tokens", 0
                                    )
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.output_ids = all_output_ids
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                    output.generated_len = len(output.itl) + 1
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    if pbar:
        pbar.update(1)
    return output


def gen_payload(input_ids, output_len, lora_path=""):
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "lora_path": lora_path,
        "return_logprob": False,
        "logprob_start_len": -1,
    }


async def _send_round(
    payloads,
    url,
    max_parallel,
):
    """Send a batch of payloads concurrently with concurrency limit."""
    semaphore = asyncio.Semaphore(max_parallel)

    async def _send_one(payload):
        async with semaphore:
            return await async_request_sglang_generate(payload, url)

    tasks = [asyncio.create_task(_send_one(p)) for p in payloads]
    return await asyncio.gather(*tasks)


def _get_page_size(base_url: str) -> int:
    """Query server for page_size used by radix cache."""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        return info.get("page_size", 1)
    except Exception:
        return 1


def run_multiturn_cache_hit_test(
    base_url: str,
    model_path: str,
    num_clients: int = 8,
    num_rounds: int = 3,
    request_length: int = 256,
    output_length: int = 32,
    miss_tolerance: int = 1,
    sub_question_input_length: int = 0,
    lora_path: str = "",
    dataset_path: str = "",
    max_parallel: int = 64,
    seed: int = 1,
) -> dict:
    """Run a multi-turn workload and verify cache hit rate.

    Sends requests in round-barrier mode: all clients complete round i
    before round i+1 starts, ensuring deterministic cache state.

    The expected cache hit rate is self-computed from the workload structure:
    - Round 0: expected cached_tokens = 0 (cold start after flush)
    - Round r (r >= 1): each client's prefix from round r-1 should be cached,
      minus up to previous round's (prompt_len + decoding output - miss_tolerance) // page * page.

    Returns metrics dict with per-round and overall cache_hit_rate.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    generate_url = f"{base_url}/generate"
    page_size = _get_page_size(base_url)

    # Flush cache for clean state
    requests.post(f"{base_url}/flush_cache")
    time.sleep(1)

    # Resolve sub-question length (0 means same as request_length)
    effective_sub_len = (
        sub_question_input_length if sub_question_input_length != 0 else request_length
    )

    # Sample initial prompts and sub-question prompts as token ids
    tokenizer = get_tokenizer(model_path)

    initial_inputs = sample_random_requests(
        input_len=request_length,
        output_len=output_length,
        num_prompts=num_clients,
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        return_text=False,
    )
    # r.prompt is now List[int] when return_text=False
    initial_token_ids = [list(r.prompt) for r in initial_inputs]

    sub_question_inputs = sample_random_requests(
        input_len=effective_sub_len,
        output_len=output_length,
        num_prompts=num_clients * max(num_rounds - 1, 1),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        return_text=False,
    )
    sub_question_token_ids = [list(r.prompt) for r in sub_question_inputs]

    # Per-round metrics and per-client tracking for expected cache computation
    round_metrics = {
        i: {"prompt_len": [], "cached_tokens": [], "ttft": []}
        for i in range(num_rounds)
    }
    # Track the previous round's prompt_len per client to compute expected cache
    prev_prompt_lens = [0] * num_clients
    # histories now stores List[int] (token ids) for each client
    histories = [list(ids) for ids in initial_token_ids]
    sub_idx = 0

    for round_num in range(num_rounds):
        payloads = [gen_payload(h, output_length, lora_path) for h in histories]
        responses = asyncio.run(_send_round(payloads, generate_url, max_parallel))

        for i, resp in enumerate(responses):
            assert resp.success, f"Round {round_num}, client {i} failed: {resp.error}"

            round_metrics[round_num]["prompt_len"].append(resp.prompt_len)
            round_metrics[round_num]["cached_tokens"].append(resp.cached_tokens)
            round_metrics[round_num]["ttft"].append(resp.ttft)

            # Verify cache hit against expected value
            if round_num == 0:
                # Cold start: no cache expected
                expected_cached = 0
            else:
                # Previous round's prompt + output are in cache.
                # Radix cache aligns to page_size, so the last partial page
                # may not be cached.
                cacheable = prev_prompt_lens[i] + output_length - miss_tolerance
                expected_cached = (cacheable // page_size) * page_size

            msg = (
                f"Round {round_num}, client {i}: "
                f"cached_tokens={resp.cached_tokens}, "
                f"expected>={expected_cached} "
                f"(prev_prompt={prev_prompt_lens[i]}, "
                f"output={output_length}, page_size={page_size})"
            )

            print(msg)

            assert resp.cached_tokens >= expected_cached

            # Record this round's prompt_len for next round's expected calc
            prev_prompt_lens[i] = resp.prompt_len

            # Accumulate history for next round using output_ids (token ids)
            histories[i].extend(resp.output_ids)
            if round_num < num_rounds - 1:
                histories[i].extend(sub_question_token_ids[sub_idx])
                sub_idx += 1

    # Compute per-round and overall cache hit rate
    total_prompt = 0
    total_cached = 0
    result = {"rounds": {}, "overall": {}}

    for r in range(num_rounds):
        rm = round_metrics[r]
        r_prompt = sum(rm["prompt_len"])
        r_cached = sum(rm["cached_tokens"])
        r_hit_rate = r_cached / r_prompt if r_prompt > 0 else 0.0
        r_avg_ttft = sum(rm["ttft"]) / len(rm["ttft"]) if rm["ttft"] else 0.0

        result["rounds"][f"round_{r}"] = {
            "cache_hit_rate": r_hit_rate,
            "average_ttft": r_avg_ttft,
            "total_prompt_tokens": r_prompt,
            "total_cached_tokens": r_cached,
            "request_count": len(rm["ttft"]),
        }

        total_prompt += r_prompt
        total_cached += r_cached

        print(
            f"  Round {r}: cache_hit_rate={r_hit_rate:.4f}, "
            f"avg_ttft={r_avg_ttft:.4f}s, "
            f"cached={r_cached}/{r_prompt} tokens"
        )

    overall_hit_rate = total_cached / total_prompt if total_prompt > 0 else 0.0
    result["overall"] = {
        "cache_hit_rate": overall_hit_rate,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
    }
    print(f"  Overall cache_hit_rate={overall_hit_rate:.4f}")

    return result

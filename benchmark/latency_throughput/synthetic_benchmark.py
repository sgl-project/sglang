"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (SRT backend)
    python -m sglang.launch_server \
        --model <your_model> --tp <num_gpus> \
        --port 30000 --enable-flashinfer --disable-radix-cache

    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --tensor <num_gpus> --swap-space 16 \
        --disable-log-requests --port 30000

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> \
        --num-prompt <num_prompts> \
        --request-rate <request_rate>
        --input-len <input_len> \
        --output-len <output_len> \
        --port 30000
"""

import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizerBase

from sglang.srt.hf_transformers_utils import get_tokenizer


def sample_requests(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_len: int,
    output_len: int,
) -> List[Tuple[str, int, int]]:
    prompt = "Hello " * input_len
    prompt_token_ids = list(tokenizer(prompt).input_ids)
    requests = []
    for i in range(num_requests):
        requests.append((prompt, len(prompt_token_ids), output_len))
    
    return requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    elif backend == "srt":
        assert not use_beam_search
        params = {
            "ignore_eos": True,
            "max_new_tokens": output_len,
        }
        pload = {
            "text": prompt,
            "sampling_params": params,
        }
    elif backend == "lightllm":
        assert not use_beam_search
        params = {
            "ignore_eos": True,
            "max_new_tokens": output_len,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    request_start_time = time.perf_counter()
    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - request_start_time
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_latency = time.perf_counter() - request_start_time
    return (prompt_len, output_len, request_latency, first_token_latency)


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                prompt,
                prompt_len,
                output_len,
                best_of,
                use_beam_search,
            )
        )
        tasks.append(task)
    request_latency = await tqdm_asyncio.gather(*tasks)
    return request_latency


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.num_prompts, tokenizer, args.input_len, args.output_len)

    benchmark_start_time = time.perf_counter()
    # (prompt len, output len, latency, first_token_latency)
    request_latency = asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
        )
    )
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")

    # Compute the perf statistics.
    throughput = np.sum([output_len for _, output_len, _, _ in request_latency]) / benchmark_time
    print(f"Throughput: {throughput} token/s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency, _ in request_latency
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [(latency - first_token_latency) / output_len for _, output_len, latency, first_token_latency in request_latency]
    )
    print(f"Average TPOT: {avg_per_output_token_latency * 1000:.0f} ms")
    avg_first_token_latency = np.mean(
        [first_token_latency for _, _, _, first_token_latency in request_latency]
    )
    print(f"Average TTFT: {avg_first_token_latency:.2f} s")

    stats = {"num_prompts": args.num_prompts, "input_len": args.input_len, "output_len": args.output_len,
            "total_time (s)": benchmark_time, "throughput (token/s)": throughput, "avg_per_token_latency (s)": avg_per_token_latency,
            "TPOT (ms)": avg_per_output_token_latency, "TTFT (s)": avg_first_token_latency}

    with open(args.output_file, "a") as f:
        f.write(json.dumps(stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "tgi", "srt", "lightllm"],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=512,
        help="Number of input tokens"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of output tokens"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="perf_stats.jsonl",
        help="output file path for performance statistics"
    )
    args = parser.parse_args()
    main(args)

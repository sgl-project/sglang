"""Compare fixed-work DiffusionGemma serving through the completions API.

Both servers must use the same checkpoint, canvas length, denoising limit and
stopping settings. Set stability_threshold above max_denoising_steps on both
servers so early convergence cannot shorten work. Run them sequentially on the
same physical GPUs, using the same run ID and arguments. Each request has a
unique prompt prefix to avoid prefix-cache reuse.

This measures complete canvas computation with EOS stopping disabled. Canvas
tokens per second include tokens after EOS and are not useful-text throughput.
Validate output quality separately with the model's default generation settings.
The denoising-steps argument records the server configuration; it does not change it.
"""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


async def benchmark(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    body = tokenizer.encode(
        "The city library has books about science, history, music and art. " * 256,
        add_special_tokens=False,
    )
    template = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Request __REQUEST_ID__.\n__BENCHMARK_BODY__\n"
                    "Write a detailed explanation of why public libraries are useful."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_template, suffix_text = template.split("__BENCHMARK_BODY__")
    suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
    results = []
    request_number = 0
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(args.url + "/v1/models") as response:
            response.raise_for_status()
            model = (await response.json())["data"][0]["id"]

        async def request(prompt):
            start = time.perf_counter()
            async with session.post(
                args.url + "/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": args.output_length,
                    "ignore_eos": True,
                    "stream": False,
                },
            ) as response:
                payload = await response.json()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {payload}")
            elapsed = time.perf_counter() - start
            usage = payload["usage"]
            if usage["completion_tokens"] != args.output_length:
                raise RuntimeError(f"Unexpected output length: {usage}")
            if usage["prompt_tokens"] != len(prompt):
                raise RuntimeError(f"Unexpected prompt length: {usage}")
            return {
                "latency_s": elapsed,
                "usage": usage,
                "text": payload["choices"][0]["text"],
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompt).encode()
                ).hexdigest(),
            }

        for input_length in args.input_lengths:
            for concurrency in args.concurrencies:
                trials = []
                for trial in range(args.warmups + args.trials):
                    prompts = []
                    for _ in range(concurrency):
                        # Vary the beginning of every prompt to avoid measuring
                        # cached prompts. Keep run_id identical across runtimes.
                        prefix = tokenizer.encode(
                            prefix_template.replace(
                                "__REQUEST_ID__", f"{args.run_id}-{request_number}"
                            ),
                            add_special_tokens=False,
                        )
                        request_number += 1
                        n = input_length - len(prefix) - len(suffix)
                        if n < 0:
                            raise ValueError("Input length is too short")
                        prompts.append(prefix + body[:n] + suffix)
                    start = time.perf_counter()
                    requests = await asyncio.gather(*(request(p) for p in prompts))
                    trials.append(
                        {
                            "warmup": trial < args.warmups,
                            "wall_s": time.perf_counter() - start,
                            "requests": requests,
                        }
                    )
                measured = trials[args.warmups :]
                wall_times = [t["wall_s"] for t in measured]
                latencies = [r["latency_s"] for t in measured for r in t["requests"]]
                result = {
                    "input_length": input_length,
                    "output_length": args.output_length,
                    "concurrency": concurrency,
                    "median_wall_s": statistics.median(wall_times),
                    "median_request_s": statistics.median(latencies),
                    "canvas_tokens_per_s": concurrency
                    * args.output_length
                    / statistics.median(wall_times),
                    "trials": trials,
                }
                results.append(result)
                print(
                    json.dumps({k: v for k, v in result.items() if k != "trials"}),
                    flush=True,
                )
                args.result.write_text(
                    json.dumps(
                        {
                            "settings": vars(args) | {"result": str(args.result)},
                            "results": results,
                        },
                        indent=2,
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--output-length", type=int, default=256)
    parser.add_argument("--denoising-steps", type=int, default=48)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    asyncio.run(benchmark(parser.parse_args()))

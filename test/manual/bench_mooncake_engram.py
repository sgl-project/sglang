"""Measure warm Engine streaming latency and throughput for an Engram backend.

Run each backend in its own process, sequentially on the same GPUs. Pass matching
Engine arguments through --engine-args. This measures the Engine API (including
IPC/scheduling/detokenization), without HTTP or input tokenization time.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def memory_snapshot():
    import psutil

    processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
    pss = 0
    for process in processes:
        try:
            pss += process.memory_full_info().pss
        except psutil.NoSuchProcess:
            pass
    return {
        "process_tree_pss_bytes": pss,
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip(),
    }


async def wave(engine, prompts, output_length, batch_submit=False):
    sampling = {
        "temperature": 0,
        "max_new_tokens": output_length,
        "ignore_eos": True,
    }

    def summarize(started, first, first_count, finished, last):
        assert first is not None and last is not None
        count = last["meta_info"]["completion_tokens"]
        assert count == output_length, last["meta_info"]
        assert last["meta_info"].get("cached_tokens", 0) == 0, last["meta_info"]
        assert count > first_count, "Need multiple streamed chunks to measure TPOT"
        return {
            "ttft_ms": (first - started) * 1000,
            "latency_ms": (finished - started) * 1000,
            "tpot_ms": (finished - first) * 1000 / (count - first_count),
            "first_chunk_tokens": first_count,
            "output_tokens": count,
            "output_ids": last.get("output_ids"),
        }

    async def request(ids):
        started = time.perf_counter()
        stream = await engine.async_generate(
            input_ids=ids,
            sampling_params=sampling,
            stream=True,
        )
        first = None
        first_count = None
        last = None
        async for chunk in stream:
            now = time.perf_counter()
            count = chunk["meta_info"]["completion_tokens"]
            if count and first is None:
                first, first_count = now, count
            last = chunk
        finished = time.perf_counter()
        return summarize(started, first, first_count, finished, last)

    start = time.perf_counter()
    if batch_submit:
        states = [[None, None, None, None] for _ in prompts]
        stream = await engine.async_generate(
            input_ids=prompts, sampling_params=sampling, stream=True
        )
        async for chunk in stream:
            now = time.perf_counter()
            state = states[chunk["index"]]
            count = chunk["meta_info"]["completion_tokens"]
            if count and state[0] is None:
                state[0], state[1] = now, count
            state[2], state[3] = now, chunk
        requests = [summarize(start, *state) for state in states]
    else:
        requests = await asyncio.gather(*(request(ids) for ids in prompts))
    duration = time.perf_counter() - start
    return {
        "input_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "seconds": duration,
        "output_tokens_per_second": sum(r["output_tokens"] for r in requests)
        / duration,
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=["native", "mooncake"], required=True)
    parser.add_argument(
        "--engine-args", required=True, help="Engine arguments JSON file"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--batch-submit",
        action="store_true",
        help="Submit each wave together using SGLang's colocated batch guard",
    )
    args = parser.parse_args()
    assert args.output_length > 1 and args.warmups >= 1 and args.repeats >= 3
    if args.batch_submit:
        os.environ["SGLANG_ENABLE_COLOCATED_BATCH_GEN"] = "1"

    from transformers import AutoTokenizer

    import sglang as sgl

    kwargs = json.loads(Path(args.engine_args).read_text())
    kwargs.update(model_path=args.model_path, engram_backend=args.backend)
    assert kwargs["disable_radix_cache"], "Prefix reuse invalidates cold-prefill timing"
    assert kwargs["stream_interval"] == 1
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    special = set(tokenizer.all_special_ids)
    token_pool = [
        i for i in range(256, min(32000, tokenizer.vocab_size)) if i not in special
    ]
    assert token_pool

    def prompts_for(length, concurrency, iteration):
        # Different rows each wave; identical inputs for the two backends.
        rng = random.Random(42 + length * 100000 + concurrency * 1000 + iteration)
        return [rng.choices(token_pool, k=length) for _ in range(concurrency)]

    cases = [
        (length, concurrency)
        for length in args.input_lengths
        for concurrency in args.concurrencies
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "engine_args": kwargs,
        "input_distribution": "uniform text token IDs in [256, min(32000, vocab_size)), excluding special IDs; different deterministic seed per wave",
        "benchmark_args": vars(args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_table": os.environ.get("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"),
        "host_table_layout": os.environ.get("SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT"),
        "cases": [],
        "complete": False,
    }
    started = time.perf_counter()
    engine = sgl.Engine(**kwargs)
    result["load_seconds"] = time.perf_counter() - started
    try:
        for length, concurrency in cases:
            for iteration in range(args.warmups):
                prompts = prompts_for(length, concurrency, iteration)
                engine.loop.run_until_complete(
                    wave(engine, prompts, args.output_length, args.batch_submit)
                )
            before = memory_snapshot()
            samples = [
                engine.loop.run_until_complete(
                    wave(
                        engine,
                        prompts_for(length, concurrency, args.warmups + iteration),
                        args.output_length,
                        args.batch_submit,
                    )
                )
                for iteration in range(args.repeats)
            ]
            summary = {
                "input_length": length,
                "output_length": args.output_length,
                "concurrency": concurrency,
                "median_output_tokens_per_second": statistics.median(
                    s["output_tokens_per_second"] for s in samples
                ),
                **{
                    "median_" + metric: statistics.median(
                        r[metric] for s in samples for r in s["requests"]
                    )
                    for metric in ("ttft_ms", "tpot_ms", "latency_ms")
                },
            }
            result["cases"].append(
                dict(
                    summary,
                    samples=samples,
                    memory_before=before,
                    memory_after=memory_snapshot(),
                )
            )
            output.write_text(json.dumps(result, indent=2) + "\n")
            print("PERF_CASE", json.dumps(summary), flush=True)
        result["complete"] = True
        output.write_text(json.dumps(result, indent=2) + "\n")
        print("PERF_COMPLETE", output, flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

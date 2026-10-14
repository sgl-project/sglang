#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

from common import require_single_visible_gpu


def load_inputs(path: Path, count: int) -> list[list[int]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append([int(token) for token in json.loads(line)["input_ids"]])
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"Expected {count} input rows, found {len(rows)}")
    return rows


def run(args: argparse.Namespace) -> None:
    from sglang import Engine

    graph_buckets = [args.beams, args.beams * args.concurrency]
    graph_kwargs = (
        {"disable_cuda_graph": True}
        if args.disable_cuda_graph
        else {
            "cuda_graph_max_bs_decode": max(graph_buckets),
            "cuda_graph_bs_decode": graph_buckets,
            "disable_prefill_cuda_graph": True,
        }
    )
    engine = Engine(
        model_path=str(args.model),
        tokenizer_path=str(args.model),
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        context_length=args.context_length,
        tp_size=1,
        dp_size=1,
        attention_backend="flashinfer",
        schedule_policy="fcfs",
        random_seed=0,
        **graph_kwargs,
    )
    rows = load_inputs(args.inputs, args.requests + args.warmup)
    sampling_params = {
        "beam_width": args.beams,
        "max_new_tokens": args.sid_length,
        "n": args.beams,
    }

    async def generate(input_ids: list[int]) -> None:
        await engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
        )

    async def execute(input_rows: list[list[int]]) -> None:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def limited(input_ids: list[int]) -> None:
            async with semaphore:
                await generate(input_ids)

        await asyncio.gather(*(limited(input_ids) for input_ids in input_rows))

    engine.loop.run_until_complete(execute(rows[: args.warmup]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine.start_profile(
        output_dir=str(args.output_dir),
        activities=["CPU", "GPU"],
        record_shapes=True,
        profile_prefix=args.profile_prefix,
    )
    started = time.perf_counter()
    engine.loop.run_until_complete(execute(rows[args.warmup :]))
    elapsed = time.perf_counter() - started
    engine.stop_profile()
    engine.shutdown()
    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "wall_seconds": elapsed,
                "completion_qps": args.requests / elapsed,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture an SGLang CPU/GPU profile for trie-head decoding."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-prefix", default="trie-head")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--beams", type=int, default=2000)
    parser.add_argument("--sid-length", type=int, default=3)
    parser.add_argument("--max-running-requests", type=int, default=20020)
    parser.add_argument("--context-length", type=int, default=10000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.55)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)
    run(args)


if __name__ == "__main__":
    main()

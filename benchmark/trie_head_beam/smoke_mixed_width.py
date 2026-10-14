#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import require_single_visible_gpu

from sglang import Engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple trie-head beam widths on one graph-enabled engine."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-ids", type=int, nargs="+", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--sid-length", type=int, default=3)
    parser.add_argument("--max-running-requests", type=int, default=4000)
    parser.add_argument("--context-length", type=int, default=10000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.55)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)

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
    )

    async def run() -> dict[str, dict[str, float | int]]:
        results = {}
        for width in args.widths:
            started = time.perf_counter()
            raw = await engine.async_generate(
                input_ids=args.input_ids,
                sampling_params={
                    "beam_width": width,
                    "max_new_tokens": args.sid_length,
                    "n": width,
                },
            )
            beams = raw.get("meta_info", {}).get("beam_results", [])
            if len(beams) != width:
                raise RuntimeError(f"Beam width {width} returned {len(beams)} results.")
            results[str(width)] = {
                "results": len(beams),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        return results

    try:
        print(json.dumps(engine.loop.run_until_complete(run()), indent=2), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

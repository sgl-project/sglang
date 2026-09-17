"""Controlled TTFT comparison for the KVCR direct linker.

Runs the same model, topology, prompt set, and request pattern under three
cache configurations on the same GPU, sequentially:

  none     no external cache (device radix cache only)
  hicache  HiCache host tier (write_through), no storage backend
  kvcr     KVCR direct linker (no SGLang host pool)

Two workloads per configuration:

  zero_hit   unique prompts; measures the overhead a backend adds when
             nothing can be reused (must not wait on any peer timeout).
  hit        prompts served once, evicted from the GPU by filler traffic, then
             replayed; measures restore-path TTFT and served cached tokens.

Every prompt is greedy with one output token so e2e latency is TTFT. Each
workload is repeated ``--repeats`` times; the report keeps every sample so the
spread is visible.

Example:
  python kvcr_bench.py --model Qwen/Qwen3-0.6B --gpus 0 --workdir /tmp/kb \
      --report bench_tp1.json --dram-gib 8 --hicache-ratio 2
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from kvcr_validate import Server, _linker_config, _prompt


def _ttft(server: Server, prompt: list[int]) -> tuple[float, int]:
    started = time.monotonic()
    result = server.generate(prompt, 1)
    wall = time.monotonic() - started
    meta = result["meta_info"]
    return (meta.get("e2e_latency") or wall), int(meta.get("cached_tokens") or 0)


def _run_config(label: str, args, workdir: Path) -> dict:
    extra = list(args.extra)
    linker = None
    if label == "hicache":
        extra += [
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            str(args.hicache_ratio),
            "--hicache-write-policy",
            "write_through",
        ]
    elif label == "kvcr":
        linker = _linker_config(args, control_port=None)
    server = Server(
        name=f"bench_{label}",
        model=args.model,
        port=args.port,
        gpus=args.gpus,
        tp=args.tp,
        workdir=workdir,
        page_size=args.page_size,
        max_total_tokens=args.max_total_tokens,
        linker_config=linker,
        extra_args=extra,
        dp_rank=args.dp_rank,
    )
    rng = random.Random(args.seed)
    prompts = [
        _prompt(rng, args.prompt_tokens, args.vocab) for _ in range(args.prompts)
    ]
    fillers = [
        _prompt(rng, args.filler_tokens, args.vocab)
        for _ in range(args.max_total_tokens // args.filler_tokens + 4)
    ]
    unique = [
        [_prompt(rng, args.prompt_tokens, args.vocab) for _ in range(args.prompts)]
        for _ in range(args.repeats)
    ]
    samples: dict[str, list] = {"zero_hit": [], "hit": [], "hit_cached_tokens": []}
    try:
        server.wait_ready()
        # Warm the kernel paths once so the first sample is not a compile.
        _ttft(server, _prompt(rng, args.prompt_tokens, args.vocab))
        for repeat in range(args.repeats):
            for prompt in unique[repeat]:
                latency, _ = _ttft(server, prompt)
                samples["zero_hit"].append(latency)
        for repeat in range(args.repeats):
            for prompt in prompts:
                _ttft(server, prompt)
            time.sleep(args.settle_s)
            for filler in fillers:
                server.generate(filler, 1)
            time.sleep(args.settle_s)
            for prompt in prompts:
                latency, cached = _ttft(server, prompt)
                samples["hit"].append(latency)
                samples["hit_cached_tokens"].append(cached)
        time.sleep(args.settle_s)
        stats = server.stats()
        host_pool_lines = server.log_matches(
            r"HiCache|host memory|KV cache is being|hicache"
        )
    finally:
        server.stop()

    def summary(values: list[float]) -> dict:
        if not values:
            return {}
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "label": label,
        "command": server.command,
        "zero_hit_ttft_s": summary(samples["zero_hit"]),
        "hit_ttft_s": summary(samples["hit"]),
        "hit_cached_tokens": summary(samples["hit_cached_tokens"]),
        "samples": samples,
        "stats": stats,
        "host_pool_lines": host_pool_lines[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp-rank", type=int, default=None)
    parser.add_argument("--port", type=int, default=30200)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=16384)
    parser.add_argument("--prompt-tokens", type=int, default=3000)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--filler-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--dram-gib", type=int, default=8)
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--deadline-ms", type=int, default=5000)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--linker-config-json", default=None, help="JSON merged into the linker config"
    )
    parser.add_argument("--configs", default="none,hicache,kvcr")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--report", required=True)
    args, args.extra = parser.parse_known_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "tp": args.tp,
        "page_size": args.page_size,
        "max_total_tokens": args.max_total_tokens,
        "prompt_tokens": args.prompt_tokens,
        "prompts": args.prompts,
        "repeats": args.repeats,
        "configs": {},
    }
    for label in args.configs.split(","):
        report["configs"][label] = _run_config(label, args, workdir)
        Path(args.report).write_text(json.dumps(report, indent=2))
    for label, result in report["configs"].items():
        print(
            f"{label:8s} zero_hit TTFT median={result['zero_hit_ttft_s'].get('median', 0):.4f}s "
            f"hit TTFT median={result['hit_ttft_s'].get('median', 0):.4f}s "
            f"hit cached tokens median={result['hit_cached_tokens'].get('median', 0):.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

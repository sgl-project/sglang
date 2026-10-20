"""CPU-only HBM/H2D scenario accounting and replay of recorded TopK traces.

This is an analytical tool, not a model latency benchmark. Example:
  python benchmark/hisparse/analyze_budget.py --output analysis/scenarios.json
  python benchmark/hisparse/analyze_budget.py --trace topk.jsonl --output replay.json

Trace rows: {"request_id": "r1", "generation": 0, "layer": 0,
             "step": 0, "token_ids": [1, 2, 3]}
Use one row per request/layer/decode step in execution order. MTP query unions
must be prepared separately; this tool does not simulate speculative rollback.
"""

import argparse
import importlib.util
import json
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path


def load_budget_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/budget.py"
    )
    spec = importlib.util.spec_from_file_location("hisparse_budget_analysis", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay_trace(path, capacities, kv_bytes_per_token, warmup_steps=0):
    caches = {capacity: {} for capacity in capacities}
    results = {
        capacity: dict(hits=0, misses=0, counted_rows=0) for capacity in capacities
    }
    last_steps, row_counts = {}, {}
    with Path(path).open(encoding="utf-8") as trace:
        for line_number, line in enumerate(trace, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["request_id"]), row.get("generation", 0), row["layer"])
            step = row["step"]
            if key in last_steps and step <= last_steps[key]:
                raise ValueError(
                    f"Trace row {line_number}: steps must increase for each request/generation/layer."
                )
            last_steps[key] = step
            count = row_counts.get(key, 0)
            row_counts[key] = count + 1
            tokens = list(
                dict.fromkeys(token for token in row["token_ids"] if token >= 0)
            )
            for capacity in capacities:
                if len(tokens) > capacity:
                    raise ValueError(
                        f"Trace row {line_number}: {len(tokens)} selected tokens exceed capacity {capacity}."
                    )
                cache = caches[capacity].setdefault(key, OrderedDict())
                hits = set(tokens).intersection(cache)
                misses = [token for token in tokens if token not in hits]
                # Match the tensor planner's round-level recency: retain hits
                # in their old order, then make misses the newest entries.
                for token in list(cache):
                    if token in hits:
                        cache.move_to_end(token)
                for token in misses:
                    if len(cache) == capacity:
                        cache.popitem(last=False)
                    cache[token] = None
                if count >= warmup_steps:
                    results[capacity]["hits"] += len(hits)
                    results[capacity]["misses"] += len(misses)
                    results[capacity]["counted_rows"] += 1
    output = []
    for capacity, result in results.items():
        total = result["hits"] + result["misses"]
        output.append(
            dict(
                cache_tokens=capacity,
                **result,
                hit_rate=result["hits"] / total if total else None,
                h2d_kv_bytes=result["misses"] * kv_bytes_per_token,
            )
        )
    return output


def scenario_grid(args):
    budget = load_budget_module()
    rows = []
    for input_len in args.input_lens:
        for output_len in args.output_lens:
            context_len = input_len + output_len
            for batch_size in args.batch_sizes:
                for capacity in args.cache_sizes:
                    allocation = budget.resolve_sparse_kv_budget(
                        request_rows=batch_size + 1,
                        context_len=context_len,
                        num_layers=args.layers,
                        top_k=args.top_k,
                        kv_bytes_per_token=args.kv_bytes,
                        cache_tokens=capacity,
                    )
                    indexer_bytes = (
                        (batch_size + 1) * context_len * args.layers * args.index_bytes
                    )
                    for hit_rate in args.hit_rates:
                        h2d_bytes_per_layer = (
                            batch_size
                            * min(args.top_k, context_len)
                            * (1 - hit_rate)
                            * args.kv_bytes
                        )
                        io_us_per_layer = (
                            h2d_bytes_per_layer / (args.h2d_gbps * 1e9) * 1e6
                        )
                        rows.append(
                            dict(
                                input_len=input_len,
                                output_len=output_len,
                                context_len=context_len,
                                batch_size=batch_size,
                                assumed_hit_rate=hit_rate,
                                allocation=asdict(allocation),
                                managed_hbm_bytes=allocation.managed_hbm_bytes,
                                indexer_hbm_bytes=indexer_bytes,
                                dense_main_kv_bytes=(batch_size + 1)
                                * context_len
                                * args.layers
                                * args.kv_bytes,
                                h2d_bytes_per_layer=h2d_bytes_per_layer,
                                ideal_h2d_us_per_layer=io_us_per_layer,
                                optimistic_exposed_io_us_per_step=max(
                                    0, io_us_per_layer - args.overlap_us
                                )
                                * args.layers,
                            )
                        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-lens", type=int, nargs="+", default=[8192, 32768, 131072]
    )
    parser.add_argument(
        "--output-lens", type=int, nargs="+", default=[1024, 8192, 32768]
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument(
        "--cache-sizes", type=int, nargs="+", default=[2048, 4096, 8192]
    )
    parser.add_argument("--hit-rates", type=float, nargs="+", default=[0.5, 0.9, 0.99])
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=61, help="KV layers on this rank")
    parser.add_argument(
        "--kv-bytes",
        type=int,
        default=1152,
        help="Actual CKV + RoPE row bytes per rank/layer",
    )
    parser.add_argument(
        "--index-bytes",
        type=int,
        default=256,
        help="Indexer row bytes; 256 models a BF16 128D NPU indexer",
    )
    parser.add_argument(
        "--h2d-gbps",
        type=float,
        default=32,
        help="Assumed effective per-rank DRAM link GB/s, decimal",
    )
    parser.add_argument(
        "--overlap-us",
        type=float,
        default=100,
        help="Assumed independent compute window per layer; NOT measured",
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Exclude initial rows per request/generation/layer from replay counts",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.top_k,
            args.layers,
            args.kv_bytes,
            args.h2d_gbps,
            *args.cache_sizes,
            *args.input_lens,
            *args.output_lens,
            *args.batch_sizes,
        )
        <= 0
    ):
        parser.error(
            "Lengths, capacities, layers, KV bytes and bandwidth must be positive."
        )
    if (
        args.warmup_steps < 0
        or args.overlap_us < 0
        or args.index_bytes < 0
        or any(not 0 <= rate <= 1 for rate in args.hit_rates)
    ):
        parser.error(
            "Warmup/overlap/index bytes must be nonnegative; hit rates must be in [0, 1]."
        )
    result = dict(
        kind="topk_trace_replay" if args.trace else "analytical_scenarios",
        notes="No device timings. Hit rates, bandwidth and overlap windows in scenarios are assumptions. Memory excludes weights, native indexer pool metadata, runtime temporaries and graph pools.",
        rows=(
            replay_trace(args.trace, args.cache_sizes, args.kv_bytes, args.warmup_steps)
            if args.trace
            else scenario_grid(args)
        ),
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Wrote {len(result['rows'])} rows to {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

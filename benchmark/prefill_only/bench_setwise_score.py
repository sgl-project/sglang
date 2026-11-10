"""Engine benchmark for setwise (multi-position) scoring.

Benchmarks `Engine.async_score(..., score_extraction_token_id=...)` — the setwise
path that pools the classification head at each candidate's anchor token and
returns an ``[N, num_labels]`` matrix in a single forward pass.

Unlike the general ``bench_score.py`` (label-token / MIS scoring), this is a
focused, self-contained harness: it sweeps the **set size** (number of
candidates packed into one item) and reports latency percentiles and throughput
(requests/s and candidates/s) so you can see how a single-forward setwise pass
scales with N.

It talks to the Engine directly (no HTTP) and requires the setwise deployment
constraints (``--disable-radix-cache``, ``--chunked-prefill-size -1``), which it
sets automatically.

Usage:
    # Default sweep over set sizes on the target model
    python bench_setwise_score.py --model-path /path/to/seqcls_model

    # Custom sweep + concurrency + per-candidate length
    python bench_setwise_score.py --model-path /path/to/model \
        --set-sizes 8 16 32 64 --concurrency 16 --num-requests 100 \
        --item-tokens 180 --query-tokens 64

    # Include pooled hidden states in the response (per-anchor pre-head vectors)
    python bench_setwise_score.py --model-path /path/to/model --return-pooled-hidden-states
"""

import argparse
import asyncio
import csv
import random
import statistics
import time
from typing import List, Optional

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark setwise scoring via Engine.async_score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a SequenceClassification model (e.g. Qwen3ForSequenceClassification).",
    )
    parser.add_argument(
        "--score-extraction-token",
        type=str,
        default="<|object_ref_start|>",
        help="Anchor token pooled at each candidate (default: <|object_ref_start|>).",
    )
    parser.add_argument(
        "--set-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Candidates per set (each is one anchor in one item). Swept in order.",
    )
    parser.add_argument(
        "--item-tokens",
        type=int,
        default=180,
        help="Content tokens per candidate before its anchor (default: 180).",
    )
    parser.add_argument(
        "--query-tokens",
        type=int,
        default=0,
        help="Shared query-prefix tokens (default: 0).",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Timed requests per set size (default: 50).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max in-flight requests (default: 8).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup requests per set size, not timed (default: 3).",
    )
    parser.add_argument("--apply-softmax", action="store_true")
    parser.add_argument("--return-pooled-hidden-states", action="store_true")

    # Engine configuration
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--max-prefill-tokens", type=int, default=60000)
    parser.add_argument("--attention-backend", type=str, default="flashinfer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Optional CSV output path."
    )
    return parser.parse_args()


def _random_content(num_tokens: int, anchor_id: int, lo: int = 10, hi: int = 30000):
    """Random content token ids that never collide with the anchor id."""
    ids = [random.randint(lo, hi) for _ in range(num_tokens)]
    return [t + 1 if t == anchor_id else t for t in ids]


def _build_setwise_request(args, anchor_id: int, n_candidates: int):
    """One setwise request: shared query prefix + ONE item with N anchors."""
    query_ids = (
        _random_content(args.query_tokens, anchor_id) if args.query_tokens else []
    )
    item_ids: List[int] = []
    for _ in range(n_candidates):
        item_ids.extend(_random_content(args.item_tokens, anchor_id))
        item_ids.append(anchor_id)
    return query_ids, [item_ids]


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, round((pct / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[k]


async def _run_set_size(engine, args, anchor_id: int, n: int) -> Optional[dict]:
    # Pre-build requests so generation cost is outside the timed section.
    requests = [
        _build_setwise_request(args, anchor_id, n)
        for _ in range(args.num_requests + args.warmup)
    ]
    prompt_tokens = len(requests[0][0]) + len(requests[0][1][0])

    async def _one(query_ids, items):
        t0 = time.perf_counter()
        res = await engine.async_score(
            query=query_ids,
            items=items,
            score_extraction_token_id=anchor_id,
            apply_softmax=args.apply_softmax,
            return_pooled_hidden_states=args.return_pooled_hidden_states,
        )
        # Setwise scores are nested (one [Ni x num_labels] matrix per item) and
        # this harness sends exactly one item, so the per-candidate row count is
        # len(res.scores[0]), not len(res.scores) (which is always 1 = num items).
        return time.perf_counter() - t0, len(res.scores[0]) if res.scores else 0

    # Warmup (also validates the row count == N).
    for q, it in requests[: args.warmup]:
        _, rows = await _one(q, it)
        if rows != n:
            print(f"[warn] set_size={n}: expected {n} score rows, got {rows}")

    sem = asyncio.Semaphore(args.concurrency)

    async def _guarded(q, it):
        async with sem:
            return await _one(q, it)

    timed = requests[args.warmup :]
    wall_start = time.perf_counter()
    results = await asyncio.gather(*(_guarded(q, it) for q, it in timed))
    wall = time.perf_counter() - wall_start

    latencies_ms = sorted(1000.0 * lat for lat, _ in results)
    n_req = len(results)
    return {
        "set_size": n,
        "prompt_tokens": prompt_tokens,
        "requests": n_req,
        "concurrency": args.concurrency,
        "avg_ms": statistics.mean(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "req_per_s": n_req / wall if wall > 0 else 0.0,
        "cand_per_s": (n_req * n) / wall if wall > 0 else 0.0,
    }


def _print_table(rows: List[dict]):
    headers = [
        "set_size",
        "prompt_tokens",
        "requests",
        "avg_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "req_per_s",
        "cand_per_s",
    ]
    print("\n" + " | ".join(f"{h:>13}" for h in headers))
    print("-" * (16 * len(headers)))
    for r in rows:
        cells = []
        for h in headers:
            v = r[h]
            cells.append(f"{v:>13.2f}" if isinstance(v, float) else f"{v:>13}")
        print(" | ".join(cells))


async def main():
    args = parse_args()
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    anchor_id = tokenizer.convert_tokens_to_ids(args.score_extraction_token)
    unk_id = tokenizer.unk_token_id
    if anchor_id is None or (unk_id is not None and anchor_id == unk_id):
        raise ValueError(
            f"score_extraction_token {args.score_extraction_token!r} did not resolve "
            f"to a dedicated token id."
        )

    from sglang.srt.entrypoints.engine import Engine

    # Setwise deployment constraints: positions are full-prompt coordinates, so
    # radix reuse / chunked prefill must be off (the engine also validates this).
    engine = Engine(
        model_path=args.model_path,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        max_prefill_tokens=args.max_prefill_tokens,
        attention_backend=args.attention_backend,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )

    print(
        f"Setwise scoring benchmark | model={args.model_path}\n"
        f"anchor={args.score_extraction_token!r} (id={anchor_id}) | "
        f"item_tokens={args.item_tokens} query_tokens={args.query_tokens} | "
        f"concurrency={args.concurrency} num_requests={args.num_requests}"
    )

    rows = []
    try:
        for n in args.set_sizes:
            row = await _run_set_size(engine, args, anchor_id, n)
            if row is not None:
                rows.append(row)
                print(
                    f"set_size={n:>4}: avg={row['avg_ms']:.1f}ms "
                    f"p95={row['p95_ms']:.1f}ms "
                    f"{row['req_per_s']:.1f} req/s {row['cand_per_s']:.0f} cand/s"
                )
    finally:
        engine.shutdown()

    _print_table(rows)

    if args.output and rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

"""Compare conditional option weights with independent native full scoring.

This is a client benchmark; it never starts a server or loads model weights.
Results are HTTP wall times, including transport and client aggregation.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import httpx

NEUTRAL = dict(
    max_new_tokens=0,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    min_p=0.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    n=1,
    beam_width=None,
    json_schema=None,
    regex=None,
    ebnf=None,
    structural_tag=None,
    custom_params=None,
    logit_bias=None,
    stop=None,
    stop_token_ids=None,
    stop_regex=None,
    min_new_tokens=0,
)


async def post(client, path, payload=None):
    result = await client.post(path, json=payload)
    result.raise_for_status()
    return result.json()


def aggregate_full(result, prefix_len):
    """Obtain a suffix mean from independent, complete native token scores."""
    rows = result["meta_info"]["input_token_logprobs"]
    if len(rows) < 2 or rows[0][0] is not None:
        raise ValueError("Invalid full-sequence reference boundary")
    values = [row[0] for row in rows[1:]]
    if any(x is None or not math.isfinite(x) for x in values):
        raise ValueError("Missing or non-finite full-sequence reference score")
    suffix_values = values[prefix_len - 1 :]
    if prefix_len < 1 or not suffix_values:
        raise ValueError("Reference requires a nonempty token prefix and option")
    return math.fsum(suffix_values) / len(suffix_values)


def normalize_options(logits):
    weights = [math.exp(x - max(logits)) for x in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


async def native_batch(client, sequences):
    return await post(
        client,
        "/generate",
        dict(
            input_ids=sequences,
            sampling_params=NEUTRAL,
            return_logprob=True,
            logprob_start_len=0,
            return_text_in_logprobs=False,
            no_logs=True,
        ),
    )


def split_batches(lengths, max_count, target_tokens):
    batches, batch, tokens = [], [], 0
    for i, count in enumerate(lengths):
        if batch and (len(batch) == max_count or tokens + count > target_tokens):
            batches.append(batch)
            batch, tokens = [], 0
        batch.append(i)
        tokens += count
    if batch:
        batches.append(batch)
    return batches


async def split_reference(client, sequences, batches, concurrency, prefix_len):
    logits = [None] * len(sequences)
    queue = iter(batches)

    async def worker():
        for batch in queue:
            result = await native_batch(client, [sequences[i] for i in batch])
            for i, row in zip(batch, result, strict=True):
                logits[i] = aggregate_full(row, prefix_len)

    tasks = [
        asyncio.create_task(worker()) for _ in range(min(concurrency, len(batches)))
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return normalize_options(logits)


def command_output(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_diagnostics(path, ids):
    if not path:
        return []
    records = []
    for line in Path(path).read_text().splitlines():
        marker = "rawsystemone {"
        if marker in line:
            try:
                record = ast.literal_eval("{" + line.split(marker, 1)[1])
            except (SyntaxError, ValueError):
                continue
            if record.get("id") in ids:
                records.append(record)
    return records


async def run(args):
    headers = {}
    if os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
    output = {
        "metadata": {
            "client_commit": command_output(["git", "rev-parse", "HEAD"]),
            "server_commit": args.server_commit,
            "model_revision": args.model_revision,
            "hardware": args.hardware,
            "client_platform": platform.platform(),
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "cache_condition": args.cache,
            "background_traffic": args.background_traffic,
            "scope": "HTTP wall time, including transport and aggregation; no device-overlap claim",
        },
        "results": [],
    }
    measured_ids = set()
    async with httpx.AsyncClient(
        base_url=args.url, headers=headers, timeout=600
    ) as client:
        info_response = await client.get("/server_info")
        info_response.raise_for_status()
        info = info_response.json()
        # Explicit allowlist: server_info can contain authentication secrets.
        keys = [
            "model_path",
            "tokenizer_path",
            "tokenizer_mode",
            "tokenizer_backend",
            "revision",
            "dtype",
            "quantization",
            "attention_backend",
            "tp_size",
            "dp_size",
            "pp_size",
            "disable_radix_cache",
            "chunked_prefill_size",
            "max_total_tokens",
            "max_running_requests",
            "max_queued_requests",
            "version",
        ]
        keys += [k for k in info if k.startswith("rawsystemone_")]
        output["metadata"]["server"] = {k: info.get(k) for k in keys}

        async def mixed_traffic():
            while True:
                await post(
                    client,
                    "/generate",
                    dict(
                        text="A brief unrelated request.",
                        sampling_params={"max_new_tokens": 8},
                        no_logs=True,
                    ),
                )
                await asyncio.sleep(0)

        traffic = [
            asyncio.create_task(mixed_traffic()) for _ in range(args.background_traffic)
        ]
        try:
            for target_prefix in args.prefix_tokens:
                # Approximate desired token count, then report actual counts.
                prefix = " A customer discusses an appointment." * max(
                    1, target_prefix // 7
                )
                for count in args.options:
                    for suffix_words in args.suffix_tokens:
                        suffixes = [
                            f" {i}"
                            + " appointment"
                            * (
                                suffix_words - 1
                                if not args.mixed_lengths
                                else i % suffix_words
                            )
                            for i in range(count)
                        ]
                        prefix_tokens = await post(
                            client,
                            "/v1/tokenize",
                            dict(prompt=prefix, add_special_tokens=True),
                        )
                        option_tokens = await post(
                            client,
                            "/v1/tokenize",
                            dict(prompt=suffixes, add_special_tokens=False),
                        )
                        prefix_ids = prefix_tokens["tokens"]
                        option_ids = option_tokens["tokens"]
                        if not prefix_ids or any(not ids for ids in option_ids):
                            raise ValueError(
                                "Benchmark needs a nonempty token prefix and options"
                            )
                        sequences = [prefix_ids + ids for ids in option_ids]
                        prefix_len = len(prefix_ids)
                        base = await native_batch(client, sequences)
                        expected = normalize_options(
                            [aggregate_full(row, prefix_len) for row in base]
                        )
                        lengths = [
                            len(row["meta_info"]["input_token_logprobs"])
                            for row in base
                        ]
                        reference_cached = sum(
                            row["meta_info"].get("cached_tokens", 0) for row in base
                        )
                        batches = split_batches(
                            lengths, args.batch_candidates, args.batch_tokens
                        )
                        payload = dict(prefix=prefix, suffixes=suffixes)
                        checked = await post(
                            client,
                            "/v1/rawsystemone",
                            {**payload, "return_token_logprobs": True},
                        )
                        if [row["option_token_count"] for row in checked["data"]] != [
                            len(ids) for ids in option_ids
                        ]:
                            raise AssertionError(
                                "Option token counts differ from independent tokenization"
                            )
                        per_token_diff = max(
                            abs(a["logprob"] - b[0])
                            for row, ref in zip(checked["data"], base, strict=True)
                            for a, b in zip(
                                row["token_logprobs"][1:],
                                ref["meta_info"]["input_token_logprobs"][1:],
                                strict=True,
                            )
                        )
                        if per_token_diff > args.atol:
                            raise AssertionError(
                                f"Per-token difference {per_token_diff} exceeds {args.atol}"
                            )

                        async def sequential():
                            # Deliberately serial test oracle/control, never production.
                            return normalize_options(
                                [
                                    aggregate_full(
                                        (await native_batch(client, [ids]))[0],
                                        prefix_len,
                                    )
                                    for ids in sequences
                                ]
                            )

                        async def single_batch():
                            return normalize_options(
                                [
                                    aggregate_full(row, prefix_len)
                                    for row in await native_batch(client, sequences)
                                ]
                            )

                        async def optimized():
                            result = await post(client, "/v1/rawsystemone", payload)
                            measured_ids.add(result["id"])
                            return [row["score"] for row in result["data"]]

                        methods = {
                            "sequential_reference": sequential,
                            "native_batch_reference": single_batch,
                            **{
                                f"split_reference_c{c}": (
                                    lambda c=c: split_reference(
                                        client, sequences, batches, c, prefix_len
                                    )
                                )
                                for c in args.concurrency
                            },
                            "rawsystemone": optimized,
                        }
                        for name, method in methods.items():
                            latencies, max_diff = [], 0.0
                            for repetition in range(args.warmups + args.repetitions):
                                if args.cache == "cold":
                                    flush = await client.post("/flush_cache")
                                    flush.raise_for_status()
                                started = time.perf_counter()
                                scores = await method()
                                elapsed = time.perf_counter() - started
                                diff = max(
                                    abs(a - b)
                                    for a, b in zip(scores, expected, strict=True)
                                )
                                if diff > args.atol:
                                    raise AssertionError(
                                        f"{name}: score difference {diff} exceeds {args.atol}"
                                    )
                                max_diff = max(max_diff, diff)
                                if repetition >= args.warmups:
                                    latencies.append(elapsed)
                            row = dict(
                                method=name,
                                target_prefix_tokens=target_prefix,
                                target_suffix_tokens=suffix_words,
                                options=count,
                                candidate_tokens_min=min(lengths),
                                candidate_tokens_max=max(lengths),
                                logical_input_tokens=sum(lengths),
                                mixed_lengths=args.mixed_lengths,
                                reference_cached_tokens=reference_cached,
                                reference_uncached_input_tokens=sum(lengths)
                                - reference_cached,
                                payload_sha256=hashlib.sha256(
                                    json.dumps(payload).encode()
                                ).hexdigest(),
                                median_seconds=statistics.median(latencies),
                                p95_seconds=sorted(latencies)[
                                    math.ceil(0.95 * len(latencies)) - 1
                                ],
                                requests_per_second=len(latencies) / sum(latencies),
                                candidates_per_second=count
                                * len(latencies)
                                / sum(latencies),
                                max_score_difference=max_diff,
                                max_per_token_difference=per_token_diff,
                            )
                            output["results"].append(row)
                            print(json.dumps(row), flush=True)
        finally:
            for task in traffic:
                task.cancel()
            outcomes = await asyncio.gather(*traffic, return_exceptions=True)
            failures = [str(x) for x in outcomes if isinstance(x, Exception)]
            output["metadata"]["background_failures"] = failures
    diagnostics = log_diagnostics(args.server_log, measured_ids)
    output["diagnostics"] = diagnostics
    output["evidence"] = {
        "diagnostic_requests": len(diagnostics),
        "observed_concurrent_submission": any(
            d["max_inflight"] > 1 for d in diagnostics
        ),
        "observed_native_cache_hits": any(
            b.get("cached_tokens", 0) > 0 for d in diagnostics for b in d["batches"]
        ),
        "observed_reduced_uncached_input": any(
            d.get("mode") == "shared_prefix"
            and all("uncached_input_tokens" in b for b in d["batches"])
            and sum(b["uncached_input_tokens"] for b in d["batches"])
            < d["logical_input_tokens"]
            for d in diagnostics
        ),
        "device_overlap_measured": False,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", default="rawsystemone-results.json")
    parser.add_argument("--server-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--hardware",
        required=True,
        help="Server GPU model/count, driver and CUDA versions",
    )
    parser.add_argument(
        "--server-log", help="Server INFO log, for per-request cache/admission evidence"
    )
    parser.add_argument(
        "--prefix-tokens", nargs="+", type=int, default=[128, 1024, 4096]
    )
    parser.add_argument("--suffix-tokens", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--options", nargs="+", type=int, default=[2, 8, 32, 96, 128])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--batch-candidates", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--cache", choices=["warm", "cold"], default="warm")
    parser.add_argument("--mixed-lengths", action="store_true")
    parser.add_argument("--background-traffic", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.0002)
    args = parser.parse_args()
    if args.cache == "cold" and args.background_traffic:
        parser.error("Cold-cache flushing requires an otherwise idle server")
    if (
        min(
            args.prefix_tokens
            + args.suffix_tokens
            + args.options
            + args.concurrency
            + [args.batch_candidates, args.batch_tokens, args.repetitions]
        )
        <= 0
    ):
        parser.error("Counts and token limits must be positive")
    if args.warmups < 0 or args.background_traffic < 0:
        parser.error("Warmups and background traffic cannot be negative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

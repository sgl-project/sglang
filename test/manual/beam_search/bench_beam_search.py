"""Benchmark a running server and optionally compare returned beam sequences."""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

TOPICS = [
    "Write a Python function to sum a list.",
    "Explain how rain forms.",
    "List reasons to learn mathematics.",
    "Describe a peaceful garden.",
    "Explain binary search.",
    "Write a short story about a robot.",
    "Describe the solar system.",
    "Give advice on learning a language.",
]


def sequences(body):
    return body.get("meta_info", {}).get("beam_results") or [body]


def requests_for(args, tokenizer, workload):
    count = args.requests or (8 if workload == "uniform" else 16)
    widths = [args.beam_width] if workload == "uniform" else args.widths
    if args.atomic_batch and len(set(widths)) > 1 and not args.best_only:
        raise ValueError(
            "Mixed-width atomic batches require --best-only (the API uses a shared n)."
        )
    items = []
    for i in range(count):
        width = widths[i % len(widths)]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Task {i}: {TOPICS[i % len(TOPICS)]}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "ignore_eos": True,
            "n": 1 if args.best_only else width,
        }
        if width > 1:
            params["beam_width"] = width
        key = "input_ids" if args.input_mode == "ids" else "text"
        value = tokenizer.encode(text) if key == "input_ids" else text
        items.append({key: value, "sampling_params": params})
    return items


async def post(session, url, payload):
    async with session.post(url, json=payload) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {text}")
        return json.loads(text)


async def prepare(session, args, items):
    if not (args.flush_cache or args.prime_cache):
        return
    async with session.post(args.base_url + "/flush_cache") as response:
        response.raise_for_status()
        await response.read()
    if args.prime_cache:
        for item in items:
            request = {k: v for k, v in item.items() if k != "sampling_params"}
            request["sampling_params"] = {
                "max_new_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
                "repetition_penalty": 1.0,
            }
            await post(session, args.base_url + "/generate", request)


async def measure(session, args, items):
    loop = asyncio.get_running_loop()

    async def one(item):
        start = loop.time()
        body = await post(session, args.base_url + "/generate", item)
        return {"body": body, "latency_s": loop.time() - start}

    start = loop.time()
    if args.atomic_batch:
        key = "input_ids" if args.input_mode == "ids" else "text"
        body = await post(
            session,
            args.base_url + "/generate",
            {
                key: [x[key] for x in items],
                "sampling_params": [x["sampling_params"] for x in items],
            },
        )
        elapsed = loop.time() - start
        responses = [{"body": x, "latency_s": elapsed} for x in body]
    else:
        responses = await asyncio.gather(*(one(item) for item in items))
        elapsed = loop.time() - start
    if len(responses) != len(items):
        raise RuntimeError("Response count does not match request count")
    tokens = 0
    for item, response in zip(items, responses):
        beams = sequences(response["body"])
        if len(beams) != item["sampling_params"]["n"]:
            raise RuntimeError("Unexpected number of returned beams")
        if any(len(b["output_ids"]) != args.max_new_tokens for b in beams):
            raise RuntimeError(
                "Unexpected generated length; cannot compare fixed-length throughput"
            )
        tokens += sum(len(b["output_ids"]) for b in beams)
    latencies = sorted(x["latency_s"] for x in responses)
    return {
        "latency_scope": "batch" if args.atomic_batch else "request",
        "elapsed_s": elapsed,
        "returned_tokens": tokens,
        "returned_tokens_s": tokens / elapsed,
        "requests_s": len(items) / elapsed,
        "p50_latency_s": statistics.median(latencies),
        "p95_latency_s": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "responses": responses,
    }


def compare(before, after):
    comparisons = {}
    if before["requests"] != after["requests"]:
        raise ValueError(
            "The two reports must use identical workloads and request parameters"
        )
    for workload, rounds in after["rounds"].items():
        reference = before["rounds"][workload]
        if len(rounds) != len(reference):
            raise ValueError("The two reports must have the same number of rounds")
        total = equal = scored_sequences = 0
        max_score_delta = 0.0
        for a, b in zip(reference, rounds):
            for x, y in zip(a["responses"], b["responses"]):
                sx, sy = sequences(x["body"]), sequences(y["body"])
                total += 1
                equal += [s["output_ids"] for s in sx] == [s["output_ids"] for s in sy]
                scores = {
                    tuple(s["output_ids"]): s.get("meta_info", {}).get("sequence_score")
                    for s in sx
                }
                for s in sy:
                    score = s.get("meta_info", {}).get("sequence_score")
                    old = scores.get(tuple(s["output_ids"]))
                    if old is not None and score is not None:
                        scored_sequences += 1
                        max_score_delta = max(max_score_delta, abs(score - old))
        comparisons[workload] = {
            "requests": total,
            "ordered_tokens_equal": equal,
            "shared_sequences_with_scores": scored_sequences,
            "max_score_delta_on_shared_sequences": max_score_delta,
        }
    return comparisons


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    workloads = ["uniform", "mixed"] if args.workload == "both" else [args.workload]
    report = {
        "arguments": vars(args).copy(),
        "requests": {},
        "rounds": {},
        "summary": {},
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(args.base_url + "/server_info") as response:
            response.raise_for_status()
            report["server_info"] = await response.json()
        for workload in workloads:
            items = requests_for(args, tokenizer, workload)
            report["requests"][workload] = items
            for _ in range(args.warmup):
                await prepare(session, args, items)
                await measure(session, args, items)
            rounds = []
            for i in range(args.rounds):
                await prepare(session, args, items)
                result = await measure(session, args, items)
                rounds.append(result)
                print(
                    f"{workload} round={i}: {result['returned_tokens_s']:.1f} returned token/s",
                    flush=True,
                )
            report["rounds"][workload] = rounds
            report["summary"][workload] = {
                key: statistics.median(r[key] for r in rounds)
                for key in (
                    "returned_tokens_s",
                    "requests_s",
                    "p50_latency_s",
                    "p95_latency_s",
                )
            }
    if args.compare:
        report["comparison"] = compare(
            json.loads(Path(args.compare).read_text()), report
        )
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    if "comparison" in report:
        print(json.dumps(report["comparison"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument(
        "--workload", choices=["uniform", "mixed", "both"], default="both"
    )
    parser.add_argument("--input-mode", choices=["ids", "text"], default="ids")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--atomic-batch", action="store_true")
    prep = parser.add_mutually_exclusive_group()
    prep.add_argument("--flush-cache", action="store_true")
    prep.add_argument("--prime-cache", action="store_true")
    parser.add_argument(
        "--compare",
        help="Compare token IDs and beam scores with a previous JSON report",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.rounds < 1
        or args.warmup < 0
        or (args.requests is not None and args.requests < 1)
    ):
        parser.error("Use positive request/round counts and a nonnegative warmup count")
    if args.max_new_tokens < 1 or args.beam_width < 1 or min(args.widths) < 1:
        parser.error("Token budgets and beam widths must be positive")
    args.base_url = args.base_url.rstrip("/")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

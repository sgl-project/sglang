#!/usr/bin/env python3

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import tempfile
import threading
import time
from collections import Counter


def consume_cumulative_sse(lines, *, start, clock=time.perf_counter):
    output_ids = []
    token_times = []
    final_meta = {}
    saw_done = False
    for raw in lines:
        if not raw or not raw.startswith(b"data: "):
            continue
        payload = raw[6:]
        if payload == b"[DONE]":
            saw_done = True
            break
        body = json.loads(payload)
        current = body.get("output_ids")
        if not isinstance(current, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in current
        ):
            raise ValueError("output_ids must be an integer list")
        if current[: len(output_ids)] != output_ids:
            raise ValueError("cumulative output_ids changed an existing prefix")
        now = clock() - start
        token_times.extend([now] * (len(current) - len(output_ids)))
        output_ids = current
        if isinstance(body.get("meta_info"), dict):
            final_meta = body["meta_info"]
    if not saw_done:
        raise ValueError("SSE stream ended without data: [DONE]")
    return output_ids, token_times, final_meta


def route_plan(num_requests, dp_size):
    if num_requests not in (1, dp_size, 2 * dp_size):
        raise ValueError("num_requests must be 1, dp_size, or 2 * dp_size")
    return [request_index % dp_size for request_index in range(num_requests)]


def deterministic_ids(length, seed, vocab_size):
    rng = random.Random(seed)
    return [rng.randrange(10, vocab_size) for _ in range(length)]


def _sharegpt_conversation_text(record):
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    role_names = {"human": "User", "gpt": "Assistant"}
    turns = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        value = turn.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        role = role_names.get(turn.get("from"), str(turn.get("from", "Turn")))
        turns.append(f"{role}:\n{value.strip()}")
    return "\n\n".join(turns) if turns else None


def build_sharegpt_corpus(records, tokenizer, *, length, seed):
    """Tokenize deterministic natural-text conversations to exactly ``length``."""
    texts = [
        text
        for record in records
        if isinstance(record, dict)
        for text in [_sharegpt_conversation_text(record)]
        if text is not None
    ]
    if not texts:
        raise ValueError("ShareGPT dataset has no usable conversations")

    rng = random.Random(seed)
    order = list(range(len(texts)))
    token_ids = []
    while len(token_ids) < length:
        rng.shuffle(order)
        appended = 0
        for index in order:
            encoded = tokenizer(texts[index] + "\n\n", add_special_tokens=False)[
                "input_ids"
            ]
            if not isinstance(encoded, list) or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in encoded
            ):
                raise ValueError("tokenizer returned invalid input_ids")
            token_ids.extend(encoded)
            appended += len(encoded)
            if len(token_ids) >= length:
                break
        if appended == 0:
            raise ValueError("ShareGPT conversations tokenize to no tokens")
    return token_ids[:length]


def sha256_ids(token_ids):
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()


def run_one(args, request_index, dp_rank, input_ids, barrier):
    import requests

    payload = {
        "rid": f"{args.tag}-req{request_index}-dp{dp_rank}",
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "routed_dp_rank": dp_rank,
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    barrier.wait()
    start = time.perf_counter()
    with requests.post(
        args.url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=(30, args.timeout),
    ) as response:
        response.raise_for_status()
        output_ids, token_times, meta = consume_cumulative_sse(
            response.iter_lines(), start=start
        )
    end = time.perf_counter()

    if len(output_ids) != args.output_len:
        raise ValueError(
            f"request {request_index}: output {len(output_ids)} != {args.output_len}"
        )
    if meta.get("dp_rank") != dp_rank:
        raise ValueError(
            f"request {request_index}: routed to {meta.get('dp_rank')} instead of {dp_rank}"
        )
    if meta.get("prompt_tokens") != args.input_len:
        raise ValueError(f"request {request_index}: prompt token count mismatch")
    if meta.get("completion_tokens") != args.output_len:
        raise ValueError(f"request {request_index}: completion token count mismatch")
    if meta.get("num_retractions") != 0:
        raise ValueError(f"request {request_index}: request retracted")
    if len(token_times) != args.output_len:
        raise ValueError(f"request {request_index}: token timestamp count mismatch")
    verify_count = meta.get("spec_verify_ct")
    if not isinstance(verify_count, int) or verify_count <= 0:
        raise ValueError(f"request {request_index}: invalid verify count")

    ttft = token_times[0]
    tpot = (token_times[-1] - token_times[0]) / (args.output_len - 1)
    if not all(math.isfinite(value) and value >= 0 for value in (ttft, tpot)):
        raise ValueError(f"request {request_index}: invalid timing")
    return {
        "request_index": request_index,
        "requested_dp_rank": dp_rank,
        "dp_rank": meta["dp_rank"],
        "prompt_tokens": meta["prompt_tokens"],
        "output_tokens": len(output_ids),
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": end - start,
        "start_perf_counter": start,
        "end_perf_counter": end,
        "input_sha256": sha256_ids(input_ids),
        "output_sha256": sha256_ids(output_ids),
        "finish_reason": meta.get("finish_reason"),
        "num_retractions": meta.get("num_retractions"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_verify_ct": verify_count,
    }


def atomic_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".mtp-release-", dir=directory)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(value, output, sort_keys=True, indent=2)
            output.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def mean(records, field):
    values = [record[field] for record in records]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return sum(values) / len(values)


def self_test():
    ticks = iter([10.1, 10.2, 10.4])
    lines = [
        b'data: {"output_ids":[1],"meta_info":{}}',
        b'data: {"output_ids":[1,2,3],"meta_info":{}}',
        b'data: {"output_ids":[1,2,3,4],"meta_info":{"dp_rank":0}}',
        b"data: [DONE]",
    ]
    ids, times, meta = consume_cumulative_sse(
        lines, start=10.0, clock=lambda: next(ticks)
    )
    assert ids == [1, 2, 3, 4]
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(times, [0.1, 0.2, 0.2, 0.4])
    )
    assert meta == {"dp_rank": 0}
    assert route_plan(1, 8) == [0]
    assert route_plan(8, 8) == list(range(8))
    assert route_plan(16, 8) == list(range(8)) * 2

    class FakeTokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return {"input_ids": [ord(char) for char in text]}

    sharegpt = [
        {
            "id": "a",
            "conversations": [
                {"from": "human", "value": "alpha"},
                {"from": "gpt", "value": "beta"},
            ],
        },
        {
            "id": "b",
            "conversations": [{"from": "human", "value": "gamma"}],
        },
    ]
    corpus = build_sharegpt_corpus(sharegpt, FakeTokenizer(), length=64, seed=9)
    assert len(corpus) == 64
    assert corpus == build_sharegpt_corpus(sharegpt, FakeTokenizer(), length=64, seed=9)
    assert corpus != build_sharegpt_corpus(
        sharegpt, FakeTokenizer(), length=64, seed=10
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31082/generate")
    parser.add_argument("--input-len", type=int, default=131072)
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--num-requests", type=int, choices=(1, 8, 16))
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tag")
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--sharegpt-file")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-start-span", type=float, default=0.1)
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test: PASS")
        return
    missing = [
        name
        for name in ("num_requests", "seed", "tag", "output")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    if args.output_len < 2:
        parser.error("--output-len must be at least 2")

    ranks = route_plan(args.num_requests, args.dp_size)
    if args.sharegpt_file:
        if not args.tokenizer_path:
            parser.error("--tokenizer-path is required with --sharegpt-file")
        from transformers import AutoTokenizer

        with open(args.sharegpt_file) as source:
            sharegpt = json.load(source)
        if not isinstance(sharegpt, list):
            raise ValueError("ShareGPT root must be a list")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        corpus = build_sharegpt_corpus(
            sharegpt,
            tokenizer,
            length=args.input_len * args.num_requests,
            seed=args.seed,
        )
        request_inputs = [
            corpus[index * args.input_len : (index + 1) * args.input_len]
            for index in range(args.num_requests)
        ]
        input_source = "sharegpt"
    else:
        request_inputs = [
            deterministic_ids(
                args.input_len, args.seed + request_index, args.vocab_size
            )
            for request_index in range(args.num_requests)
        ]
        input_source = "deterministic_ids"

    barrier = threading.Barrier(args.num_requests)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_requests) as pool:
        futures = [
            pool.submit(
                run_one,
                args,
                request_index,
                dp_rank,
                request_inputs[request_index],
                barrier,
            )
            for request_index, dp_rank in enumerate(ranks)
        ]
        records = [future.result() for future in futures]

    starts = [record["start_perf_counter"] for record in records]
    ends = [record["end_perf_counter"] for record in records]
    start_span = max(starts) - min(starts)
    if start_span > args.max_start_span:
        raise ValueError(
            f"request start span {start_span:.6f}s exceeds {args.max_start_span}s"
        )
    actual_rank_counts = Counter(record["dp_rank"] for record in records)
    expected_rank_counts = Counter(ranks)
    if actual_rank_counts != expected_rank_counts:
        raise ValueError(
            f"DP placement mismatch: {actual_rank_counts} != {expected_rank_counts}"
        )

    summary = {
        "schema": "glm52-mtp-release-v1",
        "tag": args.tag,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_requests": args.num_requests,
        "dp_size": args.dp_size,
        "seed": args.seed,
        "input_source": input_source,
        "sharegpt_file": args.sharegpt_file,
        "expected_rank_counts": dict(sorted(expected_rank_counts.items())),
        "actual_rank_counts": dict(sorted(actual_rank_counts.items())),
        "start_span_s": start_span,
        "completion_span_s": max(ends) - min(starts),
        "aggregate_output_throughput_tok_s": (
            args.num_requests * args.output_len / (max(ends) - min(starts))
        ),
        "mean_ttft_s": mean(records, "ttft_s"),
        "mean_tpot_s": mean(records, "tpot_s"),
        "mean_e2e_s": mean(records, "e2e_s"),
        "mean_spec_accept_rate": mean(records, "spec_accept_rate"),
        "mean_spec_accept_length": mean(records, "spec_accept_length"),
        "mean_spec_verify_ct": mean(records, "spec_verify_ct"),
        "mean_verify_step_wall_s": sum(
            record["tpot_s"] * (args.output_len - 1) / record["spec_verify_ct"]
            for record in records
        )
        / len(records),
        "records": records,
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

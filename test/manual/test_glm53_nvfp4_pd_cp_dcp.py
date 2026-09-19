"""Probe an existing GLM PD router at DCP/page/top-k/chunk boundaries.

Run the same requests against DCP=1, DCP=8 eager, and DCP=8 CUDA graph.
This is a transport/numerical smoke test; use GSM8K separately for accuracy.
"""

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", required=True, help="Local tokenizer/model directory"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[
            1,
            7,
            8,
            9,
            63,
            64,
            65,
            511,
            512,
            513,
            2047,
            2048,
            2049,
            8191,
            8192,
            8193,
            16384,
        ],
    )
    args = parser.parse_args()
    if args.concurrency < 1 or any(n < 1 for n in args.lengths):
        parser.error("concurrency and prompt lengths must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    filler = tokenizer.encode(
        "The notebook contains a short note about a blue square. ",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\nWhat color is the square? The square is", add_special_tokens=False
    )

    def request(length):
        if length <= len(suffix):
            ids = suffix[-length:]
        else:
            count = length - len(suffix)
            ids = (filler * ((count + len(filler) - 1) // len(filler)))[:count] + suffix
        start = time.perf_counter()
        response = requests.post(
            args.url.rstrip("/") + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        data = response.json()
        meta = data["meta_info"]
        assert meta["prompt_tokens"] == length, data
        assert meta["completion_tokens"] == 32, data
        output_probs = meta["output_token_logprobs"]
        assert len(output_probs) == 32, data
        assert all(math.isfinite(p[0]) for p in output_probs), data
        record = {
            "length": length,
            "input_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "latency": time.perf_counter() - start,
            "text": data["text"],
            "output_ids": [p[1] for p in output_probs],
            "output_logprobs": [p[0] for p in output_probs],
        }
        print(
            f"OK prompt={length} completion=32 latency={record['latency']:.2f}s",
            flush=True,
        )
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # Save completed cases even when a later request fails.
        for record in pool.map(request, args.lengths):
            records.append(record)
            Path(args.output).write_text(
                json.dumps(records, indent=2, ensure_ascii=False)
            )
    if args.baseline:
        baseline = {r["length"]: r for r in json.loads(Path(args.baseline).read_text())}
        for record in records:
            reference = baseline[record["length"]]
            assert record["input_hash"] == reference["input_hash"], (
                "Different tokenizer/input"
            )
            prefix = 0
            for actual, expected in zip(record["output_ids"], reference["output_ids"]):
                if actual != expected:
                    break
                prefix += 1
            diffs = [
                abs(a - b)
                for a, b in zip(
                    record["output_logprobs"][:prefix],
                    reference["output_logprobs"][:prefix],
                )
            ]
            print(
                f"COMPARE prompt={record['length']} equal_token_prefix={prefix}/32 "
                f"max_shared_prefix_logprob_delta={max(diffs) if diffs else 'N/A'}"
            )
        print(
            "Token differences require inspection; bitwise equality is not an accuracy criterion."
        )


if __name__ == "__main__":
    main()

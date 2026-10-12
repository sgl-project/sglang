"""Reproduce the GLM-5.2 serving smoke from PR #41913 (not bench_serving)."""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="Local checkpoint/tokenizer"
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    results = []

    def generate(ids=None, text=None, n=64, rank=None):
        payload = {"sampling_params": {"temperature": 0, "max_new_tokens": n}}
        payload["input_ids" if ids is not None else "text"] = (
            ids if ids is not None else text
        )
        if rank is not None:
            payload["routed_dp_rank"] = rank
        start = time.perf_counter()
        response = requests.post(url + "/generate", json=payload, timeout=1800)
        response.raise_for_status()
        output = response.json()
        assert output.get("text") is not None, output
        return {
            "seconds": time.perf_counter() - start,
            "text": output["text"],
            "meta_info": output.get("meta_info"),
        }

    # Retain the historical warmup and prefix-reuse requests before measuring.
    for text in (
        "Question: What is 17 + 25?\nAnswer:",
        "Write a Python function to compute the first n Fibonacci numbers.",
        "Explain why the sky looks blue in three sentences.",
    ):
        output = generate(text=text, n=128)
        results.append({"kind": "smoke", "prompt": text, **output})
        print("SMOKE", output, flush=True)
    seed = tokenizer.encode(
        "A careful archivist records observations about mathematics, science, and history. ",
        add_special_tokens=False,
    )
    lengths = [2047, 2048, 8192, 32768]
    if args.tag.startswith("nvfp4_graph"):
        lengths.append(200000)
    for length in lengths:
        prefix = (seed * math.ceil(length / len(seed)))[:length]
        for attempt, suffix in enumerate(
            (
                "\nQuestion: What is 17 plus 25?\nAnswer:",
                "\nQuestion: What is 19 plus 24?\nAnswer:",
            )
        ):
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            ids = prefix[: length - len(suffix_ids)] + suffix_ids
            assert len(ids) == length
            output = generate(ids=ids, n=64, rank=0)
            if attempt == 1:
                assert output["meta_info"]["cached_tokens"] >= length - 128, output
            results.append({"kind": "prefix_reuse", "input_tokens": length, **output})
            print("PREFIX", length, output, flush=True)

    for batch in (1, 4, 16, 32):
        for repeat in range(3):

            def work(i):
                ids = tokenizer.encode(
                    f"Experiment {batch}, round {repeat}, document {i}. ",
                    add_special_tokens=False,
                )
                ids = (ids + seed * 200)[:2048]
                return generate(ids=ids, n=128)

            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as executor:
                outputs = list(executor.map(work, range(batch)))
            elapsed = time.perf_counter() - start
            tokens = sum(o["meta_info"].get("completion_tokens", 0) for o in outputs)
            results.append(
                {
                    "kind": "serving_perf",
                    "concurrency": batch,
                    "repeat": repeat,
                    "warmup": repeat == 0,
                    "seconds": elapsed,
                    "output_tokens": tokens,
                    "output_tokens_per_second": tokens / elapsed,
                    "requests": outputs,
                }
            )
            print("PERF", batch, repeat, elapsed, tokens, tokens / elapsed, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.tag}_probe.json").write_text(
        json.dumps(results, indent=2)
    )
    for batch in (1, 4, 16, 32):
        rates = [
            r["output_tokens_per_second"]
            for r in results
            if r["kind"] == "serving_perf"
            and r["concurrency"] == batch
            and not r["warmup"]
        ]
        print("MEDIAN", batch, statistics.median(rates), flush=True)


if __name__ == "__main__":
    main()

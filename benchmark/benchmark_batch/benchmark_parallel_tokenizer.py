import argparse
import json
import random
import statistics
import time
from typing import Any, Dict, List

from sglang.srt.utils.hf_transformers_utils import ParallelTokenizer, get_tokenizer


def generate_random_prompt(num_tokens: int, tokenizer) -> str:
    vocab_size = tokenizer.vocab_size
    random_token_ids = [random.randint(0, vocab_size - 1) for _ in range(num_tokens)]
    text = tokenizer.decode(random_token_ids, clean_up_tokenization_spaces=True)
    prompt = f"Prompt: {text}"
    return prompt


def time_once(fn) -> float:
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    return (end - start) * 1000.0


def benchmark_lengths(
    model: str,
    lengths: List[int],
    repeats: int,
    trust_remote_code: bool,
    batch_sizes: List[int],
) -> List[Dict[str, Any]]:
    base_tokenizer = get_tokenizer(model, trust_remote_code=trust_remote_code)

    parallel_tokenizer = ParallelTokenizer(base_tokenizer)

    # Warmup
    _ = base_tokenizer("warmup")
    _ = parallel_tokenizer("warmup")

    results: List[Dict[str, Any]] = []

    # Print header once
    print("\nParallelTokenizer Benchmark")
    print("=" * 60)
    print(f"model={model} repeats={repeats}")
    print("-" * 60)
    print(
        f"{'tokens':>8} | {'batch_size':>10} | {'baseline(ms)':>14} | {'parallel(ms)':>14} | {'speedup (x)':>12}"
    )

    for batch_size in batch_sizes:
        for token_count in lengths:
            text = generate_random_prompt(token_count, base_tokenizer)
            texts = [text] * batch_size
            # Correctness check
            ref = base_tokenizer(texts)
            pt = parallel_tokenizer(texts)
            if ref.get("input_ids") != pt.get("input_ids"):
                raise AssertionError(
                    "Mismatch in input_ids between baseline and ParallelTokenizer"
                )

            baseline_runs: List[float] = []
            for _ in range(repeats):
                t = time_once(lambda: base_tokenizer(texts))
                baseline_runs.append(t)

            parallel_runs: List[float] = []
            for _ in range(repeats):
                t = time_once(lambda: parallel_tokenizer(texts))
                parallel_runs.append(t)

            res = {
                "token_count": token_count,
                "batch_size": batch_size,
                "baseline_ms": statistics.mean(baseline_runs),
                "parallel_ms": statistics.mean(parallel_runs),
                "speedup": (
                    statistics.mean(baseline_runs) / statistics.mean(parallel_runs)
                    if statistics.mean(parallel_runs) > 0
                    else 0.0
                ),
                "runs_ms": {
                    "baseline": baseline_runs,
                    "parallel": parallel_runs,
                },
            }
            results.append(res)

            print(
                f"{res['token_count']:>8} | {res['batch_size']:>10} | {res['baseline_ms']:>14.2f} | {res['parallel_ms']:>14.2f} | {res['speedup']:>12.2f}"
            )

    return results


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ParallelTokenizer vs baseline tokenizer (__call__)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HF model name or local path"
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="1024,2048,4096,8192,16384,32768,65536,131072",
        help="Comma-separated target token counts per prompt",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Repeat count per length"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="HF trust_remote_code"
    )
    parser.add_argument(
        "--json-out", type=str, default=None, help="Path to save JSON results"
    )

    args = parser.parse_args()

    random.seed(0)
    lengths = parse_int_list(args.lengths)
    batch_sizes = parse_int_list(args.batch_sizes)
    results = benchmark_lengths(
        model=args.model,
        lengths=lengths,
        batch_sizes=batch_sizes,
        repeats=args.repeats,
        trust_remote_code=args.trust_remote_code,
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {"model": args.model, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSaved JSON to: {args.json_out}")


if __name__ == "__main__":
    main()

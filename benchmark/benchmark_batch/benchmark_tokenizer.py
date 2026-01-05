import argparse
import random
import time
from statistics import mean

from transformers import AutoTokenizer

from sglang.srt.utils.patch_tokenizer import patch_tokenizer


def main():
    args = parse_args()

    print("Tokenizer Benchmark: Sequential vs Batch Processing")
    print("-" * 60)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Functions: {', '.join(args.function)}")
    print(f"Tokens per prompt: {args.num_tokens}")
    print(f"Number of runs per batch size: {args.num_runs}")
    print(f"Batch mode: {', '.join(args.batch_mode)}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer = patch_tokenizer(tokenizer)
    max_batch_size = max(args.batch_sizes)

    token_ids = generate_random_token_ids(
        num_prompts=max_batch_size, num_tokens=args.num_tokens, tokenizer=tokenizer
    )

    if "encode" in args.function:
        prompts = [
            tokenizer.decode(ids, clean_up_tokenization_spaces=True)
            for ids in token_ids
        ]
        run_benchmark(
            name="encode",
            data=prompts,
            sequential_fn=lambda batch: [tokenizer.encode(p) for p in batch],
            batch_fn=lambda batch: tokenizer(batch),
            batch_sizes=args.batch_sizes,
            num_runs=args.num_runs,
            batch_mode=args.batch_mode,
        )

    if "decode" in args.function:
        # mimic DetokenizerManager's usual case
        decode_kwargs = dict(
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )
        run_benchmark(
            name="decode",
            data=token_ids,
            sequential_fn=lambda batch: [
                tokenizer.decode(ids, **decode_kwargs) for ids in batch
            ],
            batch_fn=lambda batch: tokenizer.batch_decode(batch, **decode_kwargs),
            batch_sizes=args.batch_sizes,
            num_runs=args.num_runs,
            batch_mode=args.batch_mode,
        )


def run_benchmark(
    *, name, data, sequential_fn, batch_fn, batch_sizes, num_runs, batch_mode
):
    print("\n" + "=" * 60)
    print(f"{name.upper()} BENCHMARK")
    print("=" * 60)

    results = [
        benchmark(
            data=data,
            batch_size=bs,
            sequential_fn=sequential_fn,
            batch_fn=batch_fn,
            num_runs=num_runs,
            batch_mode=batch_mode,
        )
        for bs in batch_sizes
    ]
    print_results(results=results, func_name=name, batch_mode=batch_mode)


def benchmark(*, data, batch_size, sequential_fn, batch_fn, num_runs, batch_mode):
    batch_data = data[:batch_size]
    run_single = "single" in batch_mode
    run_batch = "batch" in batch_mode

    out = {"batch_size": batch_size}

    if run_single:
        sequential_times = measure_times(
            fn=lambda: sequential_fn(batch_data), num_runs=num_runs
        )
        out |= {
            "avg_sequential_ms": mean(sequential_times),
            "sequential_runs": sequential_times,
        }

    if run_batch:
        batch_times = measure_times(fn=lambda: batch_fn(batch_data), num_runs=num_runs)
        out |= {
            "avg_batch_ms": mean(batch_times),
            "batch_runs": batch_times,
        }

    if run_single and run_batch:
        out["speedup_factor"] = (
            out["avg_sequential_ms"] / out["avg_batch_ms"]
            if out["avg_batch_ms"] > 0
            else 0
        )

    return out


def print_results(*, results, func_name, batch_mode):
    run_single = "single" in batch_mode
    run_batch = "batch" in batch_mode

    for r in results:
        print(f"\nBatch size: {r['batch_size']}")
        if run_single:
            print_runs(
                label=f"Sequential {func_name}",
                runs=r["sequential_runs"],
                avg=r["avg_sequential_ms"],
            )
        if run_batch:
            print_runs(
                label=f"Batch {func_name}", runs=r["batch_runs"], avg=r["avg_batch_ms"]
            )
        if run_single and run_batch:
            print(f"  Speedup factor: {r['speedup_factor']:.2f}x")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {func_name.upper()}")
    print("=" * 60)

    headers = ["Batch Size"]
    if run_single:
        headers.append("Sequential (ms)")
    if run_batch:
        headers.append("Batch (ms)")
    if run_single and run_batch:
        headers.append("Speedup")
    print("".join(f"{h:<18}" for h in headers))
    print("-" * (18 * len(headers)))

    for r in results:
        row = [f"{r['batch_size']}"]
        if run_single:
            row.append(f"{r['avg_sequential_ms']:.2f} ms")
        if run_batch:
            row.append(f"{r['avg_batch_ms']:.2f} ms")
        if run_single and run_batch:
            row.append(f"{r['speedup_factor']:.2f}x")
        print("".join(f"{v:<18}" for v in row))


def print_runs(*, label, runs, avg):
    print(f"  {label}:")
    for i, t in enumerate(runs):
        print(f"    Run {i+1}: {t:.2f} ms")
    print(f"    Average: {avg:.2f} ms")


def measure_times(*, fn, num_runs):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return times


def generate_random_token_ids(*, num_prompts, num_tokens, tokenizer):
    vocab_size = tokenizer.vocab_size
    print(f"Generating {num_prompts} random sequences with {num_tokens} tokens each...")
    return [
        [random.randint(0, vocab_size - 1) for _ in range(num_tokens)]
        for _ in range(num_prompts)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenizer Benchmark: Sequential vs Batch Processing"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer name or path (e.g. nvidia/Kimi-K2-Thinking-NVFP4)",
    )
    parser.add_argument(
        "--function",
        type=str,
        nargs="+",
        choices=["encode", "decode"],
        default=["encode", "decode"],
        help="Functions to benchmark (default: encode decode)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=20000,
        help="Number of tokens per prompt (default: 20000)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--batch-mode",
        nargs="+",
        choices=["single", "batch"],
        default=["single", "batch"],
        help="Benchmark modes to run (default: single batch)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per batch size (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(0)
    main()

import argparse
import random
import time
from statistics import mean

from transformers import AutoTokenizer


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
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per batch size (default: 5)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Skip batch benchmark, only run sequential",
    )
    return parser.parse_args()


def generate_random_token_ids(num_prompts, num_tokens, tokenizer):
    vocab_size = tokenizer.vocab_size
    print(f"Generating {num_prompts} random sequences with {num_tokens} tokens each...")
    return [
        [random.randint(0, vocab_size - 1) for _ in range(num_tokens)]
        for _ in range(num_prompts)
    ]


def measure_times(fn, num_runs):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark(data, batch_size, sequential_fn, batch_fn, num_runs, skip_batch):
    batch_data = data[:batch_size]
    sequential_times = measure_times(lambda: sequential_fn(batch_data), num_runs)

    if skip_batch:
        return {
            "batch_size": batch_size,
            "avg_sequential_ms": mean(sequential_times),
            "sequential_runs": sequential_times,
        }

    batch_times = measure_times(lambda: batch_fn(batch_data), num_runs)
    return {
        "batch_size": batch_size,
        "avg_sequential_ms": mean(sequential_times),
        "avg_batch_ms": mean(batch_times),
        "speedup_factor": mean(sequential_times) / mean(batch_times) if mean(batch_times) > 0 else 0,
        "sequential_runs": sequential_times,
        "batch_runs": batch_times,
    }


def print_results(results, func_name, skip_batch):
    for result in results:
        print(f"\nBatch size: {result['batch_size']}")
        print(f"  Sequential {func_name}:")
        for i, t in enumerate(result["sequential_runs"]):
            print(f"    Run {i+1}: {t:.2f} ms")
        print(f"    Average: {result['avg_sequential_ms']:.2f} ms")

        if not skip_batch:
            print(f"  Batch {func_name}:")
            for i, t in enumerate(result["batch_runs"]):
                print(f"    Run {i+1}: {t:.2f} ms")
            print(f"    Average: {result['avg_batch_ms']:.2f} ms")
            print(f"  Speedup factor: {result['speedup_factor']:.2f}x")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {func_name.upper()}")
    print("=" * 60)

    if skip_batch:
        print(f"{'Batch Size':<10} {'Sequential (ms)':<18}")
        print("-" * 30)
        for r in results:
            print(f"{r['batch_size']:<10} {r['avg_sequential_ms']:.2f} ms")
    else:
        print(f"{'Batch Size':<10} {'Sequential (ms)':<18} {'Batch (ms)':<18} {'Speedup':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['batch_size']:<10} {r['avg_sequential_ms']:.2f} ms{' ' * 8} {r['avg_batch_ms']:.2f} ms{' ' * 8} {r['speedup_factor']:.2f}x")


def run_benchmark(name, data, sequential_fn, batch_fn, batch_sizes, num_runs, skip_batch):
    print("\n" + "=" * 60)
    print(f"{name.upper()} BENCHMARK")
    print("=" * 60)

    results = [
        benchmark(data, bs, sequential_fn, batch_fn, num_runs, skip_batch)
        for bs in batch_sizes
    ]
    print_results(results, name, skip_batch)


def main():
    args = parse_args()

    print("Tokenizer Benchmark: Sequential vs Batch Processing")
    print("-" * 60)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Functions: {', '.join(args.function)}")
    print(f"Tokens per prompt: {args.num_tokens}")
    print(f"Number of runs per batch size: {args.num_runs}")
    print(f"Skip batch: {args.no_batch}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    max_batch_size = max(args.batch_sizes)

    if "encode" in args.function:
        token_ids = generate_random_token_ids(max_batch_size, args.num_tokens, tokenizer)
        prompts = [tokenizer.decode(ids, clean_up_tokenization_spaces=True) for ids in token_ids]
        run_benchmark(
            "encode",
            prompts,
            lambda batch: [tokenizer.encode(p) for p in batch],
            lambda batch: tokenizer(batch),
            args.batch_sizes,
            args.num_runs,
            args.no_batch,
        )

    if "decode" in args.function:
        token_ids = generate_random_token_ids(max_batch_size, args.num_tokens, tokenizer)
        run_benchmark(
            "decode",
            token_ids,
            lambda batch: [tokenizer.decode(ids) for ids in batch],
            lambda batch: tokenizer.batch_decode(batch),
            args.batch_sizes,
            args.num_runs,
            args.no_batch,
        )


if __name__ == "__main__":
    random.seed(0)
    main()

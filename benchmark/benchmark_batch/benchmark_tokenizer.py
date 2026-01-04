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
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
    )
    return parser.parse_args()


def generate_random_prompts(num_prompts, num_tokens, tokenizer):
    vocab_size = tokenizer.vocab_size
    all_prompts = []

    print(f"Generating {num_prompts} random prompts with {num_tokens} tokens each...")
    for i in range(num_prompts):
        random_token_ids = [
            random.randint(0, vocab_size - 1) for _ in range(num_tokens)
        ]
        random_text = tokenizer.decode(
            random_token_ids, clean_up_tokenization_spaces=True
        )

        prompt = f"Prompt {i}: {random_text}"
        tokens = tokenizer.encode(prompt)
        print(f"  Prompt {i}: {len(tokens)} tokens")
        all_prompts.append(prompt)

    return all_prompts


def benchmark_sequential_vs_batch(prompts, batch_size, tokenizer, num_runs):
    sequential_times = []
    for run in range(num_runs):
        batch_prompts = prompts[:batch_size]

        start_time = time.perf_counter()
        for prompt in batch_prompts:
            tokens = tokenizer.encode(prompt)
        sequential_time = (time.perf_counter() - start_time) * 1000
        sequential_times.append(sequential_time)

    batch_times = []
    for run in range(num_runs):
        batch_prompts = prompts[:batch_size]

        start_time = time.perf_counter()
        tokens = tokenizer(batch_prompts)
        batch_time = (time.perf_counter() - start_time) * 1000
        batch_times.append(batch_time)

    return {
        "batch_size": batch_size,
        "avg_sequential_ms": mean(sequential_times),
        "avg_batch_ms": mean(batch_times),
        "speedup_factor": (
            mean(sequential_times) / mean(batch_times) if mean(batch_times) > 0 else 0
        ),
        "sequential_runs": sequential_times,
        "batch_runs": batch_times,
    }


def main():
    args = parse_args()

    print("Tokenizer Benchmark: Sequential vs Batch Processing")
    print("-" * 60)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Tokens per prompt: {args.num_tokens}")
    print(f"Number of runs per batch size: {args.num_runs}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    max_batch_size = max(args.batch_sizes)
    all_prompts = generate_random_prompts(max_batch_size, args.num_tokens, tokenizer)

    results = []
    print("\nRunning benchmark...")

    for batch_size in args.batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        result = benchmark_sequential_vs_batch(
            all_prompts, batch_size, tokenizer, args.num_runs
        )
        results.append(result)

        print(f"  Sequential tokenization (encode):")
        for i, run_time in enumerate(result["sequential_runs"]):
            print(f"    Run {i+1}: {run_time:.2f} ms")
        print(f"    Average: {result['avg_sequential_ms']:.2f} ms")

        print(f"  Batch tokenization (tokenizer):")
        for i, run_time in enumerate(result["batch_runs"]):
            print(f"    Run {i+1}: {run_time:.2f} ms")
        print(f"    Average: {result['avg_batch_ms']:.2f} ms")

        print(f"  Speedup factor: {result['speedup_factor']:.2f}x")

    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(
        f"{'Batch Size':<10} {'Sequential (ms)':<18} {'Batch (ms)':<18} {'Speedup':<10}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['batch_size']:<10} {result['avg_sequential_ms']:.2f} ms{' ' * 8} {result['avg_batch_ms']:.2f} ms{' ' * 8} {result['speedup_factor']:.2f}x"
        )


if __name__ == "__main__":
    random.seed(0)
    main()

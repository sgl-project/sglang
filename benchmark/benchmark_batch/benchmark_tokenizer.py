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


def generate_random_token_ids(num_prompts, num_tokens, tokenizer):
    vocab_size = tokenizer.vocab_size
    all_token_ids = []

    print(f"Generating {num_prompts} random token sequences with {num_tokens} tokens each...")
    for i in range(num_prompts):
        random_token_ids = [
            random.randint(0, vocab_size - 1) for _ in range(num_tokens)
        ]
        all_token_ids.append(random_token_ids)
        print(f"  Sequence {i}: {len(random_token_ids)} tokens")

    return all_token_ids


def benchmark_encode(prompts, batch_size, tokenizer, num_runs):
    sequential_times = []
    for run in range(num_runs):
        batch_prompts = prompts[:batch_size]

        start_time = time.perf_counter()
        for prompt in batch_prompts:
            tokenizer.encode(prompt)
        sequential_time = (time.perf_counter() - start_time) * 1000
        sequential_times.append(sequential_time)

    batch_times = []
    for run in range(num_runs):
        batch_prompts = prompts[:batch_size]

        start_time = time.perf_counter()
        tokenizer(batch_prompts)
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


def benchmark_decode(token_ids_list, batch_size, tokenizer, num_runs):
    sequential_times = []
    for run in range(num_runs):
        batch_token_ids = token_ids_list[:batch_size]

        start_time = time.perf_counter()
        for token_ids in batch_token_ids:
            tokenizer.decode(token_ids)
        sequential_time = (time.perf_counter() - start_time) * 1000
        sequential_times.append(sequential_time)

    batch_times = []
    for run in range(num_runs):
        batch_token_ids = token_ids_list[:batch_size]

        start_time = time.perf_counter()
        tokenizer.batch_decode(batch_token_ids)
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


def print_results(results, func_name):
    for result in results:
        print(f"\nBatch size: {result['batch_size']}")
        print(f"  Sequential {func_name}:")
        for i, run_time in enumerate(result["sequential_runs"]):
            print(f"    Run {i+1}: {run_time:.2f} ms")
        print(f"    Average: {result['avg_sequential_ms']:.2f} ms")

        print(f"  Batch {func_name}:")
        for i, run_time in enumerate(result["batch_runs"]):
            print(f"    Run {i+1}: {run_time:.2f} ms")
        print(f"    Average: {result['avg_batch_ms']:.2f} ms")

        print(f"  Speedup factor: {result['speedup_factor']:.2f}x")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {func_name.upper()}")
    print("=" * 60)
    print(
        f"{'Batch Size':<10} {'Sequential (ms)':<18} {'Batch (ms)':<18} {'Speedup':<10}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['batch_size']:<10} {result['avg_sequential_ms']:.2f} ms{' ' * 8} {result['avg_batch_ms']:.2f} ms{' ' * 8} {result['speedup_factor']:.2f}x"
        )


def main():
    args = parse_args()

    print("Tokenizer Benchmark: Sequential vs Batch Processing")
    print("-" * 60)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Functions: {', '.join(args.function)}")
    print(f"Tokens per prompt: {args.num_tokens}")
    print(f"Number of runs per batch size: {args.num_runs}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    max_batch_size = max(args.batch_sizes)

    if "encode" in args.function:
        all_prompts = generate_random_prompts(max_batch_size, args.num_tokens, tokenizer)
        print("\n" + "=" * 60)
        print("ENCODE BENCHMARK")
        print("=" * 60)

        encode_results = []
        for batch_size in args.batch_sizes:
            result = benchmark_encode(all_prompts, batch_size, tokenizer, args.num_runs)
            encode_results.append(result)

        print_results(encode_results, "encode")

    if "decode" in args.function:
        all_token_ids = generate_random_token_ids(max_batch_size, args.num_tokens, tokenizer)
        print("\n" + "=" * 60)
        print("DECODE BENCHMARK")
        print("=" * 60)

        decode_results = []
        for batch_size in args.batch_sizes:
            result = benchmark_decode(all_token_ids, batch_size, tokenizer, args.num_runs)
            decode_results.append(result)

        print_results(decode_results, "decode")


if __name__ == "__main__":
    random.seed(0)
    main()

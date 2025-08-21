import random
import time
from statistics import mean

from transformers import AutoTokenizer

# CONFIG
TOKENIZER_DIR = (
    "/shared/public/sharing/fait360brew/training/models/meta-llama/Llama-3.2-3B"
)
NUM_TOKENS = 20000  # Each prompt should contain this many tokens
BATCH_SIZES = [1, 2, 4, 8]  # Test different batch sizes
NUM_RUNS = 5  # Number of runs for each batch size to get reliable measurements


def generate_random_prompts(num_prompts, num_tokens, tokenizer):
    """Generate random prompts with specified token count."""
    vocab_size = tokenizer.vocab_size
    all_prompts = []

    print(f"Generating {num_prompts} random prompts with {num_tokens} tokens each...")
    for i in range(num_prompts):
        # Generate random token IDs - this directly gives us the exact token count
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


def benchmark_sequential_vs_batch(prompts, batch_size, tokenizer):
    """Compare sequential vs batch tokenization for a given batch size."""

    # Sequential tokenization using encode()
    sequential_times = []
    for run in range(NUM_RUNS):
        batch_prompts = prompts[:batch_size]  # Use same prompts for fair comparison

        start_time = time.perf_counter()
        for prompt in batch_prompts:
            tokens = tokenizer.encode(prompt)
        sequential_time = (time.perf_counter() - start_time) * 1000
        sequential_times.append(sequential_time)

    # Batch tokenization using tokenizer()
    batch_times = []
    for run in range(NUM_RUNS):
        batch_prompts = prompts[:batch_size]  # Use same prompts for fair comparison

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
    print("Tokenizer Benchmark: Sequential vs Batch Processing")
    print("-" * 60)
    print(f"Tokenizer: {TOKENIZER_DIR}")
    print(f"Tokens per prompt: {NUM_TOKENS}")
    print(f"Number of runs per batch size: {NUM_RUNS}")
    print("-" * 60)

    # Load tokenizer once for all operations
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    # The largest batch size determines how many prompts we need
    max_batch_size = max(BATCH_SIZES)
    all_prompts = generate_random_prompts(max_batch_size, NUM_TOKENS, tokenizer)

    results = []
    print("\nRunning benchmark...")

    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking batch size: {batch_size}")
        result = benchmark_sequential_vs_batch(all_prompts, batch_size, tokenizer)
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

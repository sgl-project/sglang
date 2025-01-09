import argparse
import json
import random
import time

import requests
from lorem_text import lorem

import sglang as sgl
from sglang import RuntimeEndpoint, set_default_backend


def generate_text(num_tokens):
    """Generates a text with approximately num_tokens."""
    num_words = int(num_tokens / 1.93)  # Assuming average word length
    return lorem.words(num_words)


def generate_prompts(
    num_groups=100,
    group_size=100,
    context_length=1000,
    cache_rate=0.8,
    order="random",
    max_tokens=1,
):
    """
    Generate prompts for the benchmark.

    Args:
        num_groups (int): Number of groups, each with shared context.
        group_size (int): Number of requests in each group.
        context_length (int): Length of the context.
        cache_rate (float): Proportion of context cached across prompts within a group.
        order (str): Order of prompts, one of 'random', 'sequential', or 'interleaved'.
        max_tokens (int): Maximum tokens to generate.

    Returns:
        list: List of generated prompts.
    """
    assert order in ["random", "sequential", "interleaved"], "Invalid prompt order"
    prompts = []

    for _ in range(num_groups):
        shared_context = generate_text(context_length * cache_rate)
        for _ in range(group_size):
            prompt = shared_context + generate_text(context_length * (1 - cache_rate))
            prompts.append({"prompt": prompt, "max_tokens": max_tokens})

    if order == "random":
        return random.sample(prompts, len(prompts))
    elif order == "sequential":
        return prompts
    else:  # interleaved
        interleaved_prompts = [prompts[i::group_size] for i in range(group_size)]
        return [item for sublist in interleaved_prompts for item in sublist]


@sgl.function
def test_sgl(s, prompt, max_tokens):
    """SGLang function for generating text based on a prompt."""
    s += prompt
    s += sgl.gen(max_tokens=max_tokens, ignore_eos=True)


def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark prompt generation and SGLang execution."
    )
    parser.add_argument("--port", type=int, default=30000, help="SGLang port")
    parser.add_argument(
        "--order",
        type=str,
        default="random",
        choices=["random", "sequential", "interleaved"],
        help="Order of prompt execution",
    )
    parser.add_argument(
        "--num_groups", type=int, default=100, help="Number of prompt groups"
    )
    parser.add_argument(
        "--group_size", type=int, default=100, help="Size of each prompt group"
    )
    parser.add_argument(
        "--context_length", type=int, default=1000, help="Length of the context"
    )
    parser.add_argument(
        "--cache_rate", type=float, default=0.8, help="Cache rate for shared context"
    )
    parser.add_argument("--output_length", type=int, default=1, help="Output length")
    parser.add_argument("--num_threads", type=int, default=64, help="Number of threads")

    args = parser.parse_args()

    # Initialize SGLang runtime
    set_default_backend(RuntimeEndpoint(f"http://localhost:{args.port}"))
    result_jsonl = []

    # Log current parameters
    print(
        f"Running with num_threads: {args.num_threads}, output_length: {args.output_length}"
    )
    print(f"Cache rate: {args.cache_rate}, Context length: {args.context_length}")
    print(
        f"Group size: {args.group_size}, Num groups: {args.num_groups}, Order: {args.order}"
    )

    # Generate prompts based on input arguments
    prompts = generate_prompts(
        num_groups=args.num_groups,
        group_size=args.group_size,
        context_length=args.context_length,
        cache_rate=args.cache_rate,
        order=args.order,
        max_tokens=args.output_length,
    )

    url = f"http://localhost:{args.port}/flush_cache"

    requests.post(url)
    # sgl.flush_cache()
    time.sleep(1)  # Wait for the cache to be flushed

    # Measure the time taken for batch execution
    tic = time.time()
    test_sgl.run_batch(prompts, num_threads=args.num_threads, progress_bar=True)
    toc = time.time()

    # Record results
    duration = toc - tic
    result_jsonl.append(
        {
            "cache_rate": args.cache_rate,
            "context_length": args.context_length,
            "group_size": args.group_size,
            "num_groups": args.num_groups,
            "order": args.order,
            "output_length": args.output_length,
            "duration": duration,
        }
    )

    # Display throughput information
    throughput = len(prompts) / duration
    print(f"Throughput: {throughput:.2f} requests per second")

    # Write the results to a JSONL file
    with open("result.jsonl", "a") as f:
        for line in result_jsonl:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()

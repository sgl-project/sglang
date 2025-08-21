import concurrent.futures
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

###############################################################################
# CONFIG
###############################################################################
ENDPOINT_URL = "http://127.0.0.1:30000"
TOKENIZER_DIR = "/models/meta-llama/Llama-3.2-3B"

# Benchmark configurations
NUM_REQUESTS = 10  # Total number of requests (each with BATCH_SIZE prompts)
NUM_TOKENS = 32000  # Tokens per prompt
BATCH_SIZE = 8  # Number of prompts per request
GEN_TOKENS = 0  # Tokens to generate per prompt


###############################################################################
# REQUEST GENERATION (in parallel)
###############################################################################
def generate_random_prompt(index, tokenizer_dir, num_tokens):
    """Generate a single random prompt with specified token count."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    vocab_size = tokenizer.vocab_size

    def generate_random_text(num_toks):
        random_token_ids = [random.randint(0, vocab_size - 1) for _ in range(num_toks)]
        return tokenizer.decode(random_token_ids, clean_up_tokenization_spaces=True)

    random_text = generate_random_text(num_tokens)
    return f"Prompt {index}: {random_text}"


def prepare_all_prompts(num_requests, batch_size, num_tokens, tokenizer_dir):
    """Generate prompts for all requests in parallel."""
    total_prompts = num_requests * batch_size
    all_prompts = [None] * total_prompts
    max_workers = min(os.cpu_count() or 1, total_prompts)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_random_prompt, i, tokenizer_dir, num_tokens)
            for i in range(total_prompts)
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_prompts,
            desc="Generating prompts",
        ):
            index = futures.index(future)
            all_prompts[index] = future.result()

    batched_prompts = [
        all_prompts[i * batch_size : (i + 1) * batch_size] for i in range(num_requests)
    ]

    print(
        f"Generated {total_prompts} prompts with {num_tokens} tokens each, grouped into {num_requests} requests of {batch_size} prompts.\n"
    )
    return batched_prompts


###############################################################################
# HTTP CALLS
###############################################################################
def send_batch_request(endpoint, prompts, gen_tokens, request_id):
    """Send a batch of prompts to the /generate endpoint synchronously."""
    sampling_params = {
        "max_new_tokens": gen_tokens,
        "temperature": 0.7,
        "stop": "\n",
    }
    data = {"text": prompts, "sampling_params": sampling_params}

    start_time = time.perf_counter()
    try:
        response = requests.post(
            endpoint.base_url + "/generate", json=data, timeout=3600
        )
        if response.status_code != 200:
            error = response.json()
            raise RuntimeError(f"Request {request_id} failed: {error}")
        result = response.json()
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        avg_per_prompt = elapsed_time / len(prompts) if prompts else 0
        return request_id, elapsed_time, avg_per_prompt, True, len(prompts)
    except Exception as e:
        print(f"[Request] Error for request {request_id}: {e}")
        return request_id, 0, 0, False, len(prompts)


def run_benchmark(endpoint, batched_prompts, batch_size, gen_tokens):
    """Run the benchmark sequentially."""
    results = []
    num_requests = len(batched_prompts)

    # Record start time for total latency
    benchmark_start_time = time.perf_counter()

    for i, batch_prompts in enumerate(batched_prompts):
        request_id = i + 1
        assert (
            len(batch_prompts) == batch_size
        ), f"Request {request_id} should have {batch_size} prompts, got {len(batch_prompts)}"

        print(
            f"[Request] Sending request {request_id}/{num_requests} with {len(batch_prompts)} prompts at {int(time.time()*1000)}"
        )
        result = send_batch_request(endpoint, batch_prompts, gen_tokens, request_id)
        results.append(result)

    # Calculate total latency
    total_latency = (time.perf_counter() - benchmark_start_time) * 1000  # Convert to ms

    return results, total_latency


###############################################################################
# RESULTS
###############################################################################
def process_results(results, total_latency, num_requests):
    """Process and display benchmark results."""
    total_time = 0
    successful_requests = 0
    failed_requests = 0
    request_latencies = []
    per_prompt_latencies = []
    total_prompts = 0

    for request_id, elapsed_time, avg_per_prompt, success, batch_size in results:
        if success:
            successful_requests += 1
            total_prompts += batch_size
            request_latencies.append(elapsed_time)
            per_prompt_latencies.append(avg_per_prompt)
            total_time += elapsed_time / 1000  # Convert to seconds
        else:
            failed_requests += 1

    avg_request_latency = mean(request_latencies) if request_latencies else 0
    avg_per_prompt_latency = mean(per_prompt_latencies) if per_prompt_latencies else 0
    throughput = total_prompts / total_time if total_time > 0 else 0

    print("\nBenchmark Summary:")
    print(f"  Total requests sent:         {len(results)}")
    print(f"  Total prompts sent:          {total_prompts}")
    print(f"  Successful requests:         {successful_requests}")
    print(f"  Failed requests:             {failed_requests}")
    print(f"  Total latency (all requests): {total_latency:.2f} ms")
    print(f"  Avg per request latency:     {avg_request_latency:.2f} ms")
    print(f"  Avg per prompt latency:      {avg_per_prompt_latency:.2f} ms")
    print(f"  Throughput:                  {throughput:.2f} prompts/second\n")


###############################################################################
# MAIN
###############################################################################
def main():
    # Initialize endpoint
    endpoint = RuntimeEndpoint(ENDPOINT_URL)

    # Generate prompts
    batched_prompts = prepare_all_prompts(
        NUM_REQUESTS, BATCH_SIZE, NUM_TOKENS, TOKENIZER_DIR
    )

    # Flush cache before benchmark
    # endpoint.flush_cache()

    # Run benchmark
    print(
        f"Starting benchmark: NUM_TOKENS={NUM_TOKENS}, BATCH_SIZE={BATCH_SIZE}, NUM_REQUESTS={NUM_REQUESTS}\n"
    )
    results, total_latency = run_benchmark(
        endpoint, batched_prompts, BATCH_SIZE, GEN_TOKENS
    )

    # Process and display results
    process_results(results, total_latency, NUM_REQUESTS)


if __name__ == "__main__":
    random.seed(0)
    main()

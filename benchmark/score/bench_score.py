"""
SGLang Scoring Benchmark Script

This script benchmarks SGLang's scoring API performance using HTTP requests.

Current Features:
- HTTP-only implementation (open source compatible)
- Uses /v1/score API endpoint directly
- Single item scoring with batching support
- Configurable RPS, duration, and batch sizes
- Progress tracking and detailed metrics
- Poisson and constant request distributions

Usage:
- Update configuration variables at the top of the file
- Ensure SGLang server is running on the configured HTTP_URL
- Run: python bench_score.py
- Each request will contain ITEM_COUNT_VALUES items for batch scoring

"""

import asyncio
import concurrent.futures  # For parallel prompt generation
import json
import os
import random
from statistics import mean

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

###############################################################################
# CONFIG
###############################################################################
# Server Configuration
SERVER_TYPE = "HTTP"  # Fixed to HTTP for open source

# HTTP Configuration
HTTP_URL = "http://localhost:30000/v1/score"  # Use score API directly

# Score API Config
# ITEM_COUNT_VALUES determines number of items per score request (batch size)
SCORE_QUERY_TOKENS = 120
SCORE_ITEM_TOKENS = 180
SCORE_MODEL_PATH = "Qwen/Qwen3-0.6B"
SCORE_LABEL_TOKEN_IDS = [9454, 2753]  # Yes/No token IDs

# Array of RPS values to test
RPS_VALUES = [70]
# Array of duration values to test
DURATION_SECS_VALUES = [60]  # Duration values in seconds
# Array of item count values to test
ITEM_COUNT_VALUES = [10]  # Number of items per request
# Number of unique requests to generate (will be reused)
NUM_UNIQUE_REQUESTS = 100
DISTRIBUTION = "POISSON"  # Options: "CONSTANT", "POISSON"

# Profiling Configuration
PROFILE = False  # Enable profiling with START_PROFILE/STOP_PROFILE prompts
# Directory for profiler output
SGLANG_TORCH_PROFILER_DIR = "/shared/user/sglang-oss-trace/remove-decode"
if PROFILE:
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = SGLANG_TORCH_PROFILER_DIR

# Special token to replicate for precise token counting
SPECIAL_REPLICATED_TOKEN = "<|im_start|>"


###############################################################################
# REQUEST GENERATION (in parallel)
###############################################################################
def prepare_all_requests_parallel(num_requests, item_count):
    """
    Generates unique requests in parallel, then reuses them to create the
    full request list. Returns a list of str prompts for HTTP.
    """
    # Load tokenizer once here to verify special token and get precise counts
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SCORE_MODEL_PATH)

    # Verify that our special token produces exactly 1 token
    special_token_count = len(
        tokenizer.encode(SPECIAL_REPLICATED_TOKEN, add_special_tokens=False)
    )
    print(
        f"Special token '{SPECIAL_REPLICATED_TOKEN}' produces "
        f"{special_token_count} token(s)"
    )

    def generate_text_with_token_count(num_toks):
        """Generate text with precise token count using replicated token."""
        if special_token_count == 1:
            # Simple case: token maps to exactly 1 token
            return SPECIAL_REPLICATED_TOKEN * num_toks
        else:
            print(
                f"Special token '{SPECIAL_REPLICATED_TOKEN}' produces more than 1 token!!!"
            )
            # Handle case where special token produces multiple tokens
            # Repeat the token enough times to get at least num_toks tokens
            repetitions = (num_toks + special_token_count - 1) // special_token_count
            text = SPECIAL_REPLICATED_TOKEN * repetitions

            # Verify we got the expected token count (approximately)
            actual_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if actual_tokens < num_toks:
                print(
                    f"Warning: Generated {actual_tokens} tokens, "
                    f"expected {num_toks}"
                )

            return text

    def build_request(index):
        """Build a single request using the shared tokenizer."""
        try:
            # Generate query and items for score API
            query = generate_text_with_token_count(SCORE_QUERY_TOKENS)
            items = [
                generate_text_with_token_count(SCORE_ITEM_TOKENS)
                for _ in range(item_count)
            ]

            # Return as dict for score API format
            score_data = {
                "query": query,
                "items": items,
                "label_token_ids": SCORE_LABEL_TOKEN_IDS,
                "model": SCORE_MODEL_PATH,
            }
            return (index, score_data)

        except Exception as e:
            print(f"Error building request {index}: {e}")
            return (index, None)

    # Generate only the unique requests
    unique_requests = [None] * NUM_UNIQUE_REQUESTS

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid
    # tokenizer loading issues across processes
    max_workers = min(8, os.cpu_count() or 1)  # Limit to 8 threads max

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in tqdm(
            range(NUM_UNIQUE_REQUESTS), desc="Submitting prompt generation tasks"
        ):
            future = executor.submit(build_request, i)
            futures.append(future)

        # Collect results as they complete
        for f in tqdm(
            concurrent.futures.as_completed(futures),
            desc="Building unique requests",
            total=NUM_UNIQUE_REQUESTS,
        ):
            try:
                index, req_data = f.result()
                if req_data is not None:
                    unique_requests[index] = req_data
                else:
                    print(f"Failed to build request {index}")
            except Exception as e:
                print(f"Error processing request result: {e}")

    # Check if we have any valid requests
    valid_requests = [req for req in unique_requests if req is not None]
    if not valid_requests:
        raise RuntimeError("Failed to generate any valid requests")

    print(
        f"Successfully generated {len(valid_requests)} out of "
        f"{NUM_UNIQUE_REQUESTS} unique requests"
    )

    # Create the full request list by cycling through unique requests
    print(
        f"Reusing {len(valid_requests)} unique requests to create "
        f"{num_requests} total requests..."
    )
    all_requests = []
    for i in tqdm(range(num_requests), desc="Reusing requests"):
        unique_index = i % len(valid_requests)
        all_requests.append(valid_requests[unique_index])

    print("All prompts/requests prepared.\n")
    return all_requests


###############################################################################
# PROFILING HELPERS
###############################################################################
async def send_profile_request(profile_text, item_count, session=None):
    """Send a profile request and wait for completion."""
    try:
        if session:
            print(f"Sending {profile_text} request via HTTP...")

            # Determine the correct endpoint
            base_url = HTTP_URL.rsplit("/", 2)[0]  # Remove /v1/score
            if profile_text == "START_PROFILE":
                endpoint_url = f"{base_url}/start_profile"
            elif profile_text == "STOP_PROFILE":
                endpoint_url = f"{base_url}/stop_profile"
            else:
                print(f"Unknown profile request: {profile_text}")
                return

            headers = {"Content-Type": "application/json"}

            async with session.post(endpoint_url, headers=headers) as resp:
                resp_text = await resp.text()
                if resp.status == 200:
                    print(f"{profile_text} request completed")
                else:
                    print(
                        f"{profile_text} request failed with status "
                        f"{resp.status}: {resp_text}"
                    )
        else:
            print(f"Cannot send {profile_text} request - missing session")

    except Exception as e:
        print(f"Error sending {profile_text} request: {e}")


###############################################################################
# HTTP CALLS
###############################################################################
def build_http_request_json(score_data):
    """Build HTTP request JSON for /v1/score endpoint.

    Score API format:
    {
        "query": "Generated query text with SCORE_QUERY_TOKENS tokens",
        "items": ["item1", "item2", ...],  # Items to score with SCORE_ITEM_TOKENS each
        "label_token_ids": [token_id1, token_id2],  # Target token IDs
        "model": "/path/to/model"
    }

    Args:
        score_data: A dict containing query, items, label_token_ids, and model
    """
    # score_data is already in the correct format from build_request
    return json.dumps(score_data)


async def make_http_call(session, score_data, request_id, results_queue):
    """HTTP call to /v1/score endpoint."""
    try:
        start_time = asyncio.get_event_loop().time()

        request_json = build_http_request_json(score_data)
        headers = {"Content-Type": "application/json"}

        async with session.post(HTTP_URL, data=request_json, headers=headers) as resp:
            resp_text = await resp.text()

            if resp.status != 200:
                print(
                    f"[HTTP] Request {request_id} failed with status "
                    f"{resp.status}: {resp_text}"
                )
                completion_time = asyncio.get_event_loop().time()
                await results_queue.put((request_id, 0, False, completion_time))
                return

            # Parse score API response
            try:
                response_data = json.loads(resp_text)
                # Score API returns scores for each item
                # For now, just verify we got a valid response
                if "scores" in response_data or "logprobs" in response_data:
                    success = True
                else:
                    print(
                        f"[HTTP] Request {request_id} missing expected fields in response"
                    )
                    success = False
            except json.JSONDecodeError:
                print(f"[HTTP] Request {request_id} failed to parse JSON response")
                success = False

        completion_time = asyncio.get_event_loop().time()
        elapsed_time = (completion_time - start_time) * 1000
        await results_queue.put((request_id, elapsed_time, success, completion_time))

    except Exception as e:
        print(f"[HTTP] Error for request {request_id}: {e}")
        completion_time = asyncio.get_event_loop().time()
        await results_queue.put((request_id, 0, False, completion_time))


###############################################################################
# RESULTS
###############################################################################
async def process_results(
    results_queue,
    num_requests,
    send_duration,
    total_duration,
    rps,
    duration_secs,
    item_count,
    test_start_time,
):
    """Processes results and groups them by minute intervals.
    Returns a list of dictionaries, one for each minute."""
    all_results = []

    # Collect all results
    for _ in range(num_requests):
        result = await results_queue.get()
        request_id, elapsed_time, success, completion_time = result
        all_results.append(
            {
                "request_id": request_id,
                "elapsed_time": elapsed_time,
                "success": success,
                "completion_time": completion_time,
            }
        )

    # Group results by minute intervals
    minute_results = []
    num_minutes = int(duration_secs // 60) + (1 if duration_secs % 60 > 0 else 0)

    for minute in range(num_minutes):
        minute_start = test_start_time + (minute * 60)
        minute_end = test_start_time + ((minute + 1) * 60)

        # Filter results that completed in this minute
        minute_data = [
            r for r in all_results if minute_start <= r["completion_time"] < minute_end
        ]

        response_times = [r["elapsed_time"] for r in minute_data if r["success"]]
        successful_requests = len([r for r in minute_data if r["success"]])
        failed_requests = len([r for r in minute_data if not r["success"]])

        avg_response_time = mean(response_times) if response_times else 0

        # Calculate percentiles using numpy
        if response_times:
            p50 = np.percentile(response_times, 50)
            p90 = np.percentile(response_times, 90)
            p99 = np.percentile(response_times, 99)
        else:
            p50 = p90 = p99 = 0

        minute_result = {
            "test_duration_secs": duration_secs,
            "minute_interval": minute + 1,
            "target_rps": rps,
            "item_count": item_count,
            "server_type": SERVER_TYPE,
            "distribution": DISTRIBUTION,
            "unique_requests": NUM_UNIQUE_REQUESTS,
            "total_requests": len(minute_data),
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "send_duration_secs": send_duration,
            "total_duration_secs": total_duration,
            "avg_response_time_ms": avg_response_time,
            "p50_response_time_ms": p50,
            "p90_response_time_ms": p90,
            "p99_response_time_ms": p99,
        }

        minute_results.append(minute_result)

        print(
            f"\nMinute {minute + 1} Summary for RPS {rps}, "
            f"Duration {duration_secs}s, Item Count {item_count}:"
        )
        print(f"  Requests completed in minute: {len(minute_data)}")
        print(f"  Successful requests:   {successful_requests}")
        print(f"  Failed requests:       {failed_requests}")
        print(f"  Average response time: {avg_response_time:.2f} ms")
        print(f"  P50 response time:     {p50:.2f} ms")
        print(f"  P90 response time:     {p90:.2f} ms")
        print(f"  P99 response time:     {p99:.2f} ms")

    # Also print overall summary
    all_response_times = [r["elapsed_time"] for r in all_results if r["success"]]
    total_successful = len([r for r in all_results if r["success"]])
    total_failed = len([r for r in all_results if not r["success"]])

    overall_avg = mean(all_response_times) if all_response_times else 0
    if all_response_times:
        overall_p50 = np.percentile(all_response_times, 50)
        overall_p90 = np.percentile(all_response_times, 90)
        overall_p99 = np.percentile(all_response_times, 99)
    else:
        overall_p50 = overall_p90 = overall_p99 = 0

    print(
        f"\nOverall Summary for RPS {rps}, Duration {duration_secs}s, "
        f"Item Count {item_count}:"
    )
    print(f"  Test duration:         {duration_secs} seconds")
    print(f"  Server type:           {SERVER_TYPE}")
    print(f"  HTTP mode:             SINGLE_ITEM_SCORING")
    print(f"  Target RPS:            {rps}")
    print(f"  Item count:            {item_count}")
    print(f"  Distribution:          {DISTRIBUTION}")
    print(f"  Unique requests generated: {NUM_UNIQUE_REQUESTS}")
    print(f"  Total requests sent:   {num_requests}")
    print(f"  Successful requests:   {total_successful}")
    print(f"  Failed requests:       {total_failed}")
    print(f"  Time to send all requests: {send_duration:.2f} seconds")
    print(f"  Time for all requests to complete: {total_duration:.2f} seconds")
    print(f"  Average response time: {overall_avg:.2f} ms")
    print(f"  P50 response time:     {overall_p50:.2f} ms")
    print(f"  P90 response time:     {overall_p90:.2f} ms")
    print(f"  P99 response time:     {overall_p99:.2f} ms\n")

    return minute_results


###############################################################################
# MAIN
###############################################################################
async def run_benchmark(rps, duration_secs, item_count):
    """Run a single benchmark with the given RPS value."""
    num_requests = int(rps * duration_secs)
    print(
        f"Starting benchmark with RPS={rps}, Duration={duration_secs}s, "
        f"Item Count={item_count}, num_requests={num_requests}"
    )
    print(f"Server Type: {SERVER_TYPE}")
    print(f"HTTP Mode: SINGLE_ITEM_SCORING")
    print(f"Profiling Enabled: {PROFILE}")

    # Build requests in parallel (unmeasured)
    all_requests = prepare_all_requests_parallel(num_requests, item_count)

    results_queue = asyncio.Queue()
    tasks = []

    # Track timing for sending requests
    send_start_time = asyncio.get_event_loop().time()

    # HTTP implementation (open source only supports HTTP with /v1/score API)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:

        # Send START_PROFILE if profiling is enabled
        if PROFILE:
            await send_profile_request("START_PROFILE", item_count, session=session)

        # Add progress bar for sending requests
        with tqdm(
            total=len(all_requests),
            desc=f"Sending HTTP score requests at {rps} RPS",
            unit="req",
        ) as pbar:
            for i, score_data in enumerate(all_requests):
                request_id = i + 1
                tasks.append(
                    asyncio.create_task(
                        make_http_call(session, score_data, request_id, results_queue)
                    )
                )

                # Update progress bar
                pbar.update(1)

                # Throttle based on distribution
                if i < len(all_requests) - 1:
                    if DISTRIBUTION == "CONSTANT":
                        interval = 1 / rps
                        await asyncio.sleep(interval)
                    elif DISTRIBUTION == "POISSON":
                        # For Poisson process, inter-arrival times follow
                        # exponential distribution
                        interval = random.expovariate(rps)
                        await asyncio.sleep(interval)
                    else:
                        raise ValueError(
                            f"Unknown distribution: {DISTRIBUTION}. "
                            f"Use 'CONSTANT' or 'POISSON'."
                        )

        send_end_time = asyncio.get_event_loop().time()
        send_duration = send_end_time - send_start_time

        # Wait for all requests to complete with progress tracking
        print(f"Waiting for {len(tasks)} HTTP score requests to complete...")
        with tqdm(
            total=len(tasks), desc="Completing HTTP score requests", unit="req"
        ) as completion_pbar:
            completed_tasks = []
            for task in asyncio.as_completed(tasks):
                await task
                completed_tasks.append(task)
                completion_pbar.update(1)

        # Send STOP_PROFILE if profiling is enabled
        if PROFILE:
            await send_profile_request("STOP_PROFILE", item_count, session=session)

    completion_end_time = asyncio.get_event_loop().time()
    total_duration = completion_end_time - send_start_time

    return await process_results(
        results_queue,
        num_requests,
        send_duration,
        total_duration,
        rps,
        duration_secs,
        item_count,
        send_start_time,
    )


async def main():
    """Main function that runs benchmarks for all RPS values."""
    total_combinations = (
        len(DURATION_SECS_VALUES) * len(RPS_VALUES) * len(ITEM_COUNT_VALUES)
    )
    print(
        f"Running benchmarks for {len(DURATION_SECS_VALUES)} duration "
        f"values, {len(RPS_VALUES)} RPS values, and "
        f"{len(ITEM_COUNT_VALUES)} item count values = "
        f"{total_combinations} total combinations"
    )
    print(f"Server Type: {SERVER_TYPE}")
    print(f"HTTP Mode: SINGLE_ITEM_SCORING")
    print(f"Score API URL: {HTTP_URL}")
    print(f"Query tokens per request: {SCORE_QUERY_TOKENS}")
    print(f"Item tokens per item: {SCORE_ITEM_TOKENS}")
    print(f"Items per request (batch size): {ITEM_COUNT_VALUES}")
    print(f"Profiling Enabled: {PROFILE}")
    print(f"Duration values: {DURATION_SECS_VALUES}")
    print(f"RPS values: {RPS_VALUES}")
    print(f"Item count values: {ITEM_COUNT_VALUES}")
    print("=" * 80)

    all_results = []

    for duration_secs in DURATION_SECS_VALUES:
        for rps in RPS_VALUES:
            for item_count in ITEM_COUNT_VALUES:
                result = await run_benchmark(rps, duration_secs, item_count)
                all_results.extend(result)  # Extend with minute results

    # Print CSV header and results
    print("\n" + "=" * 80)
    print("FINAL CSV RESULTS:")
    print("=" * 80)

    # CSV Header
    headers = [
        "test_duration_secs",
        "minute_interval",
        "target_rps",
        "item_count",
        "server_type",
        "distribution",
        "unique_requests",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "send_duration_secs",
        "total_duration_secs",
        "avg_response_time_ms",
        "p50_response_time_ms",
        "p90_response_time_ms",
        "p99_response_time_ms",
    ]
    print(",".join(headers))

    # CSV Data
    for result in all_results:
        row = [
            result["test_duration_secs"],
            result["minute_interval"],
            result["target_rps"],
            result["item_count"],
            result["server_type"],
            result["distribution"],
            result["unique_requests"],
            result["total_requests"],
            result["successful_requests"],
            result["failed_requests"],
            f"{result['send_duration_secs']:.2f}",
            f"{result['total_duration_secs']:.2f}",
            f"{result['avg_response_time_ms']:.2f}",
            f"{result['p50_response_time_ms']:.2f}",
            f"{result['p90_response_time_ms']:.2f}",
            f"{result['p99_response_time_ms']:.2f}",
        ]
        print(",".join(map(str, row)))


if __name__ == "__main__":
    asyncio.run(main())

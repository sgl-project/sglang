import argparse
import asyncio
import random
import time

import aiohttp
import requests
from lorem_text import lorem
from tqdm.asyncio import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=512,
        help="Length of each new request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=256,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--time-interval",
        type=float,
        default=1.0,
        help="Average time interval between requests for each client (seconds)",
    )
    parser.add_argument(
        "--max-parallel-threads",
        type=int,
        default=64,
        help="Max concurrency for outgoing requests",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname or IP (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000)",
    )
    return parser.parse_args()


def generate_text(num_tokens: int) -> str:
    """Generates a text with approximately `num_tokens` tokens."""
    # Approximate 1 word ~ 1.93 tokens
    num_words = int(num_tokens / 1.93)
    return lorem.words(num_words)


async def make_request(
    session: aiohttp.ClientSession,
    history_context: str,
    request_length: int,
    output_length: int,
    url: str,
) -> tuple[str, float]:
    """Make a single inference request to the server and return
    updated history and latency.
    """
    # Build request payload
    new_request = "\n".join([history_context, generate_text(request_length)])
    payload = {
        "text": new_request,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_length,
            "ignore_eos": True,
        },
    }

    start_time = asyncio.get_event_loop().time()
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        data = await response.json()
        latency = asyncio.get_event_loop().time() - start_time

    # Extract the text from the response and append to history context
    response_text = data.get("text", "")
    updated_history = "".join([new_request, response_text])
    return updated_history, latency


async def client_task(
    client_id: int,
    num_rounds: int,
    request_length: int,
    output_length: int,
    interval: float,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    progress_bar,
    url: str,
) -> float:
    """Simulate a single client sending requests num_rounds times,
    measuring average latency.
    """
    latencies = []
    history_context = ""

    for _ in range(num_rounds):
        # Random sleep to spread out requests (exponential distribution)
        await asyncio.sleep(random.expovariate(1 / interval))

        # Limit concurrency using semaphore
        async with semaphore:
            history_context, latency = await make_request(
                session, history_context, request_length, output_length, url
            )
            latencies.append(latency)

        # Update overall progress
        progress_bar.update(1)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    # print(f"Client {client_id}: Average latency = {avg_latency:.4f}s")
    return avg_latency


async def main(args):
    """Main coroutine to coordinate the benchmark."""
    # Build URLs if user hasn't specified full endpoints
    args.url = f"http://{args.host}:{args.port}/generate"
    args.flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    requests.post(args.flush_cache_url)

    print(f"Using endpoint: {args.url}")
    print(f"Using flush cache endpoint: {args.flush_cache_url}")

    async with aiohttp.ClientSession() as session:
        # Create a semaphore to cap concurrency
        semaphore = asyncio.Semaphore(args.max_parallel_threads)

        # Set up a progress bar for total requests
        total_requests = args.num_clients * args.num_rounds
        with tqdm(total=total_requests, desc="Progress", unit="req") as progress_bar:
            tasks = [
                client_task(
                    client_id=i,
                    num_rounds=args.num_rounds,
                    request_length=args.request_length,
                    output_length=args.output_length,
                    interval=args.time_interval,
                    session=session,
                    semaphore=semaphore,
                    progress_bar=progress_bar,
                    url=args.url,
                )
                for i in range(args.num_clients)
            ]

            results = await asyncio.gather(*tasks)

    # Summarize results
    overall_avg_latency = sum(results) / len(results) if results else 0
    print(f"Overall Average Latency: {overall_avg_latency:.4f}s")


if __name__ == "__main__":
    args = parse_args()

    start_time = time.time()
    asyncio.run(main(args))
    duration = time.time() - start_time
    print(f"All clients finished {args.num_rounds} rounds in {duration:.2f} seconds.")

import threading
import requests
import time
from lorem_text import lorem
from tqdm.asyncio import tqdm
import random
import asyncio
import aiohttp

# Configurable parameters
NUM_CLIENTS = 1000  # Number of concurrent clients
REQUEST_LENGTH = 128  # Length of each new request
OUTPUT_LENGTH = 128  # Length of each output
NUM_ROUNDS = 10  # Number of rounds per client
TIME_INTERVAL = 1  # Time interval between requests
MAX_PARALLEL_THREADS = 64

# API endpoint
URL = "http://localhost:30001/generate"
# Global progress counter
progress_counter = 0
progress_lock = threading.Lock()


def generate_text(num_tokens):
    """Generates a text with approximately num_tokens."""
    num_words = int(num_tokens / 1.93)  # Assuming average word length
    return lorem.words(num_words)


async def client_request(
    client_id, request_length, history_context, output_length, session, progress_bar
):
    global progress_counter
    new_request = "\n".join([history_context, generate_text(request_length)])
    payload = {
        "text": new_request,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_length,
            "ignore_eos": True,
        },
    }

    try:
        tic = asyncio.get_event_loop().time()
        async with session.post(URL, json=payload) as response:
            response.raise_for_status()
            response_text = await response.json()
            latency = asyncio.get_event_loop().time() - tic
            response_text = response_text.get("text", "")
            history_context = "\n".join([history_context, response_text])

        with progress_lock:
            progress_counter += 1
            progress_bar.update(1)
        return history_context, latency

    except Exception as e:
        raise e


async def client_task(
    client_id, num_rounds, request_length, output_length, interval, progress_bar
):
    history_context = ""
    latencies = []
    async with aiohttp.ClientSession() as session:
        for _ in range(num_rounds):
            await asyncio.sleep(random.expovariate(1 / interval))
            history_context, latency = await client_request(
                client_id,
                request_length,
                history_context,
                output_length,
                session,
                progress_bar,
            )
            latencies.append(latency)

    avg_latency = sum(latencies) / num_rounds
    print(f"Client {client_id}: Average latency = {avg_latency:.2f}s")
    return avg_latency


async def main():
    with tqdm(
        total=NUM_CLIENTS * NUM_ROUNDS, desc="Progress", unit="request"
    ) as progress_bar:
        tasks = [
            client_task(
                client_id,
                NUM_ROUNDS,
                REQUEST_LENGTH,
                OUTPUT_LENGTH,
                TIME_INTERVAL,
                progress_bar,
            )
            for client_id in range(NUM_CLIENTS)
        ]
        results = await asyncio.gather(*tasks)
    overall_avg_latency = sum(results) / len(results) if results else 0
    print(f"Overall Average Latency: {overall_avg_latency:.2f}s")
    return


if __name__ == "__main__":
    requests.post("http://localhost:30001/flush_cache")
    start_time = time.time()
    asyncio.run(main())
    print(f"All clients finished in {time.time() - start_time:.2f} seconds.")

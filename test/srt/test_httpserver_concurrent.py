"""
python3 -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --port 30000

Output:
The capital of France is Paris.\nThe capital of the United States is Washington, D.C.

The capital of the United Kindom is London.\nThe capital of the United Kingdom is London.\nThe capital of
"""

import argparse
import asyncio
import json
import time

import aiohttp
import requests


async def send_request(url, data, delay=0):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            output = await resp.json()
    return output


async def main(args):
    url = f"{args.host}:{args.port}"
    task1 = send_request(
        url + "/generate",
        {
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 128},
        },
        delay=1,
    )

    task2 = send_request(
        url + "/generate",
        {
            "text": "The capital of the United Kindom is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 128},
        },
    )

    rets = await asyncio.gather(task1, task2)
    print(rets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    asyncio.run(main(args))

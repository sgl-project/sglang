"""
Usage:
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
python3 test_httpserver_llava.py

Output:
The image features a man standing on the back of a yellow taxi cab, holding
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


async def test_concurrent(args):
    url = f"{args.host}:{args.port}"

    response = []
    for i in range(8):
        response.append(
            send_request(
                url + "/generate",
                {
                    "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this picture ASSISTANT:",
                    "image_data": "test_image.png",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                },
            )
        )

    rets = await asyncio.gather(*response)
    for ret in rets:
        print(ret["text"])


def test_streaming(args):
    url = f"{args.host}:{args.port}"

    response = requests.post(
        url + "/generate",
        json={
            "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this picture ASSISTANT:",
            "image_data": "test_image.png",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 128,
            },
            "stream": True,
        },
        stream=True,
    )

    prev = 0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            output = data["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    asyncio.run(test_concurrent(args))

    test_streaming(args)

"""
Usage:
# Run a Pixtral model with SGLang:
# HuggingFace:
python -m sglang.launch_server --model-path mistral-community/pixtral-12b --port=30000
# ModelScope:
python -m sglang.launch_server --model-path AI-ModelScope/pixtral-12b --port=30000

# Then test it with:
python pixtral_server.py

This script tests Pixtral model with both single and multiple images.
"""

import argparse
import asyncio
import json

import aiohttp
import requests

IMAGE_TOKEN_SEP = "\n[IMG]"
ROUTE = "/generate"


async def send_request(url, data, delay=0):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            output = await resp.json()
    return output


async def test_concurrent(args):
    url = f"{args.host}:{args.port}{ROUTE}"

    # Single image test
    if args.single_image:
        prompt = f"<s>[INST]Describe this image in detail.{IMAGE_TOKEN_SEP}[/INST]"
        image_url = "https://picsum.photos/id/237/400/300"
        modality = ["image"]
    # Multiple images test
    else:
        image_urls = [
            "https://picsum.photos/id/237/400/300",
            "https://picsum.photos/id/27/500/500",
        ]
        prompt = f"<s>[INST]How many photos are there? Describe each in a very short sentence.{IMAGE_TOKEN_SEP * len(image_urls)}[/INST]"
        image_url = image_urls
        modality = ["multi-images"]

    response = await send_request(
        url,
        {
            "text": prompt,
            "image_data": image_url,
            "sampling_params": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "modalities": modality,
        },
    )

    print(f"Response: {response}")
    if "text" in response:
        print("\nOutput text:", response["text"])


def test_streaming(args):
    url = f"{args.host}:{args.port}/generate"

    # Single image test
    if args.single_image:
        prompt = f"<s>[INST]Describe this image in detail.{IMAGE_TOKEN_SEP}[/INST]"
        image_data = "https://picsum.photos/id/237/400/300"
        modality = ["image"]
    # Multiple images test
    else:
        image_urls = [
            "https://picsum.photos/id/237/400/300",
            "https://picsum.photos/id/27/500/500",
        ]
        prompt = f"<s>[INST]How many photos are there? Describe each in a very short sentence.{IMAGE_TOKEN_SEP * len(image_urls)}[/INST]"
        image_data = image_urls
        modality = ["multi-images"]

    pload = {
        "text": prompt,
        "image_data": image_data,
        "sampling_params": {"max_new_tokens": 100, "temperature": 0.7, "top_p": 0.9},
        "modalities": modality,
        "stream": True,
    }

    response = requests.post(url, json=pload, stream=True)

    print("Streaming response:")
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
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--single-image",
        action="store_true",
        help="Test with single image instead of multiple images",
    )
    parser.add_argument("--no-stream", action="store_true", help="Don't test streaming")
    args = parser.parse_args()

    asyncio.run(test_concurrent(args))
    if not args.no_stream:
        test_streaming(args)

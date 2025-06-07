"""
Usage:
# Installing latest llava-next: pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# Installing latest sglang.

# Endpoint Service CLI:
python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --port=30000

python3 llama3_llava_server.py

Output:
"Friends posing for a fun photo with a life-sized teddy bear, creating a playful and memorable moment."
"""

import argparse
import asyncio
import copy
import json

import aiohttp
import requests
from llava.conversation import conv_llava_llama_3


async def send_request(url, data, delay=0):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            output = await resp.json()
    return output


async def test_concurrent(args):
    url = f"{args.host}:{args.port}"

    prompt = "<image>\nPlease generate caption towards this image."
    conv_template = copy.deepcopy(conv_llava_llama_3)
    conv_template.append_message(role=conv_template.roles[0], message=prompt)
    conv_template.append_message(role=conv_template.roles[1], message=None)
    prompt_with_template = conv_template.get_prompt()
    response = []
    for i in range(1):
        response.append(
            send_request(
                url + "/generate",
                {
                    "text": prompt_with_template,
                    "image_data": "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg",
                    "sampling_params": {
                        "max_new_tokens": 1024,
                        "temperature": 0,
                        "top_p": 1.0,
                        "presence_penalty": 2,
                        "frequency_penalty": 2,
                        "stop": "<|eot_id|>",
                    },
                },
            )
        )

    rets = await asyncio.gather(*response)
    for ret in rets:
        print(ret["text"])


def test_streaming(args):
    url = f"{args.host}:{args.port}"
    prompt = "<image>\nPlease generate caption towards this image."
    conv_template = copy.deepcopy(conv_llava_llama_3)
    conv_template.append_message(role=conv_template.roles[0], message=prompt)
    conv_template.append_message(role=conv_template.roles[1], message=None)
    prompt_with_template = conv_template.get_prompt()
    pload = {
        "text": prompt_with_template,
        "sampling_params": {
            "max_new_tokens": 1024,
            "temperature": 0,
            "top_p": 1.0,
            "presence_penalty": 2,
            "frequency_penalty": 2,
            "stop": "<|eot_id|>",
        },
        "image_data": "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg",
        "stream": True,
    }
    response = requests.post(
        url + "/generate",
        json=pload,
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

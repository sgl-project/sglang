"""
Run one test prompt.

Usage:
python3 -m sglang.test.send_one
"""

import argparse
import json

import requests


def send_one_prompt(args):
    if args.image:
        args.prompt = (
            "Human: Describe this image in a very short sentence.\n\nAssistant:"
        )
        image_data = "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png"
    else:
        image_data = None

    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": args.prompt,
            "image_data": image_data,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
            },
            "return_logprob": args.return_logprob,
            "stream": args.stream,
        },
        stream=args.stream,
    )

    if args.stream:
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                ret = json.loads(chunk[5:].strip("\n"))
    else:
        ret = response.json()

    latency = ret["meta_info"]["e2e_latency"]

    if "spec_verify_ct" in ret["meta_info"]:
        acc_length = (
            ret["meta_info"]["completion_tokens"] / ret["meta_info"]["spec_verify_ct"]
        )
    else:
        acc_length = 1.0

    speed = ret["meta_info"]["completion_tokens"] / latency

    print(ret["text"])
    print()
    print(f"{acc_length=:.2f}")
    print(f"{speed=:.2f} token/s")

    return acc_length, speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--return-logprob", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:",
    )
    parser.add_argument(
        "--image",
        action="store_true",
    )
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    send_one_prompt(args)

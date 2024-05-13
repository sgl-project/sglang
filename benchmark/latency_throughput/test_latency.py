import argparse
import random
import time

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--backend", type=str, default="srt")
    args = parser.parse_args()

    if args.port is None:
        if args.backend == "srt":
            args.port = 30000
        elif args.backend == "vllm":
            args.port = 21000
        elif args.backend == "lightllm":
            args.port = 22000
        else:
            raise ValueError(f"Invalid backend: {args.backend}")

    url = f"{args.host}:{args.port}"
    a = random.randint(0, 1 << 20)
    max_new_tokens = 256

    tic = time.time()
    if args.backend == "srt":
        response = requests.post(
            url + "/generate",
            json={
                "text": f"{a}, ",
                #"input_ids": [[2] * 256] * 196,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
    elif args.backend == "lightllm":
        response = requests.post(
            url + "/generate",
            json={
                "inputs": f"{a}, ",
                "parameters": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
    elif args.backend == "vllm":
        response = requests.post(
            url + "/generate",
            json={
                "prompt": f"{a}, ",
                "temperature": 0,
                "max_tokens": max_new_tokens,
            },
        )
    latency = time.time() - tic

    ret = response.json()
    print(ret)

    speed = max_new_tokens / latency
    print(f"latency: {latency:.2f} s, speed: {speed:.2f} token/s")

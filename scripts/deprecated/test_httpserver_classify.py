"""
Usage:
python3 -m sglang.launch_server --disable-cuda-graph --model-path /model/llama-classification

python3 test_httpserver_classify.py
"""

import argparse

import numpy as np
import requests


def get_logits(url, prompt):
    response = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 0,
            },
            "return_logprob": True,
        },
    )
    return response.json()["meta_info"]["normalized_prompt_logprob"]


def get_logits_batch(url, prompts):
    response = requests.post(
        url + "/generate",
        json={
            "text": prompts,
            "sampling_params": {
                "max_new_tokens": 0,
            },
            "return_logprob": True,
        },
    )
    ret = response.json()
    logits = np.array(
        list(
            ret[i]["meta_info"]["normalized_prompt_logprob"]
            for i in range(len(prompts))
        )
    )
    return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"

    # A single request
    prompt = "This is a test prompt.<|eot_id|>"
    logits = get_logits(url, prompt)
    print(f"{logits=}")

    # A batch of requests
    prompts = [
        "This is a test prompt.<|eot_id|>",
        "This is another test prompt.<|eot_id|>",
        "This is a long long long long test prompt.<|eot_id|>",
    ]
    logits = get_logits_batch(url, prompts)
    print(f"{logits=}")

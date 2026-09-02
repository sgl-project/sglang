#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
from pathlib import Path

import requests

PROMPTS = [
    f"Write a detailed explanation of topic {index}, using several paragraphs and concrete examples."
    for index in range(50)
]


def _generate_sample(args, index: int, prompt: str, watermarked: bool):
    headers = {"X-SGLang-Watermark": "textseal"} if watermarked else {}
    response = requests.post(
        args.base_url + "/v1/completions",
        headers=headers,
        json={
            "model": args.model,
            "prompt": prompt,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": args.max_tokens,
            "seed": 20260901 + index,
        },
        timeout=300,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "prompt_index": index,
        "watermarked": watermarked,
        "text": body["choices"][0]["text"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "model": args.model,
        "seed": 20260901 + index,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()

    jobs = [
        (index, prompt, watermarked)
        for index, prompt in enumerate(PROMPTS)
        for watermarked in (False, True)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrency
    ) as executor:
        records = executor.map(
            lambda job: _generate_sample(args, *job),
            jobs,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
            output.flush()


if __name__ == "__main__":
    main()

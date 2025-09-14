"""
Batch the same prompt in random batch sizes, and test if the results are consistent across different trials.

Usage:
python3 -m sglang.test.test_deterministic --n-trials <numer_of_trials>
"""

import argparse
import dataclasses
import json
import random

import numpy as np
import requests


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 100
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    return_logprob: bool = False
    prompt: str = "Tell me about Richard Feynman: "
    stream: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--n-trials", type=int, default=50)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument(
            "--max-new-tokens", type=int, default=BenchArgs.max_new_tokens
        )
        parser.add_argument(
            "--frequency-penalty", type=float, default=BenchArgs.frequency_penalty
        )
        parser.add_argument(
            "--presence-penalty", type=float, default=BenchArgs.presence_penalty
        )
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument("--prompt", type=str, default=BenchArgs.prompt)
        parser.add_argument("--stream", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def send_one_prompt(args):

    prompt = args.prompt
    batch_size = random.randint(1, 256)
    prompt = [prompt] * batch_size

    json_data = {
        "text": prompt,
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
        },
        "return_logprob": args.return_logprob,
        "stream": args.stream,
    }

    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=json_data,
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

    ret = ret[0]

    if response.status_code != 200:
        print(ret)
        return 0, 0

    return ret["text"], batch_size


def test_deterministic(args):
    # First do some warmups
    for i in range(3):
        send_one_prompt(args)

    texts = []
    for i in range(args.n_trials):
        text, batch_size = send_one_prompt(args)
        text = text.replace("\n", " ")
        print(f"Trial {i} with batch size {batch_size}: {text}")
        texts.append(text)

    print(f"Total samples: {len(texts)}, Unique samples: {len(set(texts))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    test_deterministic(args)

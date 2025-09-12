"""
Send the same prompt for multiple times and test if the results are deterministic.

Usage:
python3 -m sglang.test.test_deterministic --n-trials <numer_of_trials>
"""

import argparse
import dataclasses
import json

import numpy as np
import requests


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 512
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
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
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
    if args.batch_size > 1:
        prompt = [prompt] * args.batch_size

    json_data = {
        "text": prompt,
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
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

    if args.batch_size > 1:
        ret = ret[0]

    if response.status_code != 200:
        print(ret)
        return 0, 0

    latency = ret["meta_info"]["e2e_latency"]
    speed = ret["meta_info"]["completion_tokens"] / latency

    return ret["text"], speed


def test_deterministic(args):
    speeds = []
    texts = []
    for i in range(args.n_trials):
        text, speed = send_one_prompt(args)
        print(f"Trial {i}: {speed=:.2f} token/s, {text[:50]}")
        speeds.append(speed)
        texts.append(text)

    pass_test = True
    for i in range(len(texts)):
        text = texts[i]
        if text != texts[0]:
            print(
                f"Test failed: Output of trial 0: {texts[0][:50]}\n Output of trial {i}: {text[:50]}\n"
            )
            pass_test = False
            break

    if pass_test:
        print("Test passed for all trials")
    print(f"Average speed: {np.mean(speeds)=:.2f} token/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    test_deterministic(args)

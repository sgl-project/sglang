"""
Batch the same prompt in random batch sizes, and test if the results are consistent across different trials.

Usage:
python3 -m sglang.test.test_deterministic --n-trials <numer_of_trials> --test-mode <single|mixed|prefix> --profile
"""

import argparse
import dataclasses
import json
import os
import random
from typing import List

import requests

from sglang.profiler import run_profile

PROMPT_1 = "Tell me about Richard Feynman: "
PROMPT_2 = "Generate 1000 random numbers. Go directly into it, don't say Sure and don't say here are numbers. Just start with a number."
dirpath = os.path.dirname(__file__)
with open(os.path.join(dirpath, "long_prompt.txt"), "r") as f:
    LONG_PROMPT = f.read()


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    temperature: float = 0.0
    sampling_seed: int = 42
    max_new_tokens: int = 100
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    return_logprob: bool = False
    stream: bool = False
    profile: bool = False
    profile_steps: int = 3
    profile_by_stage: bool = False
    test_mode: str = "single"
    n_trials: int = 50
    n_start: int = 1

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--n-trials", type=int, default=BenchArgs.n_trials)
        parser.add_argument("--n-start", type=int, default=BenchArgs.n_start)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument(
            "--sampling-seed", type=int, default=BenchArgs.sampling_seed
        )
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
        parser.add_argument("--stream", action="store_true")
        parser.add_argument(
            "--test-mode",
            type=str,
            default=BenchArgs.test_mode,
            choices=["single", "mixed", "prefix"],
        )
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def send_single(
    args,
    batch_size: int,
    profile: bool = False,
    profile_steps: int = 3,
    profile_by_stage: bool = False,
):

    base_url = f"http://{args.host}:{args.port}"
    prompt = [PROMPT_1] * batch_size

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

    if args.sampling_seed is not None:
        # sglang server cannot parse None value for sampling_seed
        json_data["sampling_params"]["sampling_seed"] = args.sampling_seed

    if profile:
        run_profile(
            base_url, profile_steps, ["CPU", "GPU"], None, None, profile_by_stage
        )

    response = requests.post(
        f"{base_url}/generate",
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
        return -1

    return ret["text"]


def send_mixed(args, batch_size: int):
    num_long_prompt = 0 if batch_size <= 10 else random.randint(1, 10)
    num_prompt_1 = random.randint(1, batch_size - num_long_prompt)
    num_prompt_2 = batch_size - num_prompt_1 - num_long_prompt

    json_data = {
        "text": [PROMPT_1] * num_prompt_1
        + [PROMPT_2] * num_prompt_2
        + [LONG_PROMPT] * num_long_prompt,
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
        },
        "return_logprob": args.return_logprob,
        "stream": args.stream,
    }

    if args.sampling_seed is not None:
        json_data["sampling_params"]["sampling_seed"] = args.sampling_seed

    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=json_data,
        stream=args.stream,
    )
    ret = response.json()
    if response.status_code != 200:
        print(ret)
        return -1, -1, -1

    prompt_1_ret = [ret[i]["text"] for i in range(num_prompt_1)]
    prompt_2_ret = [
        ret[i]["text"] for i in range(num_prompt_1, num_prompt_1 + num_prompt_2)
    ]
    long_prompt_ret = [
        ret[i]["text"]
        for i in range(
            num_prompt_1 + num_prompt_2, num_prompt_1 + num_prompt_2 + num_long_prompt
        )
    ]

    return prompt_1_ret, prompt_2_ret, long_prompt_ret


def send_prefix(args, batch_size: int, prompts: List[str]):
    requests.post(f"http://{args.host}:{args.port}/flush_cache")

    batch_data = []
    sampled_indices = []
    for _ in range(batch_size):
        sampled_index = random.randint(0, len(prompts) - 1)
        sampled_indices.append(sampled_index)
        batch_data.append(prompts[sampled_index])

    json_data = {
        "text": batch_data,
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
        },
        "return_logprob": args.return_logprob,
        "stream": args.stream,
    }

    if args.sampling_seed is not None:
        json_data["sampling_params"]["sampling_seed"] = args.sampling_seed

    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=json_data,
        stream=args.stream,
    )
    ret = response.json()
    if response.status_code != 200:
        print(ret)
        return -1, -1, -1

    ret_dict = {i: [] for i in range(len(prompts))}
    for i in range(batch_size):
        ret_dict[sampled_indices[i]].append(ret[i]["text"])

    return ret_dict


def test_deterministic(args):
    # First do some warmups
    for i in range(3):
        send_single(args, 16, args.profile)

    if args.test_mode == "single":
        # In single mode, we test the deterministic behavior by sending the same prompt in batch sizes ranging from 1 to n_trials.
        texts = []
        for i in range(1, args.n_trials + 1):
            batch_size = i
            text = send_single(args, batch_size, args.profile)
            text = text.replace("\n", " ")
            print(f"Trial {i} with batch size {batch_size}: {text}")
            texts.append(text)

        print(f"Total samples: {len(texts)}, Unique samples: {len(set(texts))}")
        return [len(set(texts))]

    elif args.test_mode == "mixed":
        # In mixed mode, we send a mixture of two short prompts and one long prompt in the same batch with batch size ranging from 1 to n_trials.
        output_prompt_1 = []
        output_prompt_2 = []
        output_long_prompt = []
        for i in range(1, args.n_trials + 1):
            batch_size = i
            ret_prompt_1, ret_prompt_2, ret_long_prompt = send_mixed(args, batch_size)
            output_prompt_1.extend(ret_prompt_1)
            output_prompt_2.extend(ret_prompt_2)
            output_long_prompt.extend(ret_long_prompt)

            print(
                f"Testing Trial {i} with batch size {batch_size}, number of prompt 1: {len(ret_prompt_1)}, number of prompt 2: {len(ret_prompt_2)}, number of long prompt: {len(ret_long_prompt)}"
            )

        print(
            f"Prompt 1: total samples: {len(output_prompt_1)}, Unique samples: {len(set(output_prompt_1))}"
        )
        print(
            f"Prompt 2: total samples: {len(output_prompt_2)}, Unique samples: {len(set(output_prompt_2))}"
        )
        print(
            f"Long prompt: total samples: {len(output_long_prompt)}, Unique samples: {len(set(output_long_prompt))}"
        )

        return [
            len(set(output_prompt_1)),
            len(set(output_prompt_2)),
            len(set(output_long_prompt)),
        ]

    elif args.test_mode == "prefix":
        # In prefix mode, we create prompts from the same long prompt, with different lengths of common prefix.
        len_prefix = [1, 511, 2048, 4097]
        num_prompts = len(len_prefix)
        outputs = {i: [] for i in range(4)}
        prompts = [LONG_PROMPT[: len_prefix[i]] for i in range(4)]
        for i in range(args.n_start, args.n_start + args.n_trials):
            batch_size = i
            ret_dict = send_prefix(args, batch_size, prompts)
            msg = f"Testing Trial {i} with batch size {batch_size},"
            for i in range(num_prompts):
                msg += f" # prefix length {len_prefix[i]}: {len(ret_dict[i])},"
            print(msg)
            for i in range(num_prompts):
                outputs[i].extend(ret_dict[i])

        for i in range(num_prompts):
            print(
                f"Prompt {i} with prefix length {len_prefix[i]}: total samples: {len(outputs[i])}, Unique samples: {len(set(outputs[i]))}"
            )

        results = []
        for i in range(num_prompts):
            results.append(len(set(outputs[i])))
        return results

    else:
        raise ValueError(f"Invalid test mode: {args.test_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    test_deterministic(args)

"""
Run one test prompt.

Usage:
python3 -m sglang.test.send_one
python3 -m sglang.test.send_one --profile --profile-steps 5
python3 -m sglang.test.send_one --profile --profile-by-stage
"""

import argparse
import dataclasses
import json

import requests
import tabulate

from sglang.profiler import run_profile


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json: bool = False
    return_logprob: bool = False
    prompt: str = (
        "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:"
    )
    image: bool = False
    many_images: bool = False
    stream: bool = False
    profile: bool = False
    profile_steps: int = 3
    profile_by_stage: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
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
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument("--prompt", type=str, default=BenchArgs.prompt)
        parser.add_argument("--image", action="store_true")
        parser.add_argument("--many-images", action="store_true")
        parser.add_argument("--stream", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def send_one_prompt(args):
    base_url = f"http://{args.host}:{args.port}"

    if args.image:
        args.prompt = (
            "Human: Describe this image in a very short sentence.\n\nAssistant:"
        )
        image_data = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
    elif args.many_images:
        args.prompt = (
            "Human: I have one reference image and many images."
            "Describe their relationship in a very short sentence.\n\nAssistant:"
        )
        image_data = [
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
        ]
    else:
        image_data = None

    prompt = args.prompt

    if args.json:
        prompt = (
            "Human: What is the capital of France and how is that city like. "
            "Give me 3 trivial information about that city. "
            "Write in a format of json.\nAssistant:"
        )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    if args.batch_size > 1:
        prompt = [prompt] * args.batch_size

    json_data = {
        "text": prompt,
        "image_data": image_data,
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "json_schema": json_schema,
            "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
        },
        "return_logprob": args.return_logprob,
        "stream": args.stream,
    }

    # Run profiler if requested
    if args.profile:
        print(f"Running profiler with {args.profile_steps} steps...")
        run_profile(
            base_url,
            args.profile_steps,
            ["CPU", "GPU"],
            None,
            None,
            args.profile_by_stage,
        )

    response = requests.post(
        f"{base_url}/generate",
        json=json_data,
        stream=args.stream,
    )

    if args.stream:
        last_len = 0
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                ret = json.loads(chunk[5:].strip("\n"))
                chunk_str = ret["text"][last_len:]
                last_len = len(ret["text"])
                print(chunk_str, end="", flush=True)
    else:
        ret = response.json()

    if args.batch_size > 1:
        ret = ret[0]

    if response.status_code != 200:
        print(ret)
        return 0, 0

    if "spec_verify_ct" in ret["meta_info"] and ret["meta_info"]["spec_verify_ct"] > 0:
        acc_length = (
            ret["meta_info"]["completion_tokens"] / ret["meta_info"]["spec_verify_ct"]
        )
    else:
        acc_length = 1.0

    latency = ret["meta_info"]["e2e_latency"]
    speed = ret["meta_info"]["completion_tokens"] / latency
    tokens = ret["meta_info"]["completion_tokens"]

    if not args.stream:
        print(ret["text"])

    print()
    headers = ["Latency (s)", "Tokens", "Acc Length", "Speed (token/s)"]
    rows = [[f"{latency:.3f}", f"{tokens}", f"{acc_length:.3f}", f"{speed:.2f}"]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="pretty")
    print(msg)

    return acc_length, speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    send_one_prompt(args)

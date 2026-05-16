"""
Run one test prompt using the /generate endpoint with chat template applied client-side.

This gives access to rich meta_info (spec decoding stats, server-side latency, etc.)
while still using the chat message format. Results are appended to a JSONL file.

Usage:
python3 -m sglang.test.send_1
python3 -m sglang.test.send_1 --stream
python3 -m sglang.test.send_1 --model-sampling
python3 -m sglang.test.send_1 --output results.jsonl
python3 -m sglang.test.send_1 --profile --profile-steps 5
"""

import argparse
import asyncio
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
import orjson
from rich.console import Console
from rich.table import Table

from sglang.profiler import run_profile

TZ_EASTERN = ZoneInfo("America/New_York")
TZ_PACIFIC = ZoneInfo("America/Los_Angeles")
TIME_FMT = "%Y-%m-%d %H:%M:%S %Z"

IMAGE_URL = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"


def get_timestamps() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "utc": now.strftime(TIME_FMT),
        "eastern": now.astimezone(TZ_EASTERN).strftime(TIME_FMT),
        "pacific": now.astimezone(TZ_PACIFIC).strftime(TIME_FMT),
    }


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    different_prompts: bool = False
    seed: Optional[int] = None
    temperature: float = 0.0
    max_new_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json_mode: bool = False
    return_logprob: bool = False
    prompt: str = "Give me a fully functional FastAPI server. Show the python code."
    image: bool = False
    many_images: bool = False
    stream: bool = False
    model_sampling: bool = False
    profile: bool = False
    profile_steps: int = 3
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    output: str = "send_1_results.jsonl"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
        parser.add_argument(
            "--different-prompts",
            action="store_true",
            default=BenchArgs.different_prompts,
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
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
        parser.add_argument("--json-mode", action="store_true")
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument("--prompt", type=str, default=BenchArgs.prompt)
        parser.add_argument("--image", action="store_true")
        parser.add_argument("--many-images", action="store_true")
        parser.add_argument("--stream", action="store_true")
        parser.add_argument(
            "--model-sampling",
            action="store_true",
            help="Use sampling params from the model's generation_config.json "
            "(temperature, top_k, top_p, repetition_penalty). "
            "Overrides --temperature.",
        )
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix", type=str, default=BenchArgs.profile_prefix
        )
        parser.add_argument("--output", type=str, default=BenchArgs.output)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


async def get_model_info(session: aiohttp.ClientSession, base_url: str) -> dict:
    async with session.get(f"{base_url}/model_info") as resp:
        return await resp.json()


def get_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def get_model_sampling_params(model_path: str) -> dict:
    """Load sampling params from the model's generation_config.json."""
    from transformers import GenerationConfig

    gc = GenerationConfig.from_pretrained(model_path)
    config = gc.to_dict()

    keys = ["temperature", "top_k", "top_p", "repetition_penalty", "min_p"]
    return {k: config[k] for k in keys if config.get(k) is not None}


def apply_chat_template(tokenizer, prompt: str, image_data=None) -> str:
    if image_data:
        content = [{"type": "text", "text": prompt}]
        images = image_data if isinstance(image_data, list) else [image_data]
        for url in images:
            content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content = prompt

    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def print_meta_info(meta: dict, timestamps: dict, model_name: str):
    console = Console()

    latency = meta["e2e_latency"]
    tokens = meta["completion_tokens"]
    prompt_tokens = meta["prompt_tokens"]
    speed = tokens / latency
    spec_verify_ct = meta.get("spec_verify_ct", 0)
    acc_length = tokens / spec_verify_ct if spec_verify_ct > 0 else 1.0

    # Timestamp
    console.print()
    console.print(
        f"[bold yellow]{timestamps['eastern']}[/]  /  [bold yellow]{timestamps['pacific']}[/]"
    )

    # Performance
    perf = Table(title="Performance", show_header=True, header_style="bold cyan")
    perf.add_column("Metric", style="bold")
    perf.add_column("Value", justify="right")
    perf.add_row("E2E Latency", f"{latency:.3f} s")
    perf.add_row("Prompt Tokens", str(prompt_tokens))
    perf.add_row("Completion Tokens", str(tokens))
    perf.add_row("Speed", f"{speed:.2f} tok/s")
    perf.add_row("Cached Tokens", str(meta["cached_tokens"]))
    finish = meta["finish_reason"]
    detail = finish.get("length", finish.get("matched", ""))
    perf.add_row("Finish Reason", f"{finish['type']} ({detail})")
    console.print(perf)

    # Speculative decoding
    if spec_verify_ct > 0:
        spec = Table(
            title="Speculative Decoding",
            show_header=True,
            header_style="bold magenta",
        )
        spec.add_column("Metric", style="bold")
        spec.add_column("Value", justify="right")
        spec.add_row("Accept Length", f"{acc_length:.3f}")
        spec.add_row("Accept Rate", f"{meta['spec_accept_rate']:.2%}")
        spec.add_row("Verify Steps", str(spec_verify_ct))
        spec.add_row("Accepted Tokens", str(meta["spec_accept_token_num"]))
        spec.add_row("Draft Tokens", str(meta["spec_draft_token_num"]))
        spec.add_row("Total Retractions", str(meta["total_retractions"]))
        histogram = meta.get("spec_accept_histogram")
        if histogram:
            hist_str = " | ".join(f"{i}:{n}" for i, n in enumerate(histogram))
            spec.add_row("Accept Histogram", hist_str)
        console.print(spec)

    # Request info
    info = Table(title="Request Info", show_header=True, header_style="bold green")
    info.add_column("Metric", style="bold")
    info.add_column("Value", justify="right")
    info.add_row("Model", model_name)
    info.add_row("Request ID", meta["id"])
    info.add_row("Weight Version", str(meta["weight_version"]))
    if meta.get("dp_rank") is not None:
        info.add_row("DP Rank", str(meta["dp_rank"]))
    console.print(info)


async def send_one_prompt(args: BenchArgs):
    base_url = f"http://{args.host}:{args.port}"
    console = Console()

    async with aiohttp.ClientSession() as session:
        model_info = await get_model_info(session, base_url)
        model_path = model_info["tokenizer_path"]
        tokenizer = get_tokenizer(model_path)

        # Build sampling params
        sampling_params = {
            "sampling_seed": args.seed,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
        }

        if args.model_sampling:
            model_params = get_model_sampling_params(model_path)
            sampling_params.update(model_params)
            console.print(f"[bold green]Using model sampling params:[/] {model_params}")

        if args.json_mode:
            sampling_params["json_schema"] = "$$ANY$$"

        # Build prompt
        image_data = None
        if args.image:
            args.prompt = "Describe this image in a very short sentence."
            image_data = IMAGE_URL
        elif args.many_images:
            args.prompt = (
                "I have one reference image and many images. "
                "Describe their relationship in a very short sentence."
            )
            image_data = [IMAGE_URL] * 4
        elif args.json_mode:
            args.prompt = (
                "What is the capital of France and how is that city like. "
                "Give me 3 trivial information about that city. "
                "Write in a format of json."
            )

        if args.batch_size > 1 and args.different_prompts:
            text = [
                apply_chat_template(tokenizer, f"Test case {i+1}: {args.prompt}")
                for i in range(args.batch_size)
            ]
        elif args.batch_size > 1:
            text = [apply_chat_template(tokenizer, args.prompt)] * args.batch_size
        else:
            text = apply_chat_template(tokenizer, args.prompt, image_data)

        json_data = {
            "text": text,
            "image_data": image_data,
            "sampling_params": sampling_params,
            "return_logprob": args.return_logprob,
            "stream": args.stream,
        }

        # Profiler
        if args.profile:
            print(f"Running profiler with {args.profile_steps} steps...")
            run_profile(
                url=base_url,
                num_steps=args.profile_steps,
                activities=["CPU", "GPU"],
                profile_by_stage=args.profile_by_stage,
                profile_prefix=args.profile_prefix,
            )

        # Send request
        async with session.post(f"{base_url}/generate", json=json_data) as response:
            if args.stream:
                last_len = 0
                ret = None
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    # SSE events are delimited by \n\n
                    while "\n\n" in buffer:
                        event_raw, buffer = buffer.split("\n\n", 1)
                        for line in event_raw.splitlines():
                            line = line.strip()
                            if not line or line.startswith(":"):
                                continue
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            ret = orjson.loads(payload)
                            print(ret["text"][last_len:], end="", flush=True)
                            last_len = len(ret["text"])
                print()
            else:
                data = await response.read()
                ret = orjson.loads(data)
                if args.batch_size > 1:
                    ret = ret[0]
                print(ret["text"])

    # Display results
    timestamps = get_timestamps()
    model_name = model_info.get("model_path", "unknown")
    print_meta_info(ret["meta_info"], timestamps, model_name)

    # Append full response to JSONL
    record = ret.copy()
    record["timestamps"] = timestamps
    output_path = Path(args.output)
    with open(output_path, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")
    console.print(f"\n[dim]Results appended to {output_path}[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = BenchArgs.from_cli_args(parser.parse_args())
    asyncio.run(send_one_prompt(args))

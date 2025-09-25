"""
Bench the sglang-hosted vLM with benchmark MMMU

Usage:
    Host the VLM: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --port 30000

    Benchmark: python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 16

The eval output will be logged
"""

import argparse
import asyncio
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import aiohttp
import openai
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm

from sglang.test.test_utils import add_common_sglang_args_and_parse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


@dataclass
class RequestFuncOutput:
    generated_text: List[str] = field(default_factory=list)
    prompt_len: List[int] = field(default_factory=list)
    output_len: List[int] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    ttft: List[float] = field(default_factory=list)
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies

    success: bool = False
    error: str = ""


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


def _get_prefix_suffix(prompt: str) -> Tuple[str, str]:
    """Split the prompt into prefix and suffix."""
    prefix = prompt.split("<")[0]
    suffix = prompt.split(">", 1)[1]
    return prefix, suffix


async def process_sample(
    client: Any, sample: dict, sampling_params: dict, lora_path: Optional[str] = None
) -> Tuple[dict, str]:
    """Send a single sample to the LLM and return (sample, response)."""
    prompt = sample["final_input_prompt"]
    prefix, suffix = _get_prefix_suffix(prompt)
    image = sample["image"]
    assert image is not None
    image_path = sample["image_path"]
    extra_body = None if lora_path is None else {"lora_path": lora_path}
    response = await client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix},
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": suffix},
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=sampling_params["max_new_tokens"],
        max_tokens=sampling_params["max_new_tokens"],
        extra_body=extra_body,
    )
    return sample, response.choices[0].message.content


async def process_sample_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: Any,
    sample: dict,
    sampling_params: dict,
    lora_path: Optional[str] = None,
) -> Tuple[dict, str]:
    """Wrap process_sample with a semaphore for concurrency control."""
    async with semaphore:
        return await process_sample(client, sample, sampling_params, lora_path)


async def eval_mmmu(args) -> None:
    """Main evaluation loop with concurrency control."""
    eval_args = EvalArgs.from_cli_args(args)
    sampling_params = get_sampling_params(eval_args)
    samples = prepare_samples(eval_args)
    lora_path = eval_args.lora_path
    answer_dict = {}
    out_samples = {}
    client = openai.AsyncOpenAI(
        api_key="sk",
        base_url=f"http://127.0.0.1:{args.port}/v1",
        timeout=20 * 60 * 60,
    )
    start = time.perf_counter()
    base_url = f"http://127.0.0.1:{args.port}"

    if args.profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=f"{base_url}/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

        samples = samples[: args.profile_number]

    if args.concurrency == 1:
        # For concurrency == 1, run in sequential mode to ensure consistent order
        # this is mainly for profiling
        for sample in tqdm(samples):
            _, response = await process_sample(
                client, sample, sampling_params, lora_path
            )
            sample["original_response"] = response
            answer = (
                re.search(args.response_answer_regex, response)
                if response is not None
                else None
            )
            process_result(
                answer.group(1).strip() if answer else response,
                sample,
                answer_dict,
                out_samples,
            )
    else:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            process_sample_with_semaphore(
                semaphore, client, sample, sampling_params, lora_path
            )
            for sample in samples
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            sample, response = await coro
            sample["original_response"] = response
            answer = (
                re.search(args.response_answer_regex, response)
                if response is not None
                else None
            )
            process_result(
                answer.group(1).strip() if answer else response,
                sample,
                answer_dict,
                out_samples,
            )

    if args.profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=f"{base_url}/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    print(f"Benchmark time: {time.perf_counter() - start}")
    args.output_path = "./answer_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(
        model_answer_path=args.output_path,
        answer_dict=answer_dict,
        eval_output_path="./val_sglang.json",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    args = add_common_sglang_args_and_parse(parser)
    return args


def main():
    args = parse_args()
    asyncio.run(eval_mmmu(args))


if __name__ == "__main__":
    main()

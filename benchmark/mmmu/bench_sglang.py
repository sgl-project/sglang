"""
Bench the sglang-hosted vLM with benchmark MMMU

Usage:
    Host the VLM: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl --port 30000

    Benchmark: python benchmark/mmmu/bench_sglang.py --port 30000

The eval output will be logged
"""

import argparse
import asyncio
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List

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


async def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    out_samples = dict()

    sampling_params = get_sampling_params(eval_args)

    samples = prepare_samples(eval_args)

    answer_dict = {}

    # had to use an openai server, since SglImage doesn't support image data
    base_url = f"http://127.0.0.1:{args.port}"
    client = openai.Client(api_key="sk", base_url=f"{base_url}/v1")

    start = time.time()

    if args.profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=f"{base_url}/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    if args.profile:
        samples = samples[: args.profile_number]

    for i, sample in enumerate(tqdm(samples)):
        prompt = sample["final_input_prompt"]
        prefix = prompt.split("<")[0]
        suffix = prompt.split(">")[1]
        image = sample["image"]
        assert image is not None
        image_path = sample["image_path"]
        # TODO: batch

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prefix,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_path},
                        },
                        {
                            "type": "text",
                            "text": suffix,
                        },
                    ],
                }
            ],
            temperature=0,
            max_completion_tokens=sampling_params["max_new_tokens"],
            max_tokens=sampling_params["max_new_tokens"],
        )
        response = response.choices[0].message.content
        process_result(response, sample, answer_dict, out_samples)

    if args.profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=f"{base_url}/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    print(f"Benchmark time: {time.time() - start}")

    args.output_path = f"./val_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    args = add_common_sglang_args_and_parse(parser)
    args = parser.parse_args()
    asyncio.run(eval_mmmu(args))

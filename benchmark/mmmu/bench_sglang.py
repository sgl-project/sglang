"""
Bench the sglang-hosted vLM with benchmark MMMU

Usage:
    Host the VLM: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl --port 30000

    Benchmark: python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 16

The eval output will be logged
"""

import argparse
import asyncio
import time
from typing import Any, Tuple

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


def _get_prefix_suffix(prompt: str) -> Tuple[str, str]:
    """Split the prompt into prefix and suffix."""
    prefix = prompt.split("<")[0]
    suffix = prompt.split(">", 1)[1]
    return prefix, suffix


async def process_sample(
    client: Any, sample: dict, sampling_params: dict
) -> Tuple[dict, str]:
    """Send a single sample to the LLM and return (sample, response)."""
    prompt = sample["final_input_prompt"]
    prefix, suffix = _get_prefix_suffix(prompt)
    image = sample["image"]
    assert image is not None
    image_path = sample["image_path"]
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
    )
    return sample, response.choices[0].message.content


async def process_sample_with_semaphore(
    semaphore: asyncio.Semaphore, client: Any, sample: dict, sampling_params: dict
) -> Tuple[dict, str]:
    """Wrap process_sample with a semaphore for concurrency control."""
    async with semaphore:
        return await process_sample(client, sample, sampling_params)


async def eval_mmmu(args) -> None:
    """Main evaluation loop with concurrency control."""
    eval_args = EvalArgs.from_cli_args(args)
    sampling_params = get_sampling_params(eval_args)
    samples = prepare_samples(eval_args)
    answer_dict = {}
    out_samples = {}
    client = openai.AsyncOpenAI(
        api_key="sk", base_url=f"http://127.0.0.1:{args.port}/v1"
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.time()
    tasks = [
        process_sample_with_semaphore(semaphore, client, sample, sampling_params)
        for sample in samples
    ]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        sample, response = await coro
        process_result(response, sample, answer_dict, out_samples)
    print(f"Benchmark time: {time.time() - start}")
    args.output_path = f"./val_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Max concurrent OpenAI calls"
    )
    args = add_common_sglang_args_and_parse(parser)
    return args


def main():
    args = parse_args()
    asyncio.run(eval_mmmu(args))


if __name__ == "__main__":
    main()

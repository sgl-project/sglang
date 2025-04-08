"""
Bench the sglang-hosted vLM with benchmark MMMU

Usage:
    Host the VLM: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl --port 30000

    Benchmark: python benchmark/mmmu/bench_sglang.py --port 30000

The eval output will be logged
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
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


def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    sampling_params = get_sampling_params(eval_args)

    samples = prepare_samples(eval_args)

    # had to use an openai server, since SglImage doesn't support image data
    client = openai.Client(api_key="sk", base_url=f"http://127.0.0.1:{args.port}/v1")
    generate_url = f"http://127.0.0.1:{args.port}/generate"

    def process_sample(
        sample, eval_args, generate_url, client, sampling_params, answer_dict
    ):
        prompt = sample["final_input_prompt"]
        prefix = prompt.split("<")[0]
        suffix = prompt.split(">")[1]
        image = sample["image"]
        assert image is not None
        image_path = sample["image_path"]

        # Make API call based on mode
        if eval_args.query_format == "chat":
            response = call_chat_completion(
                sampling_params, client, prefix, suffix, image_path
            )
        else:
            response = call_generate(
                sampling_params, generate_url, prefix, suffix, image_path
            )

        # Process the result and return
        out_samples = {}
        answer_dict = {}
        process_result(response, sample, answer_dict, out_samples)
        return out_samples, answer_dict

    # Main processing function
    def run_benchmark(samples, eval_args, generate_url, client, sampling_params):
        start = time.time()
        out_samples = dict()
        global_answer_dict = {}

        # Define max workers (adjust based on your system)
        max_workers = min(16, len(samples))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(
                    process_sample,
                    sample,
                    eval_args,
                    generate_url,
                    client,
                    sampling_params,
                    {},
                ): sample
                for sample in samples
            }

            # Process results as they complete with progress bar
            for future in tqdm(
                as_completed(future_to_sample),
                total=len(samples),
                desc="Processing samples",
            ):
                task_out_samples, task_answer_dict = future.result()
                out_samples.update(task_out_samples)
                global_answer_dict.update(task_answer_dict)
        print(f"Benchmark time: {time.time() - start}")

        args.output_path = f"./val_sglang.json"
        save_json(args.output_path, out_samples)
        eval_result(model_answer_path=args.output_path, answer_dict=global_answer_dict)

    run_benchmark(samples, eval_args, generate_url, client, sampling_params)


def call_generate(
    sampling_params: dict, generate_url: str, prefix: str, suffix: str, image_path: str
):
    data = {
        "model": "pixtral",
        "text": f"<s>[INST]{prefix}[IMG]{suffix}[/INST]",
        "sampling_params": {
            "max_new_tokens": sampling_params["max_new_tokens"],
            "temperature": sampling_params["temperature"],
        },
        "image_data": [image_path],
        "modalities": ["image"],
    }
    response = httpx.post(generate_url, json=data, timeout=60.0)
    response = response.json()["text"]
    return response


def call_chat_completion(
    sampling_params: dict,
    client: openai.Client,
    prefix: str,
    suffix: str,
    image_path: str,
):
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
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    args = add_common_sglang_args_and_parse(parser)
    args = parser.parse_args()
    assert args.query_format in ["default", "mistral"], "Invalid query format"

    eval_mmmu(args)

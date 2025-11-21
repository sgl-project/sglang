import argparse
import ast
import asyncio
import re
import time
from typing import Optional

import numpy as np

import sglang as sgl
from sglang.srt.utils import get_or_create_event_loop
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


async def concurrent_generate(engine, prompts, sampling_param):
    tasks = []
    for prompt in prompts:
        tasks.append(asyncio.create_task(engine.async_generate(prompt, sampling_param)))

    outputs = await asyncio.gather(*tasks)
    return outputs


def run_eval(args):
    # Select backend
    engine = sgl.Engine(model_path=args.model_path, log_level="error")

    if args.local_data_path is None:
        # Read data
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        filename = download_and_cache_file(url)
    else:
        filename = args.local_data_path

    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    # construct the prompts
    prompts = []
    for i, arg in enumerate(arguments):
        q = arg["question"]
        prompt = few_shot_examples + q
        prompts.append(prompt)

    sampling_param = {
        "stop": ["Question", "Assistant:", "<|separator|>"],
        "max_new_tokens": 512,
        "temperature": 0,
    }

    # Run requests
    tic = time.perf_counter()

    loop = get_or_create_event_loop()

    outputs = loop.run_until_complete(
        concurrent_generate(engine, prompts, sampling_param)
    )

    # End requests
    latency = time.perf_counter() - tic

    # Shutdown the engine
    engine.shutdown()

    # Parse output
    preds = []

    for output in outputs:
        preds.append(get_answer_value(output["text"]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        output["meta_info"]["completion_tokens"] for output in outputs
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    return {
        "accuracy": acc,
        "latency": latency,
        "output_throughput": output_throughput,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--local-data-path", type=Optional[str], default=None)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-questions", type=int, default=200)
    args = parser.parse_args()
    metrics = run_eval(args)

import argparse
import ast
import asyncio
import json
import os
import re
import time
from typing import Any, Tuple

import numpy as np
import openai
from tqdm import tqdm

from sglang.test.test_utils import add_common_sglang_args_and_parse
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def format_few_shot_examples(lines, k):
    messages = []
    for i in range(k):
        question = "Question: " + lines[i]["question"] + "\nAnswer:"
        answer = lines[i]["answer"]
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


async def process_sample(
    client: Any, question: str, few_shot_examples: list, model: str
) -> Tuple[str, int, int]:
    messages = few_shot_examples + [{"role": "user", "content": question}]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=512,
        stop=["Question:", "\n\nQuestion"],
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


async def process_sample_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: Any,
    question: str,
    few_shot_examples: list,
    model: str,
) -> Tuple[str, int, int]:
    async with semaphore:
        return await process_sample(client, question, few_shot_examples, model)


async def eval_gsm8k(args) -> None:
    data_path = args.data_path
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    few_shot_examples = format_few_shot_examples(lines, args.num_shots)
    questions = []
    labels = []

    for i in range(args.num_questions):
        question = "Question: " + lines[i]["question"] + "\nAnswer:"
        label = get_answer_value(lines[i]["answer"])
        assert label != INVALID, f"Invalid label for question {i}"
        questions.append(question)
        labels.append(label)

    if args.backend.startswith("gpt-"):
        client = openai.AsyncOpenAI(timeout=20 * 60 * 60)
        model = args.backend
    else:
        host = args.host.replace("http://", "").replace("https://", "")
        client = openai.AsyncOpenAI(
            api_key="sk", base_url=f"http://{host}:{args.port}/v1", timeout=20 * 60 * 60
        )
        model = "default"

    start = time.perf_counter()

    if args.parallel == 1:
        responses = []
        prompt_tokens = []
        completion_tokens = []
        for question in tqdm(questions):
            resp, pt, ct = await process_sample(
                client, question, few_shot_examples, model
            )
            responses.append(resp)
            prompt_tokens.append(pt)
            completion_tokens.append(ct)
    else:
        semaphore = asyncio.Semaphore(args.parallel)
        tasks = [
            process_sample_with_semaphore(
                semaphore, client, question, few_shot_examples, model
            )
            for question in questions
        ]
        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(*tasks, desc="Processing")
        responses = [r[0] for r in results]
        prompt_tokens = [r[1] for r in results]
        completion_tokens = [r[2] for r in results]

    latency = time.perf_counter() - start

    preds = [get_answer_value(r) for r in responses]
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)
    num_output_tokens = sum(completion_tokens)
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "invalid": round(invalid, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


def main(args):
    asyncio.run(eval_gsm8k(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

import argparse
import ast
import asyncio
import json
import os
import re
import time

import numpy as np

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    dump_bench_raw_result,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

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


# OpenAI-compatible backend (using chat completions API for compatibility)
async def _openai_request(session, url, prompt, model, semaphore):
    import aiohttp

    async with semaphore:
        # Use chat completions format for dynamo frontend compatibility
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0,
            "stop": ["Question", "Assistant:", "<|separator|>"],
        }
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status != 200:
                    return "ERROR", 0
                result = await resp.json()
                return result["choices"][0]["message"]["content"], result.get(
                    "usage", {}
                ).get("completion_tokens", 0)
        except:
            return "ERROR", 0


async def _run_openai(url, model, prompts, parallel):
    import aiohttp
    from tqdm.asyncio import tqdm_asyncio

    semaphore = asyncio.Semaphore(parallel)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=parallel * 2)
    ) as session:
        tasks = [_openai_request(session, url, p, model, semaphore) for p in prompts]
        return await tqdm_asyncio.gather(*tasks)


def main(args):
    if args.backend != "openai":
        set_default_backend(select_sglang_backend(args))

    # Read data
    data_path = args.data_path
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

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

    if args.backend == "openai":
        prompts = [few_shot_examples + q for q in questions]
        tic = time.perf_counter()
        results = asyncio.run(
            _run_openai(
                f"http://{args.host}:{args.port}", args.model, prompts, args.parallel
            )
        )
        latency = time.perf_counter() - tic
        preds = [
            get_answer_value(r[0]) if r[0] != "ERROR" else INVALID for r in results
        ]
        num_output_tokens = sum(r[1] for r in results)
    else:
        #####################################
        ######### SGL Program Begin #########
        #####################################

        import sglang as sgl

        @sgl.function
        def few_shot_gsm8k(s, question):
            s += few_shot_examples + question
            s += sgl.gen(
                "answer",
                max_tokens=512,
                stop=["Question", "Assistant:", "<|separator|>"],
            )

        #####################################
        ########## SGL Program End ##########
        #####################################

        # Run requests
        tic = time.perf_counter()
        states = few_shot_gsm8k.run_batch(
            arguments,
            temperature=0,
            num_threads=args.parallel,
            progress_bar=True,
        )
        latency = time.perf_counter() - tic

        preds = []
        for i in range(len(states)):
            preds.append(get_answer_value(states[i]["answer"]))

        # Compute speed
        num_output_tokens = sum(
            s.get_meta_info("answer")["completion_tokens"] for s in states
        )

        # Dump results
        dump_state_text(f"tmp_output_{args.backend}.txt", states)
        dump_bench_raw_result(
            path=args.raw_result_file,
            states=states,
            preds=preds,
            labels=labels,
        )

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

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
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument(
        "--model", type=str, default="default", help="Model name (for --backend openai)"
    )
    args = add_common_sglang_args_and_parse(parser)
    main(args)

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_select
from sglang.utils import download_and_cache_file, read_jsonl


def get_one_example(lines, i, include_answer):
    ret = lines[i]["activity_label"] + ": " + lines[i]["ctx"] + " "
    if include_answer:
        ret += lines[i]["endings"][lines[i]["label"]]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def main(args):
    # Select backend
    call_select = get_call_select(args)

    # Read data
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    filename = download_and_cache_file(url)
    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    choices = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        choices.append(lines[i]["endings"])
        labels.append(lines[i]["label"])

    preds = [None] * len(labels)

    # Run requests
    if args.backend != "lmql":
        # Use thread pool
        def get_one_answer(i):
            preds[i] = call_select(
                context=few_shot_examples + questions[i], choices=choices[i]
            )

        tic = time.time()
        if args.parallel == 1:
            for i in tqdm(range(len(questions))):
                get_one_answer(i)
        else:
            with ThreadPoolExecutor(args.parallel) as executor:
                list(
                    tqdm(
                        executor.map(get_one_answer, list(range(len(questions)))),
                        total=len(questions),
                    )
                )
    else:
        # Use asyncio
        async def batched_call(batch_size):
            for i in range(0, len(questions), batch_size):
                tasks = []
                for q, c in zip(
                    questions[i : i + batch_size], choices[i : i + batch_size]
                ):
                    tasks.append(call_select(context=few_shot_examples + q, choices=c))
                rets = await asyncio.gather(*tasks)
                for j in range(len(rets)):
                    preds[i + j] = rets[j]

        tic = time.time()
        asyncio.run(batched_call(batch_size=args.parallel))

    latency = time.time() - tic

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    print(f"Latency: {latency:.3f}")
    print(f"Accuracy: {acc:.3f}")

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "hellaswag",
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
    parser.add_argument("--num-shots", type=int, default=20)
    parser.add_argument("--data-path", type=str, default="hellaswag_val.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_other_args_and_parse(parser)
    main(args)

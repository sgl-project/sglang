import argparse
import asyncio
import json
import os
import time
from typing import Any, Tuple

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from sglang.test.test_utils import add_common_sglang_args_and_parse

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_few_shot_messages(train_df, subject, k=-1):
    if k == -1:
        k = train_df.shape[0]
    messages = []

    for i in range(k):
        question = train_df.iloc[i, 0]
        num_choices = train_df.shape[1] - 2
        for j in range(num_choices):
            question += "\n{}. {}".format(choices[j], train_df.iloc[i, j + 1])
        question += "\nAnswer:"

        answer = train_df.iloc[i, num_choices + 1]

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
        max_tokens=1,
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
        return await process_sample(client, few_shot_examples, question, model)


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


async def eval_mmlu(args) -> None:
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    few_shot_examples_list = []
    questions = []
    labels = []
    num_questions = []

    for subject in subjects[: args.nsub]:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        num_questions.append(test_df.shape[0])

        k = args.ntrain
        few_shot_messages = gen_few_shot_messages(dev_df, subject, k)

        for i in range(test_df.shape[0]):
            question = format_example(test_df, i, include_answer=False)
            few_shot_examples_list.append(few_shot_messages)
            questions.append(question)

            label = test_df.iloc[i, test_df.shape[1] - 1]
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
        for fs, q in tqdm(zip(few_shot_examples_list, questions), total=len(questions)):
            response, pt, ct = await process_sample(client, fs, q, model)
            responses.append(response)
            prompt_tokens.append(pt)
            completion_tokens.append(ct)
    else:
        semaphore = asyncio.Semaphore(args.parallel)
        tasks = [
            process_sample_with_semaphore(semaphore, client, fs, q, model)
            for fs, q in zip(few_shot_examples_list, questions)
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Processing")

        responses = [r[0] for r in results]
        prompt_tokens = [r[1] for r in results]
        completion_tokens = [r[2] for r in results]

    latency = time.perf_counter() - start

    preds = [
        response.strip()[0] if len(response.strip()) > 0 else ""
        for response in responses
    ]

    cors = [pred == label for pred, label in zip(preds, labels)]

    pt = 0
    for subject, num_qs in zip(subjects[: args.nsub], num_questions):
        print(
            f"subject: {subject}, #q:{num_qs}, acc: {np.mean(cors[pt: pt + num_qs]):.3f}"
        )
        pt += num_qs
    assert pt == len(cors)
    weighted_acc = np.mean(cors)

    # Print results
    print("Total latency: {:.3f}".format(latency))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "mmlu",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(weighted_acc, 3),
            "num_requests": len(questions),
            "other": {
                "nsub": args.nsub,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


def main(args):
    asyncio.run(eval_mmlu(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--nsub", type=int, default=60)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

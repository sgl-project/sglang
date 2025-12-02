import argparse
import json
import os
import time

import numpy as np

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
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
    set_default_backend(select_sglang_backend(args))

    # Read data
    data_path = args.data_path
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

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
    arguments = [{"question": q, "choices": c} for q, c in zip(questions, choices)]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_hellaswag(s, question, choices):
        s += few_shot_examples + question
        s += sgl.select("answer", choices=choices)

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    rets = few_shot_hellaswag.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    preds = [choices[i].index(rets[i]["answer"]) for i in range(len(rets))]
    latency = time.perf_counter() - tic

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
    args = add_common_sglang_args_and_parse(parser)
    main(args)

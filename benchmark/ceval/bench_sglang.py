import argparse
import json
import os
import random
import re
import time

import numpy as np
from datasets import load_dataset

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)

choices = ["A", "B", "C", "D"]


def get_one_example(line, include_answer):
    res = line["question"]
    res += f"\nA. {line['A']}"
    res += f"\nB. {line['B']}"
    res += f"\nC. {line['C']}"
    res += f"\nD. {line['D']}"

    if include_answer:
        res += f"\nAnswer: {line['answer']} \n\n"
    return res


def get_few_shot_examples(lines):
    res = ""
    for line in lines:
        res += get_one_example(line, True) + "\n\n"
    return res


def get_answer_value(response):
    pattern = r"(Answer:|answer:|答案是|答案是:|正确答案是:|答案:|Assistant:)\s*([A-D])(?![\w])"
    match = re.search(pattern, response)

    if match:
        return match.group(2)

    return random.choice(choices)


def main(args):
    # Read data && Construct prompts
    arguments = []
    labels = []
    examples = "examples:\n"
    data_path = args.data_path
    for subject in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject)
        if os.path.isdir(subject_path) and subject != ".git":
            dataset = load_dataset(data_path, name=subject)
            dev_lines_temp = dataset["dev"]
            val_lines_temp = dataset["val"]
            few_shot_examples = get_few_shot_examples(dev_lines_temp)
            examples += f"{few_shot_examples}"
            for val_line in val_lines_temp:
                arguments.append(
                    {
                        "examples": few_shot_examples,
                        "question": get_one_example(val_line, False),
                    }
                )
                labels.append(val_line["answer"])

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_ceval(s, examples, question):
        s += examples + question + sgl.gen("Answer")

    #####################################
    ########## SGL Program End ##########
    #####################################

    num_questions = args.num_questions if args.num_questions else len(arguments)

    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Run requests
    tic = time.perf_counter()
    states = few_shot_ceval.run_batch(
        arguments[:num_questions],
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = [get_answer_value(states[i]["Answer"]) for i in range(num_questions)]

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels[:num_questions]))

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("Answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "ceval",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="ceval/ceval-exam")
    parser.add_argument("--num-questions", type=int, default=None)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

import argparse
import json
import time

import numpy as np

from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import read_jsonl


def get_example(lines, i, answer):
    prompt = "Question: " + lines[i]["question"] + lines[i]["passage"] + "\nAnswer:"
    if answer:
        prompt += str(lines[i]["answer"])
    return prompt


def few_shot_examples(lines, k):
    prompts = ""
    for i in range(k):
        prompts += get_example(lines, i, True) + "\n\n"
    return prompts


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Read data
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    lines_train = list(read_jsonl(train_data_path))
    lines_test = list(read_jsonl(test_data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shots = few_shot_examples(lines_train, num_shots)

    questions = []
    answer = []
    for i in range(len(lines_test[:num_questions])):
        questions.append(get_example(lines_test, i, False))
        answer.append(str(lines_test[i]["answer"]))
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_boolq(s, question):
        s += few_shots + question
        s += sgl.gen("answer", max_tokens=5, stop=["\n"])

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_boolq.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(states[i]["answer"])

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(answer))

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "boolq",
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
    parser.add_argument(
        "--train-data-path", type=str, default="./boolq/data/train-00000-of-00001.json"
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="./boolq/data/validation-00000-of-00001.json",
    )
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

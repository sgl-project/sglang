import argparse
import ast
import json
import re
import time

import numpy as np
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text


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
    numbers = re.findall(r'\d+', answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    lines = read_jsonl(args.data_path)

    # Construct prompts
    k = args.num_shot
    few_shot_examples = get_few_shot_examples(lines, k)

    questions = []
    labels = []
    for i in range(len(lines[:args.num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen("answer", max_tokens=256, stop="Question")

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = few_shot_gsm8k.run_batch(
        arguments, temperature=0, backend=backend, num_threads=args.parallel)
    latency = time.time() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)
    print(f"Latency: {latency:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Accuracy: {acc:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

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
            }
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shot", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

import argparse
import ast
from collections import Counter
import json
import re
import time

import numpy as np
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text
import sglang as sgl


INVALID = -9999999


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r'\d+', answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def most_frequent_number(numbers):
    if not numbers:
        return None

    frequency = Counter(numbers)
    most_frequent = max(frequency, key=frequency.get)
    return most_frequent


# Use a low temp to make the results more deterministic and the comparison more fair.
temp = 0.3


def propose_plan(s, question, num_branches):
    s += sgl.user(
"""Please generate a high-level plan for solving the following question. As the first step, just say what method and idea you will use to solve the question. You can reorganize the information in the question. Do not do the actual calculation. Keep your response concise and within 80 words. Question: """ + question)
    forks = s.fork(num_branches)
    forks += sgl.assistant(sgl.gen("plan", max_tokens=256, temperature=temp))
    return forks


def execute_plan(s, num_branches):
    s += sgl.user(
"""The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the high-level plan. Give me the final answer. Make your response short.""")
    forks = s.fork(num_branches)
    forks += sgl.assistant(sgl.gen("answer", max_tokens=256, temperature=temp))
    return forks


def reflect_solution(s, num_branches):
    s += sgl.user(
"""Okay. Now you evaluate your own solution and give it a score on a scale of 1 to 5. Please do rigorous check of the correctness.""")
    forks = s.fork(num_branches)
    forks += sgl.assistant(sgl.gen("score", max_tokens=256, temperature=temp))
    return forks


@sgl.function
def tree_search(s, question, num_branches):
    forks_to_join = []

    plan_forks = propose_plan(s, question, num_branches)
    forks_to_join.append(plan_forks)

    sol_states = []
    for plan in plan_forks:
        forks = execute_plan(plan, num_branches)
        forks_to_join.append(forks)
        sol_states.extend(forks)

    for sol in sol_states:
        forks = reflect_solution(sol, num_branches)
        forks_to_join.append(forks)

    for f in reversed(forks_to_join):
        f.join()


def main(args):
    lines = read_jsonl(args.data_path)

    # Construct prompts
    num_branches = 3
    questions = []
    labels = []
    for i in range(len(lines[:args.num_questions])):
        questions.append(lines[i]["question"])
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q, "num_branches": num_branches} for q in questions]

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = tree_search.run_batch(
        arguments, temperature=0, backend=backend, num_threads=args.parallel)
    latency = time.time() - tic
    answers_text = []
    for s in states:
        answers_text.append([x for xs in s["answer"] for x in xs])

    preds = []
    for i in range(len(states)):
        answers = [get_answer_value(v) for v in answers_text[i]]
        preds.append(most_frequent_number(answers))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)
    print(f"Latency: {latency:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Accuracy: {acc:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", answers_text)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "tree_of_thought_gsm8k",
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
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

import argparse
import ast
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import dump_state_text, read_jsonl

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


def most_frequent_number(numbers):
    if not numbers:
        return None

    frequency = Counter(numbers)
    most_frequent = max(frequency, key=frequency.get)
    return most_frequent


USER_PREFIX = "[INST] "
USER_SUFFIX = " [/INST]"
ASSISTANT_PREFIX = ""
ASSISTANT_SUFFIX = " </s><s>"

# Use a low temp to make the results more deterministic and the comparison more fair.
temp = 0.3


def propose_plan(s, question, num_branches, call_generate):
    s += (
        USER_PREFIX
        + """Please generate a high-level plan for solving the following question. As the first step, just say what method and idea you will use to solve the question. You can reorganize the information in the question. Do not do the actual calculation. Keep your response concise and within 80 words. Question: """
        + question
        + USER_SUFFIX
    )

    s += ASSISTANT_PREFIX
    comps = call_generate(
        s, max_tokens=256, temperature=temp, stop=None, n=num_branches
    )
    return [s + comp + ASSISTANT_SUFFIX for comp in comps]


def execute_plan(s, num_branches, call_generate):
    s += (
        USER_PREFIX
        + """The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the high-level plan. Give me the final answer. Make your response short."""
        + USER_SUFFIX
    )
    s += ASSISTANT_PREFIX
    comps = call_generate(
        s, max_tokens=256, temperature=temp, stop=None, n=num_branches
    )
    return [s + comp + ASSISTANT_SUFFIX for comp in comps]


def reflect_solution(s, num_branches, call_generate):
    s += (
        USER_PREFIX
        + """Okay. Now you evaluate your own solution and give it a score on a scale of 1 to 5. Please do rigorous check of the correctness."""
        + USER_SUFFIX
    )
    s += ASSISTANT_PREFIX
    comps = call_generate(
        s, max_tokens=256, temperature=temp, stop=None, n=num_branches
    )
    return [s + comp + ASSISTANT_SUFFIX for comp in comps]


def tree_search(question, num_branches, call_generate):
    s = ""
    solutions = []

    plan_forks = propose_plan(s, question, num_branches, call_generate)
    for plan in plan_forks:
        sol_forks = execute_plan(plan, num_branches, call_generate)
        for sol in sol_forks:
            score_forks = reflect_solution(sol, num_branches, call_generate)
        solutions.append(sol_forks)

    return solutions


def main(args):
    lines = read_jsonl(args.data_path)

    # Construct prompts
    num_branches = 3
    questions = []
    labels = []
    for i in range(len(lines[: args.num_questions])):
        questions.append(lines[i]["question"])
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q, "num_branches": num_branches} for q in questions]

    # Select backend
    call_generate = get_call_generate(args)

    # Run requests
    states = [None] * len(questions)

    def get_one_answer(i):
        states[i] = tree_search(**arguments[i], call_generate=call_generate)

    tic = time.perf_counter()
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

    latency = time.perf_counter() - tic

    answers_text = []
    for s in states:
        answers_text.append([x for xs in s for x in xs])

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
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_other_args_and_parse(parser)
    main(args)

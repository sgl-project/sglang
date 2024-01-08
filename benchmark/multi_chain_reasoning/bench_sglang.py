import argparse
import ast
import json
import re
import time

import numpy as np
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text


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


prompt_lib = [
    "Let us think step by step.",
    "Approach this methodically. Let's dissect the problem into smaller, more manageable parts.",
    "It's important to proceed step by step, ensuring accuracy at each stage.",
    "Take a deep breath and break this down.",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem.",
    "I am extremely good at math.",
]


def main(args):
    lines = read_jsonl(args.data_path)

    # Construct prompts
    #k = args.num_shot
    #few_shot_examples = get_few_shot_examples(lines, k)

    questions = []
    labels = []
    for i in range(len(lines[:args.num_questions])):
        questions.append(lines[i]["question"])
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    num_chains = args.num_chains

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def multi_chain_gsm8k(s, question):
        s += "Question: " + question + "\n"
        #s += "Answer: " + prompt_lib[0] + sgl.gen("answer", max_tokens=256, stop="Question",
        #    temperature=0)
        #return

        forks = s.fork(num_chains)
        for i in range(num_chains):
            forks[i] += ("Answer: " + prompt_lib[i % num_chains] +
                sgl.gen(f"chain", max_tokens=256, temperature=0.3, stop="Question"))
        forks.join()

        s += "Answer: To answer this question, here are some possible solutions. "
        s += "After considering all of them, I will do a majority vote.\n\n"
        for i in range(num_chains):
            s += f"Solution {i+1}: " + forks[i]["chain"].strip() + "\n\n"
        s += f"\nBy considering the above solutions and doing a majority vote, I think the final answer (a single integer number) is "
        s += sgl.gen("answer", max_tokens=16)

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = multi_chain_gsm8k.run_batch(
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
            "task": "multi_chain_gsm8k",
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
    parser.add_argument("--num-shot", type=int, default=0)
    parser.add_argument("--num-chains", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=50)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

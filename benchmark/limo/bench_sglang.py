import argparse
import json
import re
import time
from typing import List, Tuple

import jsonschema
from datasets import load_dataset

import sglang as sgl
from sglang.global_config import global_config
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


@sgl.function
def reasoning_gen(s, question: str):
    s += sgl.user(question + "\nPlease reason step by step, and put your final answer within \boxed{}.")
    s += sgl.assistant(
        sgl.gen(
            "answer",
        )
    )


def convert_dataset(path: str):
    raw_dataset = load_dataset(path)
    questions = []
    answers = []
    for data in raw_dataset["train"]:
        question = data["question"]
        answer = data["answer"]
        questions.append({"question": question})
        answers.append({"answer": answer})
    return questions, answers


def extract_boxed_text(text: str):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""

def main(args):
    # Select backend
    sgl.set_default_backend(select_sglang_backend(args))

    # Get dataset
    questions, answers = convert_dataset(args.data_path)

    # Run requests
    tic = time.time()
    states = reasoning_gen.run_batch(
        questions,
        num_threads=args.parallel,
        progress_bar=True,
        temperature=0.6,
        max_new_tokens=16384,
    )
    latency = time.time() - tic

    # Extract answers
    preds = []
    correct = 0
    for i, state in enumerate(states):
        try:
            preds.append(extract_boxed_text(state["answer"]))
            correct += 1 if preds[-1] == str(answers[i]["answer"]) else 0
        except Exception as e:
            print(f"Error extracting answer: {e}")
            preds.append("")

    # Calculate accuracy
    accuracy = correct / len(questions)
    print(f"Accuracy: {accuracy}")

    # Calculate output throughput
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency
    print(f"Output throughput: {output_throughput} token/s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "limo",
            "backend": args.backend,
            "latency": round(latency, 3),
            "accuracy": round(accuracy, 3),
            "num_requests": len(questions),
            "other": {
                "num_questions": len(questions),
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_sglang_args_and_parse(parser)
    parser.add_argument("--data-path", type=str, default="GAIR/LIMO")
    args = parser.parse_args()
    main(args)

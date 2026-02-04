import argparse
import ast
import json
import os
import re
import time

import numpy as np
from datasets import load_dataset

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    dump_bench_raw_result,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

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
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Read data
    if args.platinum:
        print("Loading GSM8K Platinum dataset from HuggingFace...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
        lines = [
            {"question": item["question"], "answer": item["answer"]} for item in dataset
        ]
    else:
        data_path = args.data_path
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
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
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Compute acceptance length (for speculative decoding)
    has_verify = "spec_verify_ct" in states[0].get_meta_info("answer")
    if has_verify:
        # Track SD-only answers
        num_sd_tokens = 0
        num_sd_verify = 0
        num_sd_answers = 0
        num_non_sd_answers = 0
        acceptance_lengths = []  # Per-question acceptance lengths

        for s in states:
            meta = s.get_meta_info("answer")
            verify_ct = meta.get("spec_verify_ct", 0)

            # Use sd_completion_tokens if available (excludes non-SD tokens)
            # Otherwise fall back to completion_tokens for backwards compatibility
            if "sd_completion_tokens" in meta:
                sd_tokens = meta["sd_completion_tokens"]
            else:
                # Fallback: only count completion_tokens if SD was used
                sd_tokens = meta["completion_tokens"] if verify_ct > 0 else 0

            # Only count answers where SD was actually used
            if verify_ct > 0:
                num_sd_tokens += sd_tokens
                num_sd_verify += verify_ct
                num_sd_answers += 1
                acceptance_lengths.append(sd_tokens / verify_ct)
            else:
                num_non_sd_answers += 1

        print(
            f"\n[DEBUG] SD answers: {num_sd_answers}, Non-SD answers: {num_non_sd_answers}"
        )
        print(f"[DEBUG] SD tokens: {num_sd_tokens}, SD verify steps: {num_sd_verify}")

        if acceptance_lengths:
            print(f"[DEBUG] Per-question acceptance length:")
            print(f"  Min: {min(acceptance_lengths):.2f}")
            print(f"  Max: {max(acceptance_lengths):.2f}")
            print(f"  Median: {np.median(acceptance_lengths):.2f}")
            print(f"  Mean: {np.mean(acceptance_lengths):.2f}")

        if num_sd_verify > 0:
            accept_length = num_sd_tokens / num_sd_verify
        else:
            accept_length = 1.0  # No SD was used
    else:
        accept_length = 1.0

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Acceptance length: {accept_length:.3f}")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    dump_bench_raw_result(
        path=args.raw_result_file,
        states=states,
        preds=preds,
        labels=labels,
    )

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k-platinum" if args.platinum else "gsm8k",
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
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument(
        "--platinum",
        action="store_true",
        help="Use GSM8K Platinum dataset (drop-in replacement with corrected labels)",
    )
    args = add_common_sglang_args_and_parse(parser)
    main(args)

"""
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding

Usage:
python3 benchmark/mtbench/bench_sglang_eagle.py --num-questions 80 --parallel 1
"""

import argparse
import json
import os
import time
import uuid

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i]["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "prompt": [answers[i][0], answers[i][1]],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_mt_bench(s, question_1, question_2):
    s += sgl.system(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def main(args):
    # Construct prompts
    questions = load_questions(args.question_file)[: args.num_questions]
    arguments = [
        {"question_1": q["prompt"][0], "question_2": q["prompt"][1]} for q in questions
    ]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.perf_counter()
    rets = answer_mt_bench.run_batch(
        arguments,
        temperature=0,
        max_new_tokens=2048,
        num_threads=args.parallel,
        progress_bar=True,
    )
    answers = [[s["answer_1"], s["answer_2"]] for s in rets]

    latency = time.perf_counter() - tic
    num_output_tokens = sum(
        s.get_meta_info("answer_1")["completion_tokens"]
        + s.get_meta_info("answer_2")["completion_tokens"]
        for s in rets
    )

    # NOTE: acceptance length is just completion_tokens / spec_verify_ct
    # {'id': '3bb9c5ead109488d8ed5ee9cbecaec29', 'finish_reason': {'type': 'length', 'length': 256}, 'prompt_tokens': 37, 'spec_verify_ct': 101, 'completion_tokens': 256, 'cached_tokens': 0}

    output_throughput = num_output_tokens / latency

    has_verify = "spec_verify_ct" in rets[0].get_meta_info("answer_1")
    if has_verify:
        num_verify_tokens = sum(
            s.get_meta_info("answer_1")["spec_verify_ct"]
            + s.get_meta_info("answer_2")["spec_verify_ct"]
            for s in rets
        )

        accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    print(
        f"#questions: {len(questions)}, Throughput: {output_throughput:.2f} token/s, Acceptance length: {accept_length:.2f}"
    )

    # Write results
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"tmp_output_{args.backend}.txt"
    write_answers(answer_file, model_id, questions, answers)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "mtbench",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "throughput": round(output_throughput, 3),
            "accept_length": round(accept_length, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

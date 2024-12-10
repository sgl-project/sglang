# reference: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/eval_long_context.py
import argparse
import json
import os
import time

from compute_scores import compute_scores
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS, create_prompt, get_answer, load_data

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def validate_args(args):
    assert args.task in ["passkey", "kv_retrieval"], f"Invalid task: {args.task}"


def write_answers(filename, model_id, results):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(results)):
            ans_json = {
                "question_id": results[i]["question_id"],
                "model_id": model_id,
                "prediction": results[i]["prediction"],
                "ground_truth": results[i]["ground_truth"],
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def infinitebench(s, question, max_tokens):
    s += question
    s += sgl.gen(
        "answer",
        max_tokens=max_tokens,
    )


def main(args):
    validate_args(args)

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Data
    data_name = args.task
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]  # max output length
    lines = load_data(data_name, data_dir=args.data_dir)
    if args.end_idx is None:
        args.end_idx = len(lines)

    # Construct prompts
    questions = []
    labels = []
    for i in range(len(lines[args.start_idx : args.end_idx])):
        questions.append(create_prompt(lines[i], data_name, args.data_dir))
        labels.append(get_answer(lines[i], data_name))
    arguments = [{"question": q, "max_tokens": max_tokens} for q in questions]

    # Run requests
    tic = time.time()
    results = infinitebench.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.time() - tic

    # Compute scores
    results = [
        {"ground_truth": label, "prediction": s["answer"], "question_id": line["id"]}
        for line, label, s in zip(lines, labels, results)
    ]
    acc = compute_scores(results, args.task)
    print(f"#questions: {len(questions)}, Latency: {latency:.2f}, Accuracy: {acc:.3f}")

    # Write results to file
    model_id = backend.model_info["model_path"]
    answer_file = f"tmp_output_{data_name}_{args.backend}.txt"
    write_answers(answer_file, model_id, results)

    with open(args.result_file, "a") as fout:
        value = {
            "task": args.task,
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": len(questions),
            "other": {
                "num_questions": len(questions),
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, choices=["passkey", "kv_retrieval"], required=True
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

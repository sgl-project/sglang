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
                    "turns": [answers[i][0], answers[i][1]],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_mt_bench(s, question_1, question_2):
    s += sgl.system()
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def main(args):
    # Construct prompts
    questions = load_questions(args.question_file)[: args.num_questions]
    arguments = [
        {"question_1": q["turns"][0], "question_2": q["turns"][1]} for q in questions
    ]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    all_rets = []
    all_questions = []

    # Calculate batch sizes based on total number of questions
    first_batch_size = int(args.num_questions * 0.8)
    remaining = args.num_questions - first_batch_size
    small_batch_size = max(1, remaining // 4)  # Divide remaining into 4 batches

    batch_configs = [
        (first_batch_size, 0),
        (small_batch_size, 0),
        (small_batch_size, 0),
        (small_batch_size, 0),
        (small_batch_size, 0),
    ]

    start_idx = 0
    overall_start = time.perf_counter()

    for batch_num, (batch_size, wait_time) in enumerate(batch_configs, 1):
        # Wait before sending (except for first batch)
        if wait_time > 0:
            print(f"\nWaiting {wait_time}s before batch {batch_num}...")
            time.sleep(wait_time)

        # Get batch arguments
        end_idx = min(start_idx + batch_size, len(arguments))
        if start_idx >= len(arguments):
            break

        batch_arguments = arguments[start_idx:end_idx]
        batch_questions = questions[start_idx:end_idx]
        actual_batch_size = len(batch_arguments)

        print(
            f"\nSending batch {batch_num}: {actual_batch_size} questions (indices {start_idx}-{end_idx-1})"
        )

        # Run batch
        batch_start = time.perf_counter()
        batch_rets = answer_mt_bench.run_batch(
            batch_arguments,
            temperature=0,
            max_new_tokens=256,
            num_threads=args.parallel,
            progress_bar=True,
        )
        batch_latency = time.perf_counter() - batch_start

        print(f"Batch {batch_num} completed in {batch_latency:.3f}s")

        all_rets.extend(batch_rets)
        all_questions.extend(batch_questions)
        start_idx = end_idx

    total_latency = time.perf_counter() - overall_start

    # Process results
    answers = [[s["answer_1"], s["answer_2"]] for s in all_rets]

    print("\n" + "=" * 50)
    print("Final Results:")
    print("=" * 50)
    print(f"#questions: {len(all_questions)}, Total Latency: {total_latency:.2f}s")

    # Write results
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"tmp_output_{args.backend}_adaserve.txt"
    write_answers(answer_file, model_id, all_questions, answers)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "mtbench_adaserve",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(total_latency, 3),
            "num_requests": len(all_questions),
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
                "batch_pattern": f"{first_batch_size} + 5x{small_batch_size}",
                "initial_wait": 0,
                "subsequent_wait": 0,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
    args = add_common_sglang_args_and_parse(parser)
    args.parallel = 48
    main(args)

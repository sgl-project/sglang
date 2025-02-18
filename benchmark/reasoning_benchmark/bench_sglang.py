import argparse
import json
import time

import answer_extraction
import eval_utils
from datasets import load_dataset

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


@sgl.function
def reasoning_gen(s, question: str):
    s += sgl.user(
        question
        + "\nPlease reason step by step, and put your final answer within \boxed{}."
    )
    s += sgl.assistant(
        sgl.gen(
            "answer",
        )
    )


def convert_dataset(path: str, question_key: str, answer_key: str, num_tries: int):
    raw_dataset = load_dataset(path)
    questions = []
    answers = []
    for data in raw_dataset["train"]:
        question = data[question_key]
        answer = data[answer_key]
        for _ in range(num_tries):
            questions.append({"question": question})
            answers.append({"answer": answer})
    return questions, answers


def main(args):
    # Select backend
    sgl.set_default_backend(select_sglang_backend(args))

    # Get dataset
    questions, answers = convert_dataset(
        args.data_path, args.question_key, args.answer_key, args.num_tries
    )

    # Run requests
    tic = time.time()
    states = reasoning_gen.run_batch(
        questions,
        num_threads=args.parallel,
        progress_bar=True,
        temperature=0.6,
        max_new_tokens=32768,
        top_p=0.95,
    )
    latency = time.time() - tic

    # Extract answers
    correct = 0
    for i, state in enumerate(states):
        try:
            pred_answer = answer_extraction.extract_math_answer(
                questions[i]["question"], state["answer"], "limo"
            )
            gt_answer = str(answers[i]["answer"])
            # Use last answer if multiple were extracted
            pred_answer = (
                pred_answer[-1] if isinstance(pred_answer, list) else pred_answer
            )
            correct += 1 if eval_utils.math_equal(pred_answer, gt_answer) else 0
        except Exception as e:
            print(f"Error extracting answer: {e}")
            pass

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
    parser.add_argument("--data-path", type=str, default="GAIR/LIMO")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--num-tries", type=int, default=1)
    add_common_sglang_args_and_parse(parser)
    args = parser.parse_args()
    main(args)

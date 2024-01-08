import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
import time
import uuid

from fastchat.model import get_conversation_template
import requests
from sglang.test.test_utils import add_common_other_args_and_parse, call_generate_lightllm, call_generate_vllm, call_generate_srt


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


def main(args):
    questions = load_questions(args.question_file)
    questions = (questions * 10)[:args.num_questions]
    max_tokens = 256
    model_id = "llama-2-chat"

    conv_main = get_conversation_template(model_id)

    # Select backend
    if args.backend == "lightllm":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_lightllm, url=url, stop=None)
    elif args.backend == "vllm":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_vllm, url=url, stop=None)
    elif args.backend == "srt":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_srt, url=url, stop=None)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    answers = [None] * len(questions)

    def get_answer(i):
        conv = conv_main.copy()
        cur_answers = []
        for j in range(2):
            q = questions[i]["turns"][j]
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt() 
            output = call_generate(prompt,
                temperature=0, max_tokens=max_tokens).strip()

            cur_answers.append(output)
            conv.update_last_message(output)

        answers[i] = cur_answers

    # Run requests
    tic = time.time()
    if args.parallel == 1:
        for i in range(len(questions)):
            get_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            executor.map(get_answer, list(range(len(questions))))
    latency = time.time() - tic

    print(f"#questions: {len(questions)}, Latency: {latency:.2f}")

    # Write results
    answer_file = args.answer_file or f"tmp_output_{args.backend}.txt"
    write_answers(answer_file, model_id, questions, answers)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "mtbench",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            }
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
    args = add_common_other_args_and_parse(parser)
    main(args)

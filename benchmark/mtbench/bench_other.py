import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastchat.model import get_conversation_template
from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import download_and_cache_file


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
    # Download question file if not exist
    question_file = args.question_file
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    if not os.path.isfile(question_file):
        question_file = download_and_cache_file(url)

    questions = load_questions(question_file)
    questions = (questions * 10)[: args.num_questions]
    max_tokens = 256
    model_id = "llama-2-chat"

    conv_main = get_conversation_template(model_id)

    # Select backend
    call_generate = get_call_generate(args)

    answers = [None] * len(questions)

    def get_answer(i):
        conv = conv_main.copy()
        cur_answers = []
        for j in range(2):
            q = questions[i]["turns"][j]
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            output = call_generate(prompt, temperature=0, max_tokens=max_tokens).strip()

            cur_answers.append(output)
            conv.update_last_message(output)

        answers[i] = cur_answers

    # Run requests
    tic = time.perf_counter()
    if args.parallel == 1:
        for i in tqdm(range(len(questions))):
            get_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            list(
                tqdm(
                    executor.map(get_answer, list(range(len(questions)))),
                    total=len(questions),
                )
            )

    latency = time.perf_counter() - tic

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
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
    args = add_common_other_args_and_parse(parser)
    main(args)

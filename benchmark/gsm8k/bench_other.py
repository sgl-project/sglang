import argparse
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import re
import time

import numpy as np
from tqdm import tqdm
from sglang.test.test_utils import add_common_other_args_and_parse, call_generate_lightllm, call_generate_vllm, call_generate_srt_raw
from sglang.utils import read_jsonl, dump_state_text


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
    numbers = re.findall(r'\d+', answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    lines = read_jsonl(args.data_path)

    # Construct prompts
    k = args.num_shot
    few_shot_examples = get_few_shot_examples(lines, k)

    questions = []
    labels = []
    for i in range(len(lines[:args.num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    states = [None] * len(labels)

    # Select backend
    if args.backend == "lightllm":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_lightllm, url=url)
    elif args.backend == "vllm":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_vllm, url=url)
    elif args.backend == "srt-raw":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_srt_raw, url=url)
    elif args.backend == "guidance":
        from guidance import models, gen

        model = models.LlamaCpp("/home/ubuntu/model_weights/Llama-2-7b-chat.gguf", n_gpu_layers=-1, n_ctx=4096)

        def call_generate(prompt, temperature, max_tokens, stop):
            out = model + prompt + gen(name="answer",
                max_tokens=max_tokens, temperature=temperature, stop=stop)
            return out["answer"]

    elif args.backend == "lmql":
        import lmql
        model = lmql.model(args.model_path,
           endpoint=f"{args.host}:{args.port}")

        @lmql.query(model=model)
        async def program(question):
            '''lmql
            """{question}[ANSWER]""" where len(TOKENS(ANSWER)) < 257 and STOPS_AT(ANSWER, "Question")
            return ANSWER
            '''

        async def call_generate(prompt, temperature, max_tokens, stop):
            return await program(question=prompt, temperature=0)

    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    # Run requests
    if args.backend != "lmql":
        # Use thread pool
        def get_one_answer(i):
            answer = call_generate(
                prompt=few_shot_examples + questions[i],
                temperature=0,
                max_tokens=256,
                stop="Question")
            states[i] = answer

        tic = time.time()
        if args.parallel == 1:
            for i in tqdm(range(len(questions))):
                get_one_answer(i)
        else:
            with ThreadPoolExecutor(args.parallel) as executor:
                executor.map(get_one_answer, list(range(len(questions))))
    else:
        # Use asyncio
        async def batched_call(batch_size):
            for i in range(0, len(questions), batch_size):
                tasks = []
                for q in questions[i:i+batch_size]:
                    tasks.append(call_generate(few_shot_examples + q,
                        temperature=0, max_tokens=256, stop="Question"))
                rets = await asyncio.gather(*tasks)
                for j in range(len(rets)):
                    states[i+j] = rets[j]

        tic = time.time()
        asyncio.run(batched_call(batch_size=args.parallel))
    latency = time.time() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]))

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
            "task": "gsm8k",
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
    parser.add_argument("--num-shot", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_other_args_and_parse(parser)
    main(args)

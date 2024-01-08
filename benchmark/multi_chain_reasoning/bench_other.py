import argparse
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import re
import time

import numpy as np
from sglang.test.test_utils import add_common_other_args_and_parse, call_generate_lightllm, call_generate_vllm, call_generate_srt_raw
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


def multi_chain_gsm8k(question, num_chains, call_generate):
    s = "Question: " + question + "\n"
    # s += call_generate(s + "Answer: " + prompt_lib[0], max_tokens=256,
    #     stop="Question", temperature=0)
    # return s

    comps = []
    for i in range(num_chains):
        comps.append(call_generate(s + "Answer: " + prompt_lib[i % num_chains],
                     max_tokens=256, temperature=0.3, stop="Question"))

    s += "Answer: To answer this question, here are some possible solutions. "
    s += "After considering all of them, I will do a majority vote.\n\n"
    for i in range(num_chains):
        s += f"Solution {i+1}: " + comps[i].strip() + "\n\n"
    s += f"\nBy considering the above solutions and doing a majority vote, I think the final answer (a single integer number) is "
    s += call_generate(s, max_tokens=16, temperature=0, stop=None)
    return s


def main(args):
    lines = read_jsonl(args.data_path)

    # Construct prompts
    k = args.num_shot

    questions = []
    labels = []
    for i in range(len(lines[:args.num_questions])):
        questions.append(lines[i]["question"])
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

        #def multi_chain_gsm8k(question, num_chains, call_generate):
        #    s = model + "Question: " + question + "\n"

        #    comps = []
        #    for i in range(num_chains):
        #        comps.append(call_generate(s + "Answer: " + prompt_lib[i % num_chains],
        #                     max_tokens=256, temperature=0.3, stop="Question"))

        #    s += "Answer: To answer this question, here are some possible solutions. "
        #    s += "After considering all of them, I will do a majority vote.\n\n"
        #    for i in range(num_chains):
        #        s += f"Solution {i+1}: " + comps[i].strip() + "\n\n"
        #    s += f"\nBy considering the above solutions and doing a majority vote, I think the final answer (a single integer number) is "
        #    return call_generate(s, max_tokens=16, temperature=0, stop=None)

    elif args.backend == "lmql":
        import lmql
        model = lmql.model("meta-llama/Llama-2-7b-chat-hf",
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
            answer = multi_chain_gsm8k(questions[i], args.num_chains,
                call_generate)
            states[i] = answer

        tic = time.time()
        if args.parallel == 1:
            for i in range(len(questions)):
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
                    states[i+j] = get_answer_value(rets[j])

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
    args = add_common_other_args_and_parse(parser)
    main(args)

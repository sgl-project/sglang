import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from functools import partial
import time

import numpy as np
from sglang.test.test_utils import add_common_other_args_and_parse, call_select_lightllm, call_select_vllm
from sglang.utils import read_jsonl


def get_one_example(lines, i, include_answer):
    ret = lines[i]["activity_label"] + ": " +  lines[i]["ctx"] + " "
    if include_answer:
        ret += lines[i]["endings"][lines[i]["label"]]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def main(args):
    lines = read_jsonl(args.data_path)

    # Construct prompts
    k = args.num_shot
    few_shot_examples = get_few_shot_examples(lines, k)

    questions = []
    choices = []
    labels = []
    for i in range(len(lines[:args.num_questions])):
        questions.append(get_one_example(lines, i, False))
        choices.append(lines[i]["endings"])
        labels.append(lines[i]["label"])

    preds = [None] * len(labels)

    # Select backend
    if args.backend == "lightllm":
        url = f"{args.host}:{args.port}/generate"
        call_select = partial(call_select_lightllm, url=url)
    elif args.backend == "vllm":
        url = f"{args.host}:{args.port}/generate"
        call_select = partial(call_select_vllm, url=url)
    elif args.backend == "guidance":
        from guidance import models, select

        model = models.LlamaCpp("/home/ubuntu/model_weights/Llama-2-7b-chat.gguf", n_gpu_layers=-1, n_ctx=4096)

        def call_select(context, choices):
            out = model + context + select(choices, name="answer")
            return choices.index(out["answer"])

        call_select("Hello,", ["world", "earth"])

    elif args.backend == "lmql":
        import lmql
        model = lmql.model("meta-llama/Llama-2-7b-chat-hf",
           endpoint=f"{args.host}:{args.port}")

        @lmql.query(model=model)
        async def program(ctx, choices):
            '''lmql
            """{ctx}[ANSWER]""" where ANSWER in set(choices)
            return ANSWER
            '''

        async def call_select(context, choices):
            answer = await program(ctx=context, choices=choices, temperature=0)
            return choices.index(answer)

    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    # Run requests
    if args.backend != "lmql":
        # Use thread pool
        def get_one_answer(i):
            preds[i] = call_select(
                context=few_shot_examples + questions[i],
                choices=choices[i])

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
                for q, c in zip(questions[i:i+batch_size], choices[i:i+batch_size]):
                    tasks.append(call_select(
                        context=few_shot_examples + q,
                        choices=c))
                rets = await asyncio.gather(*tasks)
                for j in range(len(rets)):
                    preds[i+j] = rets[j]

        tic = time.time()
        asyncio.run(batched_call(batch_size=args.parallel))

    latency = time.time() - tic

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    print(f"Latency: {latency:.3f}")
    print(f"Accuracy: {acc:.3f}")

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "hellaswag",
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
    parser.add_argument("--num-shot", type=int, default=20)
    parser.add_argument("--data-path", type=str, default="hellaswag_val.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    args = add_common_other_args_and_parse(parser)
    main(args)

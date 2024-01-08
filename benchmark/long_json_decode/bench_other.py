import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import time

from tqdm import tqdm
import numpy as np
from sglang.test.test_utils import add_common_other_args_and_parse, call_generate_lightllm, call_generate_vllm, call_generate_srt_raw
from sglang.utils import read_jsonl, dump_state_text


def json_decode(document, generate):
    s = "Please extract the information of a city from the following wikipedia page.\n"
    s += "Page begin.\n" + document + "Page end.\n"
    s += "Here is the name, country, and symbol of the city in JSON format.\n"
    s += '{\n'
    s += '  "name": "'
    s += generate(s, max_tokens=8, stop='"') + '",\n'
    s += '  "country": "'
    s += generate(s, max_tokens=8, stop='"') + '",\n'
    s += '  "air port code": "'
    s += generate(s, max_tokens=8, stop='"') + '",\n'
    s += '  "top 3 landmarks": "'
    s += generate(s, max_tokens=24, stop='"') + '",\n'
    s += '}\n'
    return s


def main(args):
    lines = read_jsonl(args.data_path)
    arguments = []
    for i in range(len(lines[:args.num_questions])):
        arguments.append({
            "document": lines[i]["document"],
        })
    states = [None] * len(arguments)

    # Select backend
    if args.backend == "lightllm":
        url = f"{args.host}:{args.port}/generate"
        generate = partial(call_generate_lightllm, url=url, temperature=0)
    elif args.backend == "vllm":
        url = f"{args.host}:{args.port}/generate"
        generate = partial(call_generate_vllm, url=url, temperature=0)
    elif args.backend == "srt-raw":
        url = f"{args.host}:{args.port}/generate"
        generate = partial(call_generate_srt_raw, url=url, temperature=0)
    elif args.backend == "guidance":
        from guidance import models, gen

        model = models.LlamaCpp("/home/ubuntu/model_weights/CodeLlama-7b-instruct-hf.gguf", n_gpu_layers=-1, n_ctx=11000)

        def generate(prompt, max_tokens, stop):
            out = model + prompt + gen(name="answer",
                max_tokens=max_tokens, temperature=0, stop=stop)
            return out["answer"]

        # warmup
        generate("Hello!", max_tokens=8, stop=None)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    # Run requests
    def get_one_answer(i):
        states[i] = json_decode(generate=generate, **arguments[i])

    tic = time.time()
    if args.parallel == 1:
        for i in tqdm(range(len(arguments))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            executor.map(get_one_answer, list(range(len(arguments))))
    latency = time.time() - tic

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "long_json_decode",
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
    parser.add_argument("--data-path", type=str, default="questions.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    args = add_common_other_args_and_parse(parser)
    main(args)

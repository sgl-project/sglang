import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import time

import numpy as np
from tqdm import tqdm
from sglang.test.test_utils import add_common_other_args_and_parse, call_generate_lightllm, call_generate_vllm, call_generate_srt_raw
from sglang.utils import read_jsonl, dump_state_text


system_prompt = (
"Please serve as an impartial judge and rigorously evaluate the quality of the following article. Apply the most stringent standards possible, showing no leniency."
)

dimension_prompts = [
"Content: This refers to the essences of the essay. The substance should be well researched, accurate, relevant to the topic and should show a thorough understanding of the subject. The essay should also reflect a clear goal or purpose.",
"Organization and Structure: An essay needs to be properly structured with a clear introduction, body, and conclusion. The essay should flow naturally, with one paragraph leading seamlessly into the next.",
"Argument and Analysis: The argument made in the essay should be logical, coherent and clearly articulated. Each point made should be backed up by solid evidence and thorough analysis.",
"Clarity and Precision: The essay should be written in a clear and concise manner. The points made should be easily understood by the reader. The language used should also be precise and unambiguous.",
"Grammar and Punctuation: Proper use of grammar and punctuation is vital in an academic essay. Errors in grammar and punctuation not only distract the reader but can also negatively impact the meaning and interpretation of the content.",
"Referencing and Citation: An essay should contain proper citations and references for all sources used. This not only prevents accusations of plagiarism but also gives credit to the authors of the works that have contributed to the essay. The citation should adhere to a specific format as required by the academic institution or specified by the professor.",
]


def multi_dimension_judge(article, generate):
    s = system_prompt
    s += "\n```\n" + article + "\n```\n\n"

    judges = []
    for i in range(len(dimension_prompts)):
        comp = generate(s +
                "USER: Please judge the quality based on the following metric. " +
                dimension_prompts[i] + " Please provide a single-paragraph judgement. " +
                "Focus on the provided metric and do not say other things. " 
                'End your judgement paragraph with the word "END"\nJUDGE:',
            max_tokens=256, stop="END")
        judges.append(comp)

    s += "I will judge the quality based on the following metrics.\n"
    for i in range(len(dimension_prompts)):
        s += dimension_prompts[i].split(":")[0] + ": " + judges[i].strip() + "\n"

    s += "In summary, on a scale of 1 to 10, I would give the article a score of"
    s += generate(s, max_tokens=2, stop=None)

    return s


def main(args):
    lines = read_jsonl(args.data_path)[:args.num_questions]
    states = [None] * len(lines)

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

        model = models.LlamaCpp("/home/ubuntu/model_weights/Llama-2-7b-chat.gguf", n_gpu_layers=-1, n_ctx=4096)

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
        states[i] = multi_dimension_judge(lines[i], generate)

    tic = time.time()
    if args.parallel == 1:
        for i in tqdm(range(len(lines))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            executor.map(get_one_answer, list(range(len(lines))))
    latency = time.time() - tic

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "llm_judge",
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
    parser.add_argument("--data-path", type=str, default="articles.jsonl")
    parser.add_argument("--num-questions", type=int, default=20)
    args = add_common_other_args_and_parse(parser)
    main(args)

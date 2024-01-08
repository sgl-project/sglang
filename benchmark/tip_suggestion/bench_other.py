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


number = 5


def expand_tip(topic, tip, generate):
    s = (
"""Please expand a tip for a topic into a detailed paragraph.

Topic: staying healthy
Tip: Regular Exercise
Paragraph: Incorporate physical activity into your daily routine. This doesn't necessarily mean intense gym workouts; it can be as simple as walking, cycling, or yoga. Regular exercise helps in maintaining a healthy weight, improves cardiovascular health, boosts mental health, and can enhance cognitive function, which is crucial for fields that require intense intellectual engagement.

Topic: building a campfire
Tip: Choose the Right Location
Paragraph: Always build your campfire in a safe spot. This means selecting a location that's away from trees, bushes, and other flammable materials. Ideally, use a fire ring if available. If you're building a fire pit, it should be on bare soil or on a bed of stones, not on grass or near roots which can catch fire underground. Make sure the area above is clear of low-hanging branches.

Topic: writing a blog post
Tip: structure your content effectively
Paragraph: A well-structured post is easier to read and more enjoyable. Start with an engaging introduction that hooks the reader and clearly states the purpose of your post. Use headings and subheadings to break up the text and guide readers through your content. Bullet points and numbered lists can make information more digestible. Ensure each paragraph flows logically into the next, and conclude with a summary or call-to-action that encourages reader engagement.

Topic: """ + topic + "\nTip: " + tip + "\nParagraph:")
    return generate(s, max_tokens=128, stop=["\n\n"])


def suggest_tips(topic, generate):
    s = "Please act as a helpful assistant. Your job is to provide users with useful tips on a specific topic.\n"
    s += "USER: Give some tips for " + topic + ".\n"
    s += ("ASSISTANT: Okay. Here are " + str(number) + " concise tips, each under 8 words:\n")

    tips = []
    for i in range(1, 1 + number):
        s += f"{i}."
        tip = generate(s, max_tokens=24, stop=[".", "\n"])
        s += tip + ".\n"
        tips.append(tip)

    paragraphs = [expand_tip(topic, tip, generate=generate) for tip in tips]

    for i in range(1, 1 + number):
        s += f"Tip {i}:" + paragraphs[i-1] + "\n"
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
        states[i] = suggest_tips(lines[i]["topic"], generate)

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
            "task": "tip_suggestion",
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
    parser.add_argument("--data-path", type=str, default="topic.jsonl")
    parser.add_argument("--num-questions", type=int, default=100)
    args = add_common_other_args_and_parse(parser)
    main(args)

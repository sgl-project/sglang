import itertools
import json
import os
import random
import string
import threading
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from tqdm import tqdm

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


def gen_prompt(tokenizer, token_num):
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    ret = tokenizer.decode(selected_tokens)
    return ret


def get_cache_path(args):
    # Create cache directory under ~/.cache/sglang
    cache_dir = Path.home() / ".cache" / "sglang"

    # Create a unique cache filename based on the arguments that affect generation
    cache_key = f"qa_{args.num_qa}_{args.turns}_{args.system_prompt_len}_{args.len_q}_{args.len_a}_{args.tokenizer.replace('/', '_')}.json"
    return cache_dir / cache_key


def gen_arguments(args, tokenizer):
    cache_path = get_cache_path(args)

    # Try to load from cache first
    if cache_path.exists():
        print(f"Loading cached arguments from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    print("Generating new arguments...")
    # First progress bar for system prompts
    multi_qas = []
    for _ in tqdm(range(args.num_qa), desc="Generating system prompts"):
        multi_qas.append(
            {"system_prompt": gen_prompt(tokenizer, args.system_prompt_len), "qas": []}
        )

    # Nested progress bars for QA pairs
    for i in tqdm(range(args.num_qa), desc="Generating QA pairs"):
        qas = multi_qas[i]["qas"]
        for j in range(args.turns):
            qas.append(
                {
                    "prompt": gen_prompt(tokenizer, args.len_q),
                    "new_tokens": args.len_a,
                }
            )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(multi_qas, f)
    print(f"Cached arguments saved to {cache_path}")

    return multi_qas


@sgl.function
def multi_turns(s, system_prompt, qas):
    s += system_prompt

    for i, qa in enumerate(qas):
        s += qa["prompt"]
        s += sgl.gen(max_tokens=qa["new_tokens"], ignore_eos=True)


def main(args):
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)

    multi_qas = gen_arguments(args, tokenizer)

    backend = select_sglang_backend(args)

    tic = time.time()
    states = multi_turns.run_batch(
        multi_qas,
        temperature=0,
        backend=backend,
        num_threads="auto",
        progress_bar=True,
    )
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "multi_turn_system_prompt_chat",
            "backend": args.backend,
            "latency": round(latency, 3),
            "num_requests": args.num_qa,
            "num_turns": args.turns,
            "other": {
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--num-qa", type=int, default=128)
    parser.add_argument("--system-prompt-len", type=int, default=2048)
    parser.add_argument("--len-q", type=int, default=32)
    parser.add_argument("--len-a", type=int, default=128)
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = add_common_sglang_args_and_parse(parser)

    print(args)
    main(args)

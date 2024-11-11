import itertools
import json
import random
import string
import threading
import time
from argparse import ArgumentParser

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenize
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text

random.seed(42)


def gen_prompt(tokenizer, token_num):
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    ret = tokenizer.decode(selected_tokens)
    return ret


def gen_arguments(args, tokenizer):
    multi_qas = [
        {"system_prompt": gen_prompt(tokenizer, args.system_prompt_len), "qas": []}
        for _ in range(args.num_qa)
    ]
    for i in range(args.num_qa):
        qas = multi_qas[i]["qas"]
        for j in range(args.turns):
            qas.append(
                {
                    "prompt": gen_prompt(tokenizer, args.len_q),
                    "new_tokens": args.len_a,
                }
            )
    return multi_qas


@sgl.function
def multi_turns(s, system_prompt, qas):
    s += system_prompt

    for qa in qas:
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
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "multi_turn_system_prompt_chat",
            "backend": args.backend,
            "num_gpus": 1,
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

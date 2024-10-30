import itertools
import json
import threading
import time
from argparse import ArgumentParser

from vllm.transformers_utils.tokenizer import get_tokenizer

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text

import random
import string

random.seed(42)


def gen_prompt(tokenizer, token_num):
    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=token_num))
    # print("token_num", token_num)
    # import pdb; pdb.set_trace()
    while len(tokenizer(ret).input_ids) < token_num:
        ret += random.choice(cha_set)
    print("actual token num (input id)", len(tokenizer(ret).input_ids))
    print("actual token num (encode)", len(tokenizer.encode(ret)))
    return ret


def gen_arguments(args, tokenizer):
    multi_qas = [{"qas": []} for _ in range(args.num_qa)]
    for i in range(args.num_qa):
        # print("hi")
        qas = multi_qas[i]["qas"]
        for j in range(args.turns):
            prompt_len = random.randint(args.min_len_q, args.max_len_q)
            # if j == 0:
            #     prompt_len *= 3
            new_tokens = random.randint(args.min_len_a, args.max_len_a)

            print("prompt_len", prompt_len)
            print("new_tokens", new_tokens)
            qas.append(
                {
                    "prompt": gen_prompt(tokenizer, prompt_len),
                    "new_tokens": new_tokens,
                }
            )
    #import pdb; pdb.set_trace()
    return multi_qas

@sgl.function
def multi_turns(s, qas):
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
            "task": "multi_turn_chat",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_qa,
            "num_turns": args.turns,
            "other": {
                "parallel": args.parallel,
                "output_mode": "long" if args.long else "short",
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=1)
    parser.add_argument("--min-len-q", type=int, default=512)
    parser.add_argument("--max-len-q", type=int, default=1024)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = add_common_sglang_args_and_parse(parser)

    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20

    print(args)
    main(args)
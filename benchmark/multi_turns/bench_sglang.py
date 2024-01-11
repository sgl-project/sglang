from argparse import ArgumentParser
import random
from vllm.transformers_utils.tokenizer import get_tokenizer
from sglang.utils import dump_state_text
import json

import string
import time

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def gen_prompt(tokenizer, token_num):
    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=token_num))
    while len(tokenizer(ret).input_ids) < token_num:
        ret += random.choice(cha_set)
    return ret


def gen_arguments(args, tokenizer):
    multi_qas = [{} for _ in range(args.num_qa)]
    for i in range(args.num_qa):
        prompt_len = random.randint(args.min_len, args.max_len)
        new_tokens = random.randint(args.min_len, args.max_len)
        multi_qas[i] = {
            "prompt": gen_prompt(tokenizer, prompt_len),
            "new_tokens": new_tokens,
        }

    return multi_qas


def main(args):
    print(args)
    random.seed(args.seed)

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)

    multi_qas = gen_arguments(args, tokenizer)

    backend = select_sglang_backend(args)

    @sgl.function
    def multi_turns(s, prompt, new_tokens):
        for _ in range(args.turns):
            s += prompt
            s += sgl.gen(
                max_tokens=new_tokens
                if isinstance(new_tokens, int)
                else new_tokens.value
            )

    tic = time.time()
    states = multi_turns.run_batch(
        multi_qas, temperature=0, backend=backend, num_threads=args.parallel
    )
    for state in states:
        state.sync()
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "multi_turns",
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
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--num-qa", type=int, default=10)
    parser.add_argument("--min-len", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = add_common_sglang_args_and_parse(parser)
    main(args)

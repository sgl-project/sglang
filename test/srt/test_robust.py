import argparse
import random
import string

from vllm.transformers_utils.tokenizer import get_tokenizer

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)

TOKENIZER = None
RANDOM_PREFILL_LEN = None
RANDOM_DECODE_LEN = None


def gen_prompt(token_num):
    if RANDOM_PREFILL_LEN:
        token_num = random.randint(1, token_num)

    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=token_num))
    while len(TOKENIZER(ret).input_ids) < token_num:
        ret += random.choice(cha_set)

    return ret


def robust_test_dfs(s, d, args, leaf_states):
    if d == 0:
        s += "END"
        leaf_states.append(s)
        return

    s += gen_prompt(args.len_prefill)
    forks = s.fork(args.num_fork)
    for fork_s in forks:
        fork_s += gen_prompt(args.len_prefill)
        new_tokens = (
            args.len_decode
            if not RANDOM_DECODE_LEN
            else random.randint(1, args.len_decode)
        )
        fork_s += sgl.gen(
            max_tokens=new_tokens,
            ignore_eos=True,
        )

    for fork_s in forks:
        robust_test_dfs(fork_s, d - 1, args, leaf_states)


def robust_test_bfs(s, args, leaf_states):
    old_forks = [s]
    new_forks = []
    for _ in range(args.depth):
        for old_fork in old_forks:
            old_fork += gen_prompt(args.len_prefill)
            forks = old_fork.fork(args.num_fork)
            for fork_s in forks:
                fork_s += gen_prompt(args.len_prefill)
                new_tokens = (
                    args.len_decode
                    if not RANDOM_DECODE_LEN
                    else random.randint(1, args.len_decode)
                )
                fork_s += sgl.gen(
                    max_tokens=new_tokens,
                    ignore_eos=True,
                )
            new_forks.extend(forks)

        old_forks = new_forks
        new_forks = []

    for old_fork in old_forks:
        old_fork += "END"
        leaf_states.append(old_fork)


@sgl.function
def robust_test(s, args):
    leaf_states = []
    if args.mode == "bfs":
        robust_test_bfs(s, args, leaf_states)
    else:
        robust_test_dfs(s, args.depth, args, leaf_states)
    return leaf_states


def main(args):
    backend = select_sglang_backend(args)

    arguments = [{"args": args} for _ in range(args.num_req)]

    states = robust_test.run_batch(
        arguments, temperature=0, backend=backend, num_threads=args.parallel
    )

    with open(f"tmp_robust_{args.mode}.txt", "w") as f:
        for state in states:
            leaf_states = state.ret_value
            for leaf_state in leaf_states:
                assert leaf_state.text()[-3:] == "END"
                f.write(leaf_state.text()[:-3] + "\n")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-req", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-fork", type=int, default=2)
    parser.add_argument("--len-prefill", type=int, default=128)
    parser.add_argument("--len-decode", type=int, default=128)
    parser.add_argument("--random-prefill-len", action="store_true")
    parser.add_argument("--random-decode-len", action="store_true")
    parser.add_argument("--mode", type=str, default="bfs", choices=["dfs", "bfs"])
    parser.add_argument("--tokenizer", type=str, default = "meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = add_common_sglang_args_and_parse(parser)
    # fmt: on

    RANDOM_PREFILL_LEN = args.random_prefill_len
    RANDOM_DECODE_LEN = args.random_decode_len
    TOKENIZER = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)

    random.seed(args.seed)

    main(args)

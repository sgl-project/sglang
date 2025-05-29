import json
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from data_gen import gen_arguments
from tqdm import tqdm
from vllm.transformers_utils.tokenizer import get_tokenizer

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import dump_state_text


def multi_turns(generate, qas):
    s = ""
    for qa in qas:
        s += qa["prompt"]
        s += generate(s, max_tokens=qa["new_tokens"])

    return s


def main(args):
    print(args)

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)

    multi_qas = gen_arguments(args, tokenizer)

    states = [None] * args.num_qa

    call_generate = partial(get_call_generate(args), temperature=0)

    def get_one_answer(i):
        states[i] = multi_turns(generate=call_generate, **multi_qas[i])

    tic = time.perf_counter()
    if args.parallel == 1:
        for i in tqdm(range(len(multi_qas))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            rets = list(
                tqdm(
                    executor.map(get_one_answer, list(range(len(multi_qas)))),
                    total=len(multi_qas),
                )
            )
            for _ in rets:
                pass

    latency = time.perf_counter() - tic

    # Compute accuracy
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
    parser.add_argument("--num-qa", type=int, default=20)
    parser.add_argument("--min-len-q", type=int, default=256)
    parser.add_argument("--max-len-q", type=int, default=512)
    parser.add_argument("--min-len-a", type=int, default=4)
    parser.add_argument("--max-len-a", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = add_common_other_args_and_parse(parser)

    if args.long:
        args.min_len_a = 256
        args.max_len_a = 512
        args.num_qa = 20
    main(args)

import json
import random
import string
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import requests
from sglang.test.test_utils import add_common_other_args_and_parse
from sglang.utils import dump_state_text
from tqdm import tqdm
from vllm.transformers_utils.tokenizer import get_tokenizer


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

    # Select backend
    if args.backend == "vllm":
        url = f"{args.host}:{args.port}/generate"

        def generate(prompt, max_tokens, stop=None, temperature=0, url=url, n=1):
            data = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stop": stop,
                "stream": False,
                "n": n,
            }
            res = requests.post(url, json=data)
            assert res.status_code == 200
            return res.json()["text"][0][len(prompt) :]

    elif args.backend == "guidance":
        from guidance import gen, models

        model = models.LlamaCpp(
            "/home/ubuntu/model_weights/Llama-2-7b-chat-hf/ggml-model-f16.gguf",
            n_gpu_layers=-1,
            n_ctx=4096,
        )

        def generate(prompt, max_tokens, stop=None):
            out = (
                model
                + prompt
                + gen(name="answer", max_tokens=max_tokens, temperature=0, stop=stop)
            )
            return out["answer"]

        # warmup
        generate("Hello!", max_tokens=8, stop=None)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    def multi_turns(generate, prompt, new_tokens):
        s = ""
        for _ in range(args.turns):
            s += prompt
            s += generate(s, max_tokens=new_tokens)
        return s

    states = [None] * args.num_qa

    def get_one_answer(i):
        states[i] = multi_turns(generate=generate, **multi_qas[i])

    tic = time.time()
    if args.parallel == 1:
        for i in tqdm(range(len(multi_qas))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            rets = executor.map(get_one_answer, list(range(len(multi_qas))))
            for _ in rets:
                pass

    latency = time.time() - tic

    # Compute accuracy
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
    args = add_common_other_args_and_parse(parser)
    main(args)

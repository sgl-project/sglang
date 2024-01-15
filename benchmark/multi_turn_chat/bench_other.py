import json
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import requests
from sglang.test.test_utils import add_common_other_args_and_parse
from sglang.utils import dump_state_text
from tqdm import tqdm
from vllm.transformers_utils.tokenizer import get_tokenizer

from data_gen import gen_arguments


def get_generate(args):
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
        for _ in range(3):
            generate("Hello!" * 10, max_tokens=64, stop=None)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    return generate


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

    generate = get_generate(args)

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

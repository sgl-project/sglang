import argparse
from functools import partial
import json
import time
from pathlib import Path

from tqdm import tqdm
from sglang.test.test_utils import (
    add_common_other_args_and_parse,
    call_generate_lightllm,
    call_generate_vllm,
    call_generate_srt_raw,
)
from sglang.utils import read_jsonl, dump_state_text

from agent_functions import (
    poignancy_event_prompt,
    generate_event_triple_prompt,
    generate_pronunciatio_prompt,
    action_location_sector_prompt,
    action_location_object_prompt,
)


def main(args):
    lines = read_jsonl(args.data_path)[:args.num_events]
    mapping = {
        "poignancy_event": poignancy_event_prompt,
        "generate_event_triple": generate_event_triple_prompt,
        "generate_pronunciatio": generate_pronunciatio_prompt,
        "action_location_sector": action_location_sector_prompt,
        "action_location_object": action_location_object_prompt,
    }

    arguments = [mapping[k](**v) for l in lines for k, v in l.items()]
    states = []

    # Select backend
    if args.backend == "lightllm":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_lightllm, url=url)
    elif args.backend == "vllm":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_vllm, url=url)
    elif args.backend == "srt-raw":
        url = f"{args.host}:{args.port}/generate"
        call_generate = partial(call_generate_srt_raw, url=url)
    elif args.backend == "guidance":
        from guidance import models, gen

        model = models.LlamaCpp(
            str(Path.home()) + "/model_weights/Llama-2-7b-chat.gguf",
            n_gpu_layers=-1,
            n_ctx=4096,
        )

        def call_generate(prompt, temperature, max_tokens, stop):
            out = model + prompt + gen(
                name="result",
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            return out["result"]

    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    def get_one_answer(arg):
        answer = call_generate(**arg, temperature=0)
        states.append(answer)

    tic = time.time()
    # we always sequentially execute agent calls to maintain its dependency
    for arg in tqdm(arguments):
        get_one_answer(arg)
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "Generative Agents",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            # to pack weighted functions as a single agent            
            "num_requests": len(arguments) / len(mapping),
            "other": {
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="agent_calls.jsonl")
    parser.add_argument("--num-events", type=int, default=10)
    args = add_common_other_args_and_parse(parser)
    main(args)

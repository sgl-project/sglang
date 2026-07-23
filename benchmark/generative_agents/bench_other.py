import argparse
import json
import time

from agent_functions import (
    action_location_object_prompt,
    action_location_sector_prompt,
    generate_event_triple_prompt,
    generate_pronunciatio_prompt,
    poignancy_event_prompt,
)
from tqdm import tqdm

from sglang.test.test_utils import add_common_other_args_and_parse, get_call_generate
from sglang.utils import dump_state_text, read_jsonl


def main(args):
    lines = read_jsonl(args.data_path)[: args.num_events]
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
    call_generate = get_call_generate(args)

    def get_one_answer(arg):
        answer = call_generate(**arg, temperature=0)
        states.append(answer)

    async def get_one_answer_async(arg):
        answer = await call_generate(**arg, temperature=0)
        states.append(answer)

    tic = time.perf_counter()
    # we always sequentially execute agent calls to maintain its dependency
    if args.backend != "lmql":
        for arg in tqdm(arguments):
            get_one_answer(arg)
    else:
        import asyncio

        loop = asyncio.get_event_loop()
        for arg in tqdm(arguments):
            loop.run_until_complete(get_one_answer_async(arg))
    latency = time.perf_counter() - tic

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

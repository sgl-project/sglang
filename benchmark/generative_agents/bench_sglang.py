import argparse
import json
import time

from agent_functions import (
    action_location_object,
    action_location_sector,
    generate_event_triple,
    generate_pronunciatio,
    poignancy_event,
)

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, read_jsonl


def main(args):
    lines = read_jsonl(args.data_path)[: args.num_events]
    mapping = {
        "poignancy_event": poignancy_event,
        "generate_event_triple": generate_event_triple,
        "generate_pronunciatio": generate_pronunciatio,
        "action_location_sector": action_location_sector,
        "action_location_object": action_location_object,
    }
    arguments = [{mapping[k]: v for k, v in l.items()} for l in lines]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    states = []
    # Run requests
    tic = time.perf_counter()
    for a in arguments:
        # only a single key in the dict
        for func, arg in a.items():
            result = func.run(**arg)
        result.sync()
        states.append(result)
    latency = time.perf_counter() - tic

    # Compute accuracy
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
                "num_events": args.num_events,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="agent_calls.jsonl")
    parser.add_argument("--num-events", type=int, default=10)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

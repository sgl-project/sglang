import argparse
import json
import time

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text

# there are some FSM bugs with json regex converted from pydantic model
# here use a string regex instead
# regex_string = build_regex_from_object(HarryPoterRole)
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)

# fmt: off
@sgl.function
def character_gen(s, name):
    s += name + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)
# fmt: on


def bench_character(args):
    arguments = []
    with open(args.data_path, "r") as f:
        for line in f:
            arguments.append({"name": line.strip()})
    arguments = arguments[: args.num_jsons]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    states = character_gen.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=(args.parallel == 1),
    )
    latency = time.time() - tic

    return states, latency


def main(args):
    states, latency = bench_character(args)

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    with open(f"{args.backend}.json", "w") as fout:
        for state in states:
            fout.write(state["json_output"] + "\n")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "json_fast_forward",
            "backend": args.backend,
            "latency": round(latency, 3),
            "num_jsons": args.num_jsons,
            "parallel": args.parallel,
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset.txt")
    parser.add_argument("--num-jsons", type=int, default=50)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

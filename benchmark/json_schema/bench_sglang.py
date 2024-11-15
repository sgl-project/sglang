import argparse
import json
import time
from typing import Tuple

import jsonschema
from datasets import load_dataset

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


@sgl.function
def schema_gen(s, message: Tuple[str, str], json_schema: str):
    system, user = message
    s += sgl.system(system)
    s += sgl.user(user)
    s += sgl.assistant(
        sgl.gen("json_output", temperature=0, max_tokens=256, json_schema=json_schema)
    )


def convert_dataset(path: str):
    raw_dataset = load_dataset(path)
    dataset = []
    for data in raw_dataset["train"]:
        messages = data["prompt"]
        schema = data["schema"]
        obj = json.loads(schema)
        if obj.get("type", None) is None:
            continue
        system = messages[0]
        user = messages[1]
        assert system["role"] == "system", "invalid role"
        assert user["role"] == "user", "invalid role"
        assert len(messages) == 2, "invalid message length"
        message = json.dumps(system["content"]), json.dumps(user["content"])
        dataset.append(
            {
                "message": message,
                "json_schema": schema,
            }
        )

    # skip a corrupted example
    dataset = dataset[:53] + dataset[54:]

    return dataset


def bench_schema(args):
    arguments = convert_dataset(args.data_path)

    if args.num_jsons < 0 or args.num_jsons > len(arguments):
        args.num_jsons = len(arguments)
    arguments = arguments[: args.num_jsons]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    states = schema_gen.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.time() - tic

    # Check if the outputs are valid
    indexs = []
    for i, state in enumerate(states):
        try:
            schema = json.loads(arguments[i]["json_schema"])
            obj = json.loads(state["json_output"])
            assert jsonschema.validate(obj, schema) is None
        except Exception as e:
            print(e)
            indexs.append(i)

    assert len(indexs) == 0, f"Invalid json outputs: {indexs}"
    return states, latency


def main(args):
    states, latency = bench_schema(args)

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    with open(f"{args.backend}.json", "w") as fout:
        for state in states:
            fout.write(state["json_output"] + "\n")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "json_schema",
            "backend": args.backend,
            "latency": round(latency, 3),
            "num_jsons": args.num_jsons,
            "parallel": args.parallel,
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="NousResearch/json-mode-eval")
    parser.add_argument("--num-jsons", type=int, default=-1)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

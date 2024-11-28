import argparse
import json
import time
from typing import List, Tuple

import jsonschema
from datasets import load_dataset

import sglang as sgl
from sglang.global_config import global_config
from sglang.srt.hf_transformers_utils import get_tokenizer
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


def contains_formats(schema, formats: List[str]):
    if isinstance(schema, dict):
        if schema.get("format", None) in formats:
            return True
        for value in schema.values():
            if contains_formats(value, formats):
                return True
    elif isinstance(schema, list):
        for item in schema:
            if contains_formats(item, formats):
                return True
    return False


def convert_dataset(path: str):
    raw_dataset = load_dataset(path)
    dataset = []
    for data in raw_dataset["train"]:
        messages = data["prompt"]
        schema = data["schema"]
        obj = json.loads(schema)

        # skip some corrupted examples
        if obj.get("type", None) is None:
            continue

        # skip schema with format "email"
        # which is not supported by outlines for now
        if contains_formats(obj, ["email"]):
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

    return states, latency


def main(args):
    states, latency = bench_schema(args)

    # Compute accuracy
    tokenizer = get_tokenizer(
        global_config.default_backend.get_server_info()["tokenizer_path"]
    )
    output_jsons = [state["json_output"] for state in states]
    num_output_tokens = sum(len(tokenizer.encode(x)) for x in output_jsons)
    print(f"Latency: {latency:.3f}")
    print(f"Output throughput: {num_output_tokens / latency:.3f} token/s")
    print(f"#output tokens: {num_output_tokens}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    with open(f"{args.backend}.jsonl", "w") as fout:
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

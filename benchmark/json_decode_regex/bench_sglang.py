import argparse
import json
import time

import sglang as sgl
from sglang.lang.ir import REGEX_INT, REGEX_STRING, REGEX_FLOAT
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, read_jsonl

REGEX_LIST = r"\[(" + REGEX_STRING + ", )*" + REGEX_STRING + r"\]"

# fmt: off
@sgl.function
def json_warm_up(s):
    s += "The information about Hogwarts is in the following JSON format.\n"
    with s.var_scope("json_output"):
        s += "{\n"
        s += '  "name": ' + sgl.gen("name", max_tokens=8, regex=REGEX_STRING + ",") + "\n"
        s += '  "country": ' + sgl.gen("country", max_tokens=8, regex=REGEX_STRING + ",") + "\n"
        s += '  "latitude": ' + sgl.gen("latitude", max_tokens=8, regex=REGEX_FLOAT + ",") + "\n"
        s += '  "population": ' + sgl.gen("population", max_tokens=8, regex=REGEX_INT + ",") + "\n"
        s += '  "top 3 landmarks": ' + sgl.gen( "landmarks", max_tokens=24, regex=REGEX_LIST) + "\n"
        s += "}\n"
    print(f'The warmp up json result is:\n{s["json_output"]}')
# fmt: on

# fmt: off
@sgl.function
def json_decode(s, document):
    s += "Please extract the information of a city from the following wikipedia page.\n"
    s += "Page begin.\n" + document + "Page end.\n"
    s += "Here is the name, country, and symbol of the city in JSON format.\n"
    with s.var_scope("json_output"):
        s += "{\n"
        s += '  "name": ' + sgl.gen("name", max_tokens=8, regex=REGEX_STRING + ",") + "\n"
        s += '  "country": ' + sgl.gen("country", max_tokens=8, regex=REGEX_STRING + ",") + "\n"
        s += '  "latitude": ' + sgl.gen("latitude", max_tokens=8, regex=REGEX_FLOAT + ",") + "\n"
        s += '  "population": ' + sgl.gen("population", max_tokens=8, regex=REGEX_INT + ",") + "\n"
        s += '  "top 3 landmarks": ' + sgl.gen( "landmarks", max_tokens=24, regex=REGEX_LIST) + "\n"
        s += "}\n"
# fmt: on


def main(args):
    lines = read_jsonl(args.data_path)
    arguments = []
    for i in range(len(lines[: args.num_questions])):
        arguments.append(
            {
                "document": lines[i]["document"],
            }
        )

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Warm up
    json_warm_up.run().sync()

    # Run requests
    tic = time.time()
    states = json_decode.run_batch(arguments, temperature=0, num_threads=args.parallel)
    latency = time.time() - tic

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(f"tmp_{args.backend}_json_results.txt", "w") as fout:
        for state in states:
            fout.write(state["json_output"] + "\n")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "json_decode_regex",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_questions,
            "other": {
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="questions.jsonl")
    parser.add_argument("--num-questions", type=int, default=20)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

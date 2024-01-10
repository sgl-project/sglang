import argparse
import json
import time

import numpy as np
import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import read_jsonl, dump_state_text


@sgl.function
def json_decode(s, document):
    s += "Please extract the information of a city from the following wikipedia page.\n"
    s += "Page begin.\n" + document + "Page end.\n"
    s += "Here is the name, country, and symbol of the city in JSON format.\n"
    s += '{\n'
    s += '  "name": "' + sgl.gen("name", max_tokens=8, stop='"') + '",\n'
    s += '  "country": "' + sgl.gen("country", max_tokens=8, stop='"') + '",\n'
    s += '  "air port code": "' + sgl.gen("air port code", max_tokens=8, stop='"') + '",\n'
    s += '  "top 3 landmarks": "' + sgl.gen("landmarks", max_tokens=24, stop='"') + '",\n'
    s += '}\n'


def main(args):
    lines = read_jsonl(args.data_path)
    arguments = []
    for i in range(len(lines[:args.num_questions])):
        arguments.append({
            "document": lines[i]["document"],
        })

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    states = json_decode.run_batch(
        arguments, temperature=0, num_threads=args.parallel)
    latency = time.time() - tic

    # Compute accuracy 
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "long_json_decode",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            }
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="questions.jsonl")
    parser.add_argument("--num-questions", type=int, default=10)
    args = add_common_sglang_args_and_parse(parser)
    main(args)

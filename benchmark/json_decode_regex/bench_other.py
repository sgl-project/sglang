import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sglang.test.test_utils import (
    add_common_other_args_and_parse,
    call_generate_outlines,
)
from sglang.utils import dump_state_text, read_jsonl
from sglang.lang.ir import REGEX_INT, REGEX_STRING, REGEX_FLOAT
from tqdm import tqdm

REGEX_LIST = r"\[(" + REGEX_STRING + ", )*" + REGEX_STRING + r"\]"


# fmt: off
def json_decode(document, generate):
    s = "Please extract the information of a city from the following wikipedia page.\n"
    s += "Page begin.\n" + document + "Page end.\n"
    s += "Here is the name, country, and symbol of the city in JSON format.\n"
    s += "{\n"
    s += '  "name": '
    s += generate(s, max_tokens=8, regex=REGEX_STRING + ",") + "\n"
    s += '  "country": '
    s += generate(s, max_tokens=8, regex=REGEX_STRING + ",") + "\n"
    s += '  "latitude": '
    s += generate(s, max_tokens=8, regex=REGEX_FLOAT + ",") + "\n"
    s += '  "population": '
    s += generate(s, max_tokens=8, regex=REGEX_INT + ",") + "\n"
    s += '  "top 3 landmarks": '
    s += generate(s, max_tokens=24, regex=REGEX_LIST) + "\n"
    s += "}\n"

    return s
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
    states = [None] * len(arguments)

    # Select backend
    if args.backend == "vllm":
        url = f"{args.host}:{args.port}/generate"
        generate = partial(call_generate_outlines, url=url, temperature=0)
    elif args.backend == "guidance":
        from guidance import gen, models

        model = models.LlamaCpp(
            "/home/ubuntu/model_weights/Llama-2-7b-chat-hf/ggml-model-f16.gguf",
            n_gpu_layers=-1,
            n_ctx=4096,
        )

        def generate(prompt, max_tokens, stop=None, regex=None):
            out = (
                model
                + prompt
                + gen(
                    name="answer",
                    max_tokens=max_tokens,
                    temperature=0,
                    stop=stop,
                    regex=regex,
                )
            )
            return out["answer"]

        # warmup
        for _ in range(3):
            generate("Hello!" * 10, max_tokens=64, stop=None)
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    # Run requests
    def get_one_answer(i):
        states[i] = json_decode(generate=generate, **arguments[i])

    tic = time.time()
    if args.parallel == 1:
        for i in tqdm(range(len(arguments))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            rets = executor.map(get_one_answer, list(range(len(arguments))))
            for _ in rets:
                pass

    latency = time.time() - tic

    # Compute accuracy
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

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
    args = add_common_other_args_and_parse(parser)
    main(args)

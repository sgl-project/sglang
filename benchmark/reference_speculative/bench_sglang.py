import time
import datasets
import argparse
import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


@sgl.function
def hagrid_func(s, quotes, question):
    s += "Please answer a question based on the following quotes.\n"
    s += "The quotes are:\n\n"
    for i, quote in enumerate(quotes):
        s += f"Quote{i}: {quote}\n\n"
    s += "The questions is: " + question + "\n\n"
    s += "Answer: Based on the quotes provided," + sgl.gen(
        "answer", max_tokens=512, temperature=0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    args = add_common_sglang_args_and_parse(parser)

    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    dataset = datasets.load_dataset(
        "miracl/hagrid", split="dev", trust_remote_code=True
    )
    dataset = dataset.select(range(args.num))

    arguments = [
        {"quotes": [q["text"] for q in d["quotes"]], "question": d["query"]}
        for d in dataset
    ]

    tic = time.time()

    states = hagrid_func.run_batch(
        arguments, num_threads=args.parallel, progress_bar=True
    )

    print(f"Latency: {time.time() - tic:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)


if __name__ == "__main__":
    main()

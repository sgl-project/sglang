"""Flush cache in the backend by sending random requests."""
import argparse
import random
import string
import time

from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)

import sglang as sgl


@sgl.function
def flush_radix_cache(s, prompt):
    s += prompt + sgl.gen("flush", max_tokens=1, stop="END")


def main(args, max_total_tokens, context_length, print_flag):
    backend = select_sglang_backend(args)
    flush_length = int(context_length * 0.8)
    batch_size = int(max_total_tokens / flush_length)
    prompt_length = flush_length * 2
    prompts = [
        " ".join(random.choices(string.ascii_letters, k=int(prompt_length)))
        for _ in range(batch_size)
    ]
    arguments = [{"prompt": prompts[i]} for i in range(batch_size)]

    start_time = time.time()
    flush_radix_cache.run_batch(
        arguments, temperature=0, backend=backend, num_threads=1
    )
    end_time = time.time()

    if print_flag:
        print(
            f"Flush length: {flush_length}\n",
            f"Prompt length: {prompt_length}\n",
            f"Total Prompt letters: {batch_size * prompt_length}\n",
            f"Flush radix cache latency: {end_time - start_time:.3f}",
            sep="",
        )

    # to prevent the backend still running
    time.sleep(1)


def run_flush(args, max_total_tokens=20000, context_length=1024, print_flag=False):
    main(args, max_total_tokens, context_length, print_flag=print_flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-total-tokens", type=int, default=20000)
    parser.add_argument("--context-length", type=int, default=1024)
    args = add_common_sglang_args_and_parse(parser)
    random.seed(0)
    main(args, args.max_total_tokens, args.context_length, print_flag=True)

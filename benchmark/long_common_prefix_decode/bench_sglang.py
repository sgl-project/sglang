# Benchmark with long common prefixes. Used to benchmark cascade attention performance.
import os
import random
import string
import sys
import time

from tqdm import tqdm
from transformers import AutoTokenizer

import sglang as sgl
from sglang import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def generate_unique_suffix():
    tasks = ["bubble sort", "quick sort", "merge sort", "tarverse tree", "Dijkstra"]
    languages = ["Python", "Java", "C++", "JavaScript", "Rust", "Go"]
    suffixes = []
    for task in  tasks:
        for language in languages:
            suffixes.append(f"write a {task} program in {language}")
    suffixes *= 10
    return suffixes


@sgl.function
def text_gen(s, system, user):
    s += "System: " + system + "\n"
    s += "User: " + user + "\n"
    s += "Assisstant:" + sgl.gen("answer", temperature=0, max_tokens=256)


def test_send_all(sys_prompt, user_prompts):
    backend.flush_cache()

    tic = time.time()
    text_gen.run_batch(
        list(zip([sys_prompt] * len(user_prompts), user_prompts)),
    )
    tot_time = time.time() - tic

    return tot_time


if __name__ == "__main__":
    backend = RuntimeEndpoint("http://127.0.0.1:30000")
    set_default_backend(backend)
    with open('claude_system_prompt.md', 'r') as file:
        prefix = file.read()
    suffixes = generate_unique_suffix()
    print("Start warming up......")
    cost = test_send_all(prefix, suffixes[:8])
    print(f"Latency of warm up: {cost:.4f} s\n")
    cost = test_send_all(prefix, suffixes)
    print(f"Latency of test_send_all: {cost:.4f} s\n")

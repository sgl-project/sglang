import os
import random
import time

import requests


def gen_radix_tree(num_nodes=400, chunk_len=256):
    # Use seed from environment or generate one for reproducibility
    seed = int(os.environ.get("TEST_SEED", int(time.time() * 1000) % (2**31)))
    random.seed(seed)
    print(f"[DEBUG radix_cache] gen_radix_tree using seed={seed}")
    num0 = num_nodes // 2
    num1 = num_nodes - num0
    nodes = [{"input_ids": [37] * 117, "decode_len": 217}]
    for _ in range(num0):
        parent = random.choice(nodes)
        unique_len = random.randint(0, chunk_len)
        decode_len = random.randint(0, chunk_len)
        token_id = random.randint(0, 32000)
        child = {
            "input_ids": parent["input_ids"] + [token_id] * unique_len,
            "decode_len": decode_len,
        }
        nodes.append(child)

    while num1 > 0:
        num_branch = random.randint(1, min(num1, 10))
        parent = random.choice(nodes)
        for _ in range(num_branch):
            unique_len = random.randint(0, chunk_len)
            decode_len = random.randint(0, chunk_len)
            token_id = random.randint(0, 32000)
            child = {
                "input_ids": parent["input_ids"] + [token_id] * unique_len,
                "decode_len": decode_len,
            }
            nodes.append(child)

        num1 -= num_branch

    random.shuffle(nodes)
    return nodes


def run_radix_attention_test(base_url: str):
    nodes = gen_radix_tree()

    # Debug: log test data statistics
    input_lens = [len(node["input_ids"]) for node in nodes]
    decode_lens = [node["decode_len"] for node in nodes]
    print(
        f"[DEBUG radix_cache] num_nodes={len(nodes)}, "
        f"input_lens: min={min(input_lens)}, max={max(input_lens)}, avg={sum(input_lens)/len(input_lens):.1f}, "
        f"decode_lens: min={min(decode_lens)}, max={max(decode_lens)}, avg={sum(decode_lens)/len(decode_lens):.1f}"
    )

    data = {
        "input_ids": [node["input_ids"] for node in nodes],
        "sampling_params": [
            {"max_new_tokens": node["decode_len"], "temperature": 0} for node in nodes
        ],
    }

    res = requests.post(base_url + "/generate", json=data)
    assert res.status_code == 200

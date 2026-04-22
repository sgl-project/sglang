import random

import requests


def gen_radix_tree(num_nodes=400, chunk_len=256):
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
    data = {
        "input_ids": [node["input_ids"] for node in nodes],
        "sampling_params": [
            {"max_new_tokens": node["decode_len"], "temperature": 0} for node in nodes
        ],
    }

    res = requests.post(base_url + "/generate", json=data)
    assert res.status_code == 200

import os
import random
import unittest

import requests

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)


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


def run_test(base_url, nodes):
    data = {
        "input_ids": [node["input_ids"] for node in nodes],
        "sampling_params": [
            {"max_new_tokens": node["decode_len"], "temperature": 0} for node in nodes
        ],
    }

    res = requests.post(base_url + "/generate", json=data)
    assert res.status_code == 200


class TestRadixCacheFCFS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "fcfs",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        nodes = gen_radix_tree()
        run_test(self.base_url, nodes)


class TestRadixCacheLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "lpm",
            ],
        )


class TestRadixCacheNonOverlapLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--chunked-prefill-size",
                "128",
                "--max-total-tokens",
                "20000",
                "--schedule-policy",
                "lpm",
            ],
        )


if __name__ == "__main__":
    os.environ["SGLANG_TEST_RETRACT"] = "true"
    unittest.main()

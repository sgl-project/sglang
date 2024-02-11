"""
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
"""

import json
import unittest

from sglang.test.test_programs import (
    test_decode_int,
    test_decode_json_regex,
    test_expert_answer,
    test_few_shot_qa,
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_react,
    test_regex,
    test_select,
    test_stream,
    test_tool_use,
)

import sglang as sgl


class TestSRTBackend(unittest.TestCase):
    backend = None

    def setUp(self):
        cls = type(self)

        if cls.backend is None:
            cls.backend = sgl.RuntimeEndpoint(base_url="http://localhost:30000")
            sgl.set_default_backend(cls.backend)

    def test_few_shot_qa(self):
        test_few_shot_qa()

    def test_mt_bench(self):
        test_mt_bench()

    def test_select(self):
        test_select(check_answer=False)

    def test_decode_int(self):
        test_decode_int()

    def test_decode_json_regex(self):
        test_decode_json_regex()

    def test_expert_answer(self):
        test_expert_answer()

    def test_tool_use(self):
        test_tool_use()

    def test_parallel_decoding(self):
        test_parallel_decoding()

    def test_stream(self):
        test_stream()

    def test_regex(self):
        test_regex()

    # def test_parallel_encoding(self):
    #     test_parallel_encoding(check_answer=False)


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # from sglang.global_config import global_config

    # global_config.verbosity = 2
    # t = TestSRTBackend()
    # t.setUp()
    # t.test_regex()

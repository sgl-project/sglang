import unittest

import sglang as sgl
from sglang.test.test_programs import (
    test_decode_int,
    test_decode_json_regex,
    test_dtype_gen,
    test_expert_answer,
    test_few_shot_qa,
    test_hellaswag_select,
    test_mt_bench,
    test_parallel_decoding,
    test_regex,
    test_select,
    test_stream,
    test_tool_use,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestSRTBackend(unittest.TestCase):
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.backend = sgl.Runtime(model_path=DEFAULT_MODEL_NAME_FOR_TEST)
        sgl.set_default_backend(cls.backend)

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown()

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

    def test_dtype_gen(self):
        test_dtype_gen()

    def test_hellaswag_select(self):
        # Run twice to capture more bugs
        for _ in range(2):
            accuracy, latency = test_hellaswag_select()
            assert accuracy > 0.71


if __name__ == "__main__":
    unittest.main()

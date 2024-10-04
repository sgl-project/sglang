import unittest

import sglang as sgl
from sglang.test.test_programs import (
    test_decode_int,
    test_decode_json_regex,
    test_dtype_gen,
    test_expert_answer,
    test_few_shot_qa,
    test_gen_min_new_tokens,
    test_hellaswag_select,
    test_mt_bench,
    test_parallel_decoding,
    test_regex,
    test_select,
    test_stream,
    test_tool_use,
)

# from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

DEFAULT_MODEL_NAME_FOR_TEST = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"


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
            assert accuracy > 0.71, f"{accuracy=}"

    # TODO (ByronHsu): intentionally add "0" in the test name to ensure the alpha-numeric order is ahead, so this test is run before test_decode_int
    # See issue at https://github.com/sgl-project/sglang/issues/1575
    def test_0_gen_min_new_tokens(self):
        test_gen_min_new_tokens(DEFAULT_MODEL_NAME_FOR_TEST)


if __name__ == "__main__":
    unittest.main()

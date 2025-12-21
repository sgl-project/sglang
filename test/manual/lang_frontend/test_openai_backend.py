import unittest

from sglang import OpenAI, set_default_backend
from sglang.test.test_programs import (
    test_chat_completion_speculative,
    test_completion_speculative,
    test_decode_int,
    test_decode_json,
    test_expert_answer,
    test_few_shot_qa,
    test_image_qa,
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_react,
    test_select,
    test_stream,
    test_tool_use,
)
from sglang.test.test_utils import CustomTestCase


class TestOpenAIBackend(CustomTestCase):
    instruct_backend = None
    chat_backend = None
    chat_vision_backend = None

    @classmethod
    def setUpClass(cls):
        cls.instruct_backend = OpenAI("gpt-3.5-turbo-instruct")
        cls.chat_backend = OpenAI("gpt-3.5-turbo")
        cls.chat_vision_backend = OpenAI("gpt-4-turbo")

    def test_few_shot_qa(self):
        set_default_backend(self.instruct_backend)
        test_few_shot_qa()

    def test_mt_bench(self):
        set_default_backend(self.chat_backend)
        test_mt_bench()

    def test_select(self):
        set_default_backend(self.instruct_backend)
        test_select(check_answer=True)

    def test_decode_int(self):
        set_default_backend(self.instruct_backend)
        test_decode_int()

    def test_decode_json(self):
        set_default_backend(self.instruct_backend)
        test_decode_json()

    def test_expert_answer(self):
        set_default_backend(self.instruct_backend)
        test_expert_answer()

    def test_tool_use(self):
        set_default_backend(self.instruct_backend)
        test_tool_use()

    def test_react(self):
        set_default_backend(self.instruct_backend)
        test_react()

    def test_parallel_decoding(self):
        set_default_backend(self.instruct_backend)
        test_parallel_decoding()

    def test_parallel_encoding(self):
        set_default_backend(self.instruct_backend)
        test_parallel_encoding()

    def test_image_qa(self):
        set_default_backend(self.chat_vision_backend)
        test_image_qa()

    def test_stream(self):
        set_default_backend(self.instruct_backend)
        test_stream()

    def test_completion_speculative(self):
        set_default_backend(self.instruct_backend)
        test_completion_speculative()

    def test_chat_completion_speculative(self):
        set_default_backend(self.chat_backend)
        test_chat_completion_speculative()


if __name__ == "__main__":
    unittest.main()

import unittest
import os
import openai
from sglang import OpenAI, set_default_backend
from sglang.test.test_programs import (
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
    #test_completion_speculative,
    #test_chat_completion_speculative
)


class TestOpenAIBackend(unittest.TestCase):
    backend = None
    chat_backend = None
    chat_vision_backend = None

    def setUp(self):
        cls = type(self)

        if cls.backend is None:
            api_key = os.getenv("OPENAI_API_KEY", "ACTUAL_OPENAI_API_KEY_HERE")
            openai.api_key = api_key
            cls.backend = OpenAI(api_key=api_key,model_name="gpt-3.5-turbo-instruct")
            cls.chat_backend = OpenAI(api_key=api_key,model_name="gpt-3.5-turbo")
            cls.chat_vision_backend = OpenAI(api_key=api_key,model_name="gpt-4-turbo")

    def test_few_shot_qa(self):
        set_default_backend(self.backend)
        test_few_shot_qa()

    def test_mt_bench(self):
        set_default_backend(self.chat_backend)
        test_mt_bench()

    def test_select(self):
        set_default_backend(self.backend)
        test_select(check_answer=True)

    def test_decode_int(self):
        set_default_backend(self.backend)
        test_decode_int()

    def test_decode_json(self):
        set_default_backend(self.backend)
        test_decode_json()

    def test_expert_answer(self):
        set_default_backend(self.backend)
        test_expert_answer()

    def test_tool_use(self):
        set_default_backend(self.backend)
        test_tool_use()

    def test_react(self):
        set_default_backend(self.backend)
        test_react()

    def test_parallel_decoding(self):
        set_default_backend(self.backend)
        test_parallel_decoding()

    def test_parallel_encoding(self):
        set_default_backend(self.backend)
        test_parallel_encoding()

    def test_image_qa(self):
        set_default_backend(self.chat_vision_backend)
        test_image_qa()

    def test_stream(self):
        set_default_backend(self.backend)
        test_stream()
   """
    def test_completion_speculative(self):
        set_default_backend(self.backend)
        test_completion_speculative()

    def test_chat_completion_speculative(self):
        set_default_backend(self.chat_backend)
        test_chat_completion_speculative()
    """

if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # from sglang.global_config import global_config

    # global_config.verbosity = 2
    # t = TestOpenAIBackend()
    # t.setUp()
    # t.test_chat_completion_speculative()
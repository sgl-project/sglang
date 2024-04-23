import unittest

from sglang.test.test_programs import (
    test_expert_answer,
    test_few_shot_qa,
    test_image_qa,
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_stream,
)

from sglang import VertexAI, set_default_backend


class TestVertexAIBackend(unittest.TestCase):
    backend = None
    chat_backend = None
    chat_vision_backend = None

    def setUp(self):
        cls = type(self)

        if cls.backend is None:
            cls.backend = VertexAI("gemini-pro")
            cls.chat_backend = VertexAI("gemini-pro")
            cls.chat_vision_backend = VertexAI("gemini-pro-vision")

    def test_few_shot_qa(self):
        set_default_backend(self.backend)
        test_few_shot_qa()

    def test_mt_bench(self):
        set_default_backend(self.chat_backend)
        test_mt_bench()

    def test_expert_answer(self):
        set_default_backend(self.backend)
        test_expert_answer()

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


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # from sglang.global_config import global_config

    # global_config.verbosity = 2
    # t = TestVertexAIBackend()
    # t.setUp()
    # t.test_stream()

import unittest

from sglang import VertexAI, set_default_backend
from sglang.test.test_programs import (
    test_expert_answer,
    test_few_shot_qa,
    test_image_qa,
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_stream,
)
from sglang.test.test_utils import CustomTestCase


class TestVertexAIBackend(CustomTestCase):
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.backend = VertexAI("gemini-1.5-pro-001")

    def test_few_shot_qa(self):
        set_default_backend(self.backend)
        test_few_shot_qa()

    def test_mt_bench(self):
        set_default_backend(self.backend)
        test_mt_bench()

    def test_expert_answer(self):
        set_default_backend(self.backend)
        test_expert_answer(check_answer=False)

    def test_parallel_decoding(self):
        set_default_backend(self.backend)
        test_parallel_decoding()

    def test_parallel_encoding(self):
        set_default_backend(self.backend)
        test_parallel_encoding()

    def test_image_qa(self):
        set_default_backend(self.backend)
        test_image_qa()

    def test_stream(self):
        set_default_backend(self.backend)
        test_stream()


if __name__ == "__main__":
    unittest.main()

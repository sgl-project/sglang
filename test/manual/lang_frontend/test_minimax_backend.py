import unittest

from sglang import MiniMax, set_default_backend
from sglang.test.test_programs import (
    test_expert_answer,
    test_mt_bench,
    test_stream,
)
from sglang.test.test_utils import CustomTestCase


class TestMiniMaxBackend(CustomTestCase):
    chat_backend = None

    @classmethod
    def setUpClass(cls):
        cls.chat_backend = MiniMax("MiniMax-M2.5")

    def test_mt_bench(self):
        set_default_backend(self.chat_backend)
        test_mt_bench()

    def test_stream(self):
        set_default_backend(self.chat_backend)
        test_stream()

    def test_expert_answer(self):
        set_default_backend(self.chat_backend)
        test_expert_answer()


if __name__ == "__main__":
    unittest.main()

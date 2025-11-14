import unittest

from sglang import LiteLLM, set_default_backend
from sglang.test.test_programs import test_mt_bench, test_stream
from sglang.test.test_utils import CustomTestCase


class TestAnthropicBackend(CustomTestCase):
    chat_backend = None

    @classmethod
    def setUpClass(cls):
        cls.chat_backend = LiteLLM("gpt-3.5-turbo")
        set_default_backend(cls.chat_backend)

    def test_mt_bench(self):
        test_mt_bench()

    def test_stream(self):
        test_stream()


if __name__ == "__main__":
    unittest.main()

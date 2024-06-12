import json
import unittest

from sglang import LiteLLM, set_default_backend
from sglang.test.test_programs import test_mt_bench, test_stream


class TestAnthropicBackend(unittest.TestCase):
    backend = None
    chat_backend = None

    def setUp(self):
        cls = type(self)

        if cls.backend is None:
            cls.backend = LiteLLM("gpt-3.5-turbo")
            set_default_backend(cls.backend)

    def test_mt_bench(self):
        test_mt_bench()

    def test_stream(self):
        test_stream()


if __name__ == "__main__":
    unittest.main(warnings="ignore")

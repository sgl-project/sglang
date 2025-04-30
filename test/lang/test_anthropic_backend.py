import json
import unittest

from sglang import Anthropic, set_default_backend
from sglang.test.test_programs import test_mt_bench, test_stream
from sglang.test.test_utils import CustomTestCase


class TestAnthropicBackend(CustomTestCase):
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.backend = Anthropic("claude-3-haiku-20240307")
        set_default_backend(cls.backend)

    def test_mt_bench(self):
        test_mt_bench()

    def test_stream(self):
        test_stream()


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from sglang.test.test_programs import test_mt_bench, test_stream

from sglang import Anthropic, set_default_backend


class TestAnthropicBackend(unittest.TestCase):
    backend = None
    chat_backend = None

    def setUp(self):
        cls = type(self)

        if cls.backend is None:
            cls.backend = Anthropic("claude-3-haiku-20240307")
            set_default_backend(cls.backend)

    def test_mt_bench(self):
        test_mt_bench()

    def test_stream(self):
        test_stream()


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # from sglang.global_config import global_config

    # global_config.verbosity = 2
    # t = TestAnthropicBackend()
    # t.setUp()
    # t.test_mt_bench()

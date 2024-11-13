import os
import unittest

from sglang import Shortfin, set_default_backend
from sglang.test.test_programs import test_mt_bench, test_stream


class TestShortfinBackend(unittest.TestCase):
    chat_backend = None

    @classmethod
    def setUpClass(cls):
        base_url = os.environ["SHORTFIN_BASE_URL"]
        cls.chat_backend = Shortfin(base_url=base_url)
        set_default_backend(cls.chat_backend)

    def test_mt_bench(self):
        test_mt_bench()

    def test_stream(self):
        test_stream()


if __name__ == "__main__":
    unittest.main()

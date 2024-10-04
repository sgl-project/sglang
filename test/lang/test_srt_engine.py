import unittest

import sglang as sgl
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestSRTBackend(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(model_path=DEFAULT_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_generate(self):
        prompts = [
            "The capital of China is",
            "The square root of 144 is",
        ]

        outputs = self.engine.generate(prompts, {"max_new_tokens": 128})

        assert "Beijing" in outputs[0]["text"]
        assert "12" in outputs[1]["text"]


if __name__ == "__main__":
    unittest.main()

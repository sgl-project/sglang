import unittest

from test_ascend_llm_models import TestMistral


class TestLing(TestMistral):
    model = "/root/.cache/modelscope/hub/models/inclusionAI/Ling-lite"


if __name__ == "__main__":
    unittest.main()

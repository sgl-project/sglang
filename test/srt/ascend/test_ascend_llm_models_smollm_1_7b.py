import unittest

from test_ascend_llm_models import TestMistral


class TestSmolLM(TestMistral):
    model = "/root/.cache/modelscope/hub/models/HuggingFaceTB/SmolLM-1.7B"
    accuracy = 0.00


if __name__ == "__main__":
    unittest.main()

import unittest

from test_ascend_llm_models import TestMistral


class TestGEMMA_3_1B_IT(TestMistral):
    model = "/root/.cache/modelscope/hub/models/LLM-Research/gemma-3-1b-it"
    accuracy = -1


if __name__ == "__main__":
    unittest.main()

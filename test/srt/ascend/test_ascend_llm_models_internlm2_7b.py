import unittest

from test_ascend_llm_models_2 import TestLlama_2_7B


class TestInternlm2_7b(TestLlama_2_7B):
    model = "/root/.cache/modelscope/hub/models/Shanghai_AI_Laboratory/internlm2-7b"
    accuracy = 0.05


if __name__ == "__main__":
    unittest.main()
